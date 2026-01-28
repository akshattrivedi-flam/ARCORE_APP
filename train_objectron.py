import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import gc

# --- CONFIGURATION ---
DATA_DIR = "/home/user/Desktop/ARCORE_APP/seq_120737245"
JSON_FILE = os.path.join(DATA_DIR, "annotations.json")
BATCH_SIZE = 32
LEARNING_RATE = 2e-4 
EPOCHS = 1000
IMAGE_SIZE = 224
MODEL_PATH = "bottle_objectron_overfit.pth"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- SMART ALIGNMENT LOGIC ---
def align_keypoints(kpts, img_w, img_h, json_w=640, json_h=480):
    """
    Applies the necessary Rotation and Center Crop adjustments to map
    Sensor Coordinates (JSON) to Screen Coordinates (Image).
    
    1. Rotation: Sensor (Landscape) -> Screen (Portrait) usually implies -90 or +90 rotation.
       User Diagnostics confirmed CW (x' = 1-y, y' = x) is closest.
    
    2. Aspect Ratio / Crop:
       Sensor (Rotated): 480 x 640 (Ratio 0.75)
       Screen: 1080 x 2023 (Ratio 0.53)
       
       To fit 480x640 content into 1080x2023 via CENTER_CROP (ARCore default?):
       It scales the Rotated Sensor to COVER the Screen.
       
       Scale based on Height? 640 -> 2023 (Scale 3.16)
       Then Width 480 * 3.16 = 1516.
       Screen Width is 1080.
       1516 > 1080. So it chops off sides (Left/Right).
       
       Or Scale based on Width? 480 -> 1080 (Scale 2.25)
       Then Height 640 * 2.25 = 1440.
       Screen Height is 2023.
       1440 < 2023. Black bars?
       Typically ARFullScreen fills the screen. So it uses the MAX scale (Scale based on Height).
       
       So:
       Scale = 2023 / 640 = ~3.1609
       New Sensor W = 480 * Scale = 1517.2
       Crop X = (1517.2 - 1080) / 2 = 218.6 px.
       
       We need to map Normalized Sensor Coords -> Normalized Screen Coords.
    """
    
    # 1. Apply CW Rotation (Sensor -> Portrait)
    # kpts is [N, 2]. x=0..1, y=0..1
    # CW: x' = 1 - y, y' = x
    kpts_rot = np.zeros_like(kpts)
    kpts_rot[:, 0] = 1.0 - kpts[:, 1]
    kpts_rot[:, 1] = kpts[:, 0]
    
    # 2. Dimensions after Rotation
    sens_w_rot = json_h # 480
    sens_h_rot = json_w # 640
    
    # 3. Calculate Scale (Cover Mode)
    scale_w = img_w / sens_w_rot
    scale_h = img_h / sens_h_rot
    scale = max(scale_w, scale_h) # Max to cover
    
    # 4. Calculate projected dimensions
    proj_w = sens_w_rot * scale
    proj_h = sens_h_rot * scale
    
    # 5. Calculate Crop Offsets (Centered)
    off_x = (proj_w - img_w) / 2.0
    off_y = (proj_h - img_h) / 2.0 # Should be 0 if H scaled
    
    # 6. Transform Normalized Coords
    # SensorNorm -> Pixels(Projected) -> Minus Offset -> Pixels(Screen) -> ScreenNorm
    
    kpts_screen = np.zeros_like(kpts_rot)
    
    # X transform
    # x_px = x_norm * proj_w
    # x_screen_px = x_px - off_x
    # x_screen_norm = x_screen_px / img_w
    kpts_screen[:, 0] = ((kpts_rot[:, 0] * proj_w) - off_x) / img_w
    
    # Y transform
    kpts_screen[:, 1] = ((kpts_rot[:, 1] * proj_h) - off_y) / img_h
    
    return kpts_screen


# --- DATASET CLASS (VRAM CACHED) ---
class ObjectronBottleDataset(Dataset):
    def __init__(self, json_path, img_dir, transform=None):
        print(f"Loading dataset from {json_path}...")
        with open(json_path, 'r') as f:
            self.annotations = json.load(f)
        self.img_dir = img_dir
        
        self.images = []
        self.keypoints = []
        
        print(f"Propagating Data to {DEVICE}...")
        
        # Get Frame 0 dimensions once for alignment logic
        # Assuming all frames are same size
        first_img_path = os.path.join(self.img_dir, self.annotations[0]['image'])
        with Image.open(first_img_path) as tmp:
            real_w, real_h = tmp.size
        print(f"Detected Image Resolution: {real_w}x{real_h}")
        
        count = 0
        for entry in self.annotations:
            img_name = entry['image']
            img_path = os.path.join(self.img_dir, img_name)
            
            try:
                with Image.open(img_path) as img:
                    image = img.convert('RGB')
                    
                if transform:
                    image = transform(image)
                self.images.append(image.to(DEVICE))
                
                # --- KEYPOINT ALIGNMENT ---
                kpts_raw = np.array(entry['keypoints_2d'])
                # Extract X,Y only (ignore depth for alignment logic, though depth is just pass through)
                kpts_xy = kpts_raw[:, :2]
                
                # Align
                kpts_aligned = align_keypoints(kpts_xy, real_w, real_h, 
                                               entry['camera_intrinsics']['image_width'], 
                                               entry['camera_intrinsics']['image_height'])
                
                # Flatten
                kpts_flat = kpts_aligned.flatten()
                
                self.keypoints.append(torch.tensor(kpts_flat, dtype=torch.float32).to(DEVICE))
                count += 1
                
            except Exception as e:
                print(f"Warning: Could not load {img_path}: {e}")

        print(f"Successfully cached {count} aligned samples.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.keypoints[idx]

# --- MODEL DEFINITION ---
class ObjectronMobileNetV3(nn.Module):
    def __init__(self, num_keypoints=9):
        super(ObjectronMobileNetV3, self).__init__()
        # MobileNetV3 Large
        self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(1024, 512),
            nn.Hardswish(inplace=True),
            nn.Linear(512, num_keypoints * 2),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.backbone(x)

# --- TRAINING LOOP ---
def train():
    print(f"Starting Aligned Training on {torch.cuda.get_device_name(0)}...")
    
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ObjectronBottleDataset(JSON_FILE, DATA_DIR, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = ObjectronMobileNetV3().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(reduction='none')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Fresh Start
    model.train()
    best_loss = 1e9

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for images, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            
            # Weighted Loss
            weights_mask = torch.ones_like(targets)
            weights_mask[:, :2] *= 10.0 # Center
            weights_mask[:, 2:18] *= 2.0 # Corners
            
            loss = (weights_mask * criterion(outputs, targets)).mean()
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloader)
        scheduler.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), MODEL_PATH)
        
        if (epoch + 1) % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    print(f"Training complete. Model saved as {MODEL_PATH}")

if __name__ == "__main__":
    train()
