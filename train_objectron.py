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
BASE_DIRS = [
    "/home/user/Desktop/ARCORE_APP/red",
    "/home/user/Desktop/ARCORE_APP/blue",
    "/home/user/Desktop/ARCORE_APP/silver"
]
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 1000
IMAGE_SIZE = 256
MODEL_PATH = "objectron_cans_final.pth"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- SMART ALIGNMENT LOGIC ---
def align_keypoints(kpts, img_w, img_h, json_w=640, json_h=480):
    # (Kept identical to original as it is the correct sensor-to-screen mapping)
    kpts_rot = np.zeros_like(kpts)
    kpts_rot[:, 0] = 1.0 - kpts[:, 1]
    kpts_rot[:, 1] = kpts[:, 0]
    sens_w_rot, sens_h_rot = json_h, json_w
    scale = max(img_w / sens_w_rot, img_h / sens_h_rot)
    proj_w, proj_h = sens_w_rot * scale, sens_h_rot * scale
    off_x, off_y = (proj_w - img_w) / 2.0, (proj_h - img_h) / 2.0
    kpts_screen = np.zeros_like(kpts_rot)
    kpts_screen[:, 0] = ((kpts_rot[:, 0] * proj_w) - off_x) / img_w
    kpts_screen[:, 1] = ((kpts_rot[:, 1] * proj_h) - off_y) / img_h
    return kpts_screen

# --- DATASET CLASS (MULTI-SEQUENCE LAZY LOADING) ---
class MultiSequenceObjectronDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        self.samples = []
        self.transform = transform
        
        print("Scanning for sequences...")
        for root in root_dirs:
            if not os.path.exists(root): continue
            
            # Find all video_* subdirectories (e.g., video_01_red)
            for seq_name in os.listdir(root):
                seq_path = os.path.join(root, seq_name)
                json_path = os.path.join(seq_path, "annotations.json")
                
                if os.path.isdir(seq_path) and os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        annotations = json.load(f)
                    
                    # Get dimensions for this specific sequence
                    first_img = os.path.join(seq_path, annotations[0]['image'])
                    with Image.open(first_img) as tmp:
                        w, h = tmp.size
                    
                    for ann in annotations:
                        self.samples.append({
                            'img_path': os.path.join(seq_path, ann['image']),
                            'kpts': np.array(ann['keypoints_2d'])[:, :2],
                            'intrinsics': ann['camera_intrinsics'],
                            'real_dim': (w, h)
                        })
        
        print(f"Total frames aggregated across all sequences: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. Load Image
        try:
            with Image.open(sample['img_path']) as img:
                image = img.convert('RGB')
            if self.transform:
                image = self.transform(image)
        except:
            image = torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE))

        # 2. Align Keypoints
        w, h = sample['real_dim']
        kpts_aligned = align_keypoints(
            sample['kpts'], w, h,
            sample['intrinsics']['image_width'], 
            sample['intrinsics']['image_height']
        )
        
        return image, torch.tensor(kpts_aligned.flatten(), dtype=torch.float32)

# --- MODEL DEFINITION ---
class ObjectronMobileNetV3(nn.Module):
    def __init__(self, num_keypoints=9):
        super(ObjectronMobileNetV3, self).__init__()
        # MobileNetV3 Large remains the best balance of speed and pose precision
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
    print(f"Starting Multi-Sequence Cans Training on {torch.cuda.get_device_name(0)}...")
    
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = MultiSequenceObjectronDataset(BASE_DIRS, transform=train_transform)
    if len(dataset) == 0:
        print("Error: No data found in BASE_DIRS. Have you recorded any videos yet?")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)

    model = ObjectronMobileNetV3().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(reduction='none')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    model.train()
    best_loss = 1e9

    try:
        for epoch in range(EPOCHS):
            running_loss = 0.0
            for images, targets in dataloader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                
                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                
                # Weighted Loss: Focus on center and corners
                weights_mask = torch.ones_like(targets)
                weights_mask[:, :2] *= 10.0 # Center
                weights_mask[:, 2:18] *= 2.0 # Corners
                
                loss = (weights_mask * criterion(outputs, targets)).mean()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            avg_loss = running_loss / len(dataloader)
            scheduler.step()
            
            if (epoch + 1) % 1 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(model.state_dict(), MODEL_PATH)
                    
    except Exception as e:
        print(f"Training stop: {e}")

    print(f"Final Model Saved: {MODEL_PATH}")

if __name__ == "__main__":
    train()



