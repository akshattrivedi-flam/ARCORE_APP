import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2
import json
import os
import numpy as np
import copy

# --- CONFIGURATION ---
# Assumes running from "scripts/" folder or root
# If running depending on CWD, we might need absolute paths or relative adjustments.
# New Root: ../data
import datetime

# --- CONFIGURATION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(current_dir, "../data/raw")

# Create Unique Run Directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_DIR = os.path.join(current_dir, f"../models/run_tracker_{timestamp}")
os.makedirs(os.path.join(MODEL_DIR, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(MODEL_DIR, "final"), exist_ok=True)

print(f"Training Run ID: {timestamp}")
print(f"Saving models to: {MODEL_DIR}")

BATCH_SIZE = 64 
LR = 0.0005
EPOCHS = 100 
IMG_SIZE = 224 # Objectron Standard
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CUSTOM AUGMENTATION ---
class RandomCutout(object):
    """Randomly mask out a rectangular chunk of the image to force structure learning."""
    def __init__(self, p=0.5, scale=(0.02, 0.2)):
        self.p = p
        self.scale = scale

    def __call__(self, img):
        if torch.rand(1) < self.p:
            c, h, w = img.shape
            cut_h = int(h * np.random.uniform(*self.scale))
            cut_w = int(w * np.random.uniform(*self.scale))
            y = np.random.randint(0, h - cut_h)
            x = np.random.randint(0, w - cut_w)
            img[:, y:y+cut_h, x:x+cut_w] = 0.0 
        return img

# --- DATASET CLASS ---
class CokeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = [] # [(img_path, kpts_2d, kpts_3d, label)]

        # 1. Load POSITIVES (Red)
        self._load_category("red", label=1.0)

        # 2. Load NEGATIVES (Blue, Silver)
        self._load_category("blue", label=0.0)
        self._load_category("silver", label=0.0)
        
        print(f"Total Samples: {len(self.samples)}")
        
    def _load_category(self, category, label):
        ann_dir = os.path.join(self.root_dir, "annotations", category)
        frames_dir = os.path.join(self.root_dir, "frames", category)
        
        if not os.path.exists(ann_dir):
            print(f"Skipping missing category annotations: {ann_dir}")
            return

        for ann_file in os.listdir(ann_dir):
            if not ann_file.endswith(".json"): continue
            video_id = ann_file.replace("_annotations.json", "")
            full_ann_path = os.path.join(ann_dir, ann_file)
            
            with open(full_ann_path, 'r') as f:
                anns = json.load(f)
                
            for ann in anns:
                img_name = ann['image']
                img_path = os.path.join(frames_dir, video_id, img_name)
                
                if not os.path.exists(img_path): continue
                
                if label == 1.0:
                    # 2D Keypoints (x, y)
                    raw_kpts_2d = ann['keypoints_2d']
                    kpts_2d = []
                    for kp in raw_kpts_2d:
                        kpts_2d.extend([kp[0], kp[1]])
                    kpts_2d = np.array(kpts_2d, dtype=np.float32)
                    
                    # 3D Keypoints (x, y, z)
                    # Note: These are in Camera Coordinate System usually (metric)
                    raw_kpts_3d = ann['keypoints_3d']
                    kpts_3d = []
                    for kp in raw_kpts_3d:
                        kpts_3d.extend(kp) # Flatten [x, y, z]
                    kpts_3d = np.array(kpts_3d, dtype=np.float32)
                    
                else:
                    kpts_2d = np.zeros(18, dtype=np.float32)
                    kpts_3d = np.zeros(27, dtype=np.float32)

                self.samples.append((img_path, kpts_2d, kpts_3d, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, kpts_2d, kpts_3d, label = self.samples[idx]
        
        # Load Image
        img = cv2.imread(img_path)
        if img is None: return self.__getitem__((idx + 1) % len(self))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # ROTATION: Landscape -> Portrait (90 CW)
        # img: HxW -> WxH
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h_new, w_new = img.shape[:2]
        
        # Adjust 2D Keypoints for Rotation (x -> 1-y, y -> x)
        if label == 1.0:
            kpts_2d_rot = []
            for i in range(0, 18, 2):
                x = kpts_2d[i]
                y = kpts_2d[i+1]
                xn = 1.0 - y
                yn = x
                kpts_2d_rot.extend([xn, yn])
            kpts_2d = np.array(kpts_2d_rot, dtype=np.float32)
            
            # Adjust 3D Keypoints for Rotation??
            # 3D Points are in Camera Frame. 
            # If we rotate the IMAGE, the "Camera" implies a roll of 90 degrees.
            # We assume the 3D points are RELATIVE to the object center for regression?
            # NO, Objectron regresses 3D points in the *Camera Coordinate System* of the crop.
            # This is complex. 
            # SIMPLE APPROX: We regress them as is, but we must account that X_cam and Y_cam swap axes in screen space.
            # For now, let's keep 3D points AS IS (Metric in Camera Frame)
            # The network will learn the mapping from Rotated Image -> Original Camera Frame.
            pass

        # --- CROP LOGIC ---
        if label == 1.0:
            # Box from 2D points
            xs = kpts_2d[0::2]
            ys = kpts_2d[1::2]
            min_x, max_x = np.min(xs), np.max(xs)
            min_y, max_y = np.min(ys), np.max(ys)
            
            box_w = max_x - min_x
            box_h = max_y - min_y
            center_x = (min_x + max_x) / 2.0
            center_y = (min_y + max_y) / 2.0
            
            scale = np.random.uniform(1.5, 2.0) 
            size = max(box_w * w_new, box_h * h_new) * scale
            
            jitter_x = np.random.uniform(-0.1, 0.1) * size
            jitter_y = np.random.uniform(-0.1, 0.1) * size
            
            cx = int(center_x * w_new + jitter_x)
            cy = int(center_y * h_new + jitter_y)
            crop_size = int(size)
            
        else:
            crop_size = int(min(h_new, w_new) * np.random.uniform(0.5, 0.8))
            cx = np.random.randint(crop_size//2, w_new - crop_size//2)
            cy = np.random.randint(crop_size//2, h_new - crop_size//2)
            
        # Perform Crop
        x1 = cx - crop_size // 2
        y1 = cy - crop_size // 2
        x2 = x1 + crop_size
        y2 = y1 + crop_size
        
        pad_l = max(0, -x1); pad_t = max(0, -y1); pad_r = max(0, x2 - w_new); pad_b = max(0, y2 - h_new)
        bbox_x1 = max(0, x1); bbox_y1 = max(0, y1); bbox_x2 = min(w_new, x2); bbox_y2 = min(h_new, y2)
        
        cropped_img = img[bbox_y1:bbox_y2, bbox_x1:bbox_x2]
        
        if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
            cropped_img = cv2.copyMakeBorder(cropped_img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0,0,0))
            
        cropped_img = cv2.resize(cropped_img, (IMG_SIZE, IMG_SIZE))
        
        # Transform 2D Keypoints to Crop Space
        if label == 1.0:
            kpts_crop = []
            for i in range(0, 18, 2):
                gx = kpts_2d[i] * w_new
                gy = kpts_2d[i+1] * h_new
                nx = (gx - x1) / crop_size
                ny = (gy - y1) / crop_size
                kpts_crop.extend([nx, ny])
            kpts_2d = np.array(kpts_crop, dtype=np.float32)
            
            # Transform 3D Keypoints? 
            # In Objectron, 3D keypoints are usually regressed RELATIVE to the camera center of the crop.
            # But the 'Camera Intrinsics' change when we crop.
            # We will rely on the Network to learn "3D offsets from the crop center".
            # Normalization helps: Centroid should be ~0.
            pass
        
        pil_img = transforms.ToPILImage()(cropped_img)
        if self.transform:
            pil_img = self.transform(pil_img)
            
        return pil_img, torch.tensor(kpts_2d), torch.tensor(kpts_3d), torch.tensor([label], dtype=torch.float32)

# --- MODEL ---
class CokeTrackerReduced(nn.Module):
    def __init__(self):
        super(CokeTrackerReduced, self).__init__()
        self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        num_features = self.backbone.classifier[0].in_features
        
        # 1. 2D Regression Head (18 coordinates) - Auxiliary
        self.regressor_2d = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.Hardswish(),
            nn.Dropout(p=0.2), 
            nn.Linear(512, 18), 
            nn.Sigmoid() # 0-1 (Crop Relative)
        )
        
        # 2. 3D Regression Head (27 coordinates) - Primary
        # 9 points * 3 (x,y,z)
        self.regressor_3d = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.Hardswish(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 27) # No Sigmoid! Metric units (m) can be negative/unbounded
        )
        
        # 3. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.Hardswish(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1)
        )
        self.backbone.classifier = nn.Identity()

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        kpts_2d = self.regressor_2d(x)
        kpts_3d = self.regressor_3d(x)
        conf = self.classifier(x)
        
        return kpts_2d, kpts_3d, conf

# --- TRAINING ---
def train_model():
    print(f"Using device: {DEVICE}")

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.15),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.5)),
        transforms.ToTensor(),
        RandomCutout(p=0.5), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = CokeDataset(root_dir=DATA_ROOT, transform=train_transform)
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    model = CokeTrackerReduced().to(DEVICE)
    
    criterion_reg = nn.L1Loss(reduction='none') 
    criterion_cls = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    START_EPOCH = 0
    # No resume logic here to ensure clean slate for new architecture
    print("Starting FRESH training with 3D Head...")

    print(f"Starting training on {len(full_dataset)} crops...")
    
    for epoch in range(START_EPOCH, EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, kpts_2d, kpts_3d, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            kpts_2d = kpts_2d.to(DEVICE)
            kpts_3d = kpts_3d.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            pred_2d, pred_3d, pred_conf = model(images)
            
            # Loss Calculation
            # 1. Classification
            loss_cls = criterion_cls(pred_conf, labels)
            
            # 2. 2D Regression (Masked by label)
            l2d = criterion_reg(pred_2d, kpts_2d)
            loss_2d = (l2d.mean(dim=1, keepdim=True) * labels).mean()
            
            # 3. 3D Regression (Masked by label)
            l3d = criterion_reg(pred_3d, kpts_3d)
            loss_3d = (l3d.mean(dim=1, keepdim=True) * labels).mean()
            
            # Total Loss Weights
            # 3D is primary goal. 2D is helper.
            loss = loss_cls + (10.0 * loss_2d) + (50.0 * loss_3d)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * images.size(0)
            
            if batch_idx % 10 == 0:
                 print(f"Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] L_2D: {loss_2d.item():.4f} L_3D: {loss_3d.item():.4f}")
            
        scheduler.step()
        epoch_loss /= len(full_dataset)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Total Loss: {epoch_loss:.6f}")
        
        if (epoch + 1) % 5 == 0:
             torch.save(model.state_dict(), os.path.join(MODEL_DIR, "checkpoints", f"coke_tracker_3d_epoch_{epoch+1}.pth"))

    save_path = os.path.join(MODEL_DIR, "final", "model_3d.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_model()

