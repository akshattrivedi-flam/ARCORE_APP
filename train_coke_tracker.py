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
DATA_ROOT = "/home/user/Desktop/ARCORE_APP/DATASET_TRAINING"
BATCH_SIZE = 32
LR = 0.0005
EPOCHS = 60 # Detection usually converges faster than pure regression
IMG_SIZE = 320 
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
        self.samples = [] # [(img_path, keypoints_18, class_label_1)]

        # 1. Load POSITIVES (Red)
        red_dir = os.path.join(root_dir, "red")
        self._load_category(red_dir, label=1.0)

        # 2. Load NEGATIVES (Blue, Silver)
        self._load_category(os.path.join(root_dir, "blue"), label=0.0)
        self._load_category(os.path.join(root_dir, "silver"), label=0.0)
        
        print(f"Total Samples: {len(self.samples)}")
        
    def _load_category(self, category_path, label):
        if not os.path.exists(category_path):
            print(f"Skipping missing category: {category_path}")
            return

        # Walk through video folders
        for video_dir in os.listdir(category_path):
            full_video_path = os.path.join(category_path, video_dir)
            if not os.path.isdir(full_video_path): continue
            
            json_file = os.path.join(full_video_path, "annotations.json")
            if not os.path.exists(json_file): continue
            
            with open(json_file, 'r') as f:
                anns = json.load(f)
                
            for ann in anns:
                img_name = ann['image']
                img_path = os.path.join(full_video_path, img_name)
                
                if not os.path.exists(img_path): continue
                
                # Keypoints (Only matters for Positive)
                if label == 1.0:
                    raw_kpts = ann['keypoints_2d']
                    kpts = []
                    for kp in raw_kpts:
                        # Implicit Rotation Logic (Landscape -> Portrait)
                        # x_new = 1 - y, y_new = x
                        kpts.extend([1.0 - kp[1], kp[0]])
                    kpts = np.array(kpts, dtype=np.float32)
                else:
                    kpts = np.zeros(18, dtype=np.float32) # Dummy for negatives
                
                self.samples.append((img_path, kpts, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, kpts, label = self.samples[idx]
        
        # Load Image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Check integrity
        if img is None:
             # Fail safe
             return self.__getitem__((idx + 1) % len(self))

        # Rotate Image (Landscape -> Portrait) because app logic is Portrait
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        
        pil_img = transforms.ToPILImage()(img)
        
        if self.transform:
            pil_img = self.transform(pil_img)
            
        return pil_img, torch.tensor(kpts), torch.tensor([label], dtype=torch.float32)

# --- MODEL ---
class CokeTrackerReduced(nn.Module):
    def __init__(self):
        super(CokeTrackerReduced, self).__init__()
        # MobileNetV3 Small Backbone
        self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        
        # Replace Classifier
        num_features = self.backbone.classifier[0].in_features
        
        # 1. Regression Head (18 coordinates)
        self.regressor = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.Hardswish(),
            nn.Dropout(p=0.3), 
            nn.Linear(1024, 18), 
            nn.Sigmoid() 
        )
        
        # 2. Classification Head (1 scalar: Is it a Red Coke?)
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.Hardswish(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1)
            # No Sigmoid here if using BCEWithLogitsLoss, for stability
        )
        
        # Remove original classifier to avoid unused params warning (optional)
        self.backbone.classifier = nn.Identity()

    def forward(self, x):
        # Extract features (Need to access features before the original classifier)
        # MobileNetV3 features function returns the Map before pooling usually?
        # Let's check source source. 
        # features(x) -> GAP -> classifier.
        # We can just run features.
        
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        kpts = self.regressor(x)
        conf = self.classifier(x)
        
        return kpts, conf

# --- TRAINING ---
def train_model():
    print(f"Using device: {DEVICE}")

    # FINAL ROBUST TRANSFORMS
    # 1. Geometry Preservation: We resize carefully.
    # 2. Appearance Invariance: ColorJitter + Blur.
    # 3. Structure Forcing: Cutout (prevents memorizing labels).
    
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), # Simple resize is fine as network adapts, but Cutout is key.
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.15),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.5)),
        transforms.ToTensor(),
        RandomCutout(p=0.5), # <--- NEW: Forces model to "imagine" missing parts
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Create Dataset
    full_dataset = CokeDataset(root_dir=DATA_ROOT, transform=train_transform)
    
    # Split (Optional, currently using all for training since we have validation inherent in negatives)
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    model = CokeTrackerReduced().to(DEVICE)
    
    criterion_reg = nn.L1Loss(reduction='none') # We need per-element to mask negatives
    criterion_cls = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # --- RESUME TRAINING ---
    START_EPOCH = 45
    checkpoint_path = os.path.join(DATA_ROOT, f"coke_tracker_checkpoint_{START_EPOCH}.pth")
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from Epoch {START_EPOCH}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        # Adjust scheduler to match epoch
        # (Technically we should step scheduler, but simple starting is fine for fine-tuning)
    else:
        print(f"Checkpoint {checkpoint_path} not found! Starting from scratch.")
        START_EPOCH = 0

    print(f"Starting training on {len(full_dataset)} frames...")
    
    for epoch in range(START_EPOCH, EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, keypoints, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            keypoints = keypoints.to(DEVICE)
            labels = labels.to(DEVICE) # [B, 1]
            
            optimizer.zero_grad()
            
            # Forward
            pred_kpts, pred_conf = model(images)
            
            # 1. Classification Loss (All samples)
            loss_cls = criterion_cls(pred_conf, labels)
            
            # 2. Regression Loss (Only Positive samples)
            loss_reg_raw = criterion_reg(pred_kpts, keypoints) # [B, 18]
            loss_reg = (loss_reg_raw.mean(dim=1, keepdim=True) * labels).mean() # Mask by label (0 or 1)
            
            # Total Loss
            # We explicitly want Regression to be dominant when object is present
            loss = loss_cls + (20.0 * loss_reg) 
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * images.size(0)
            
            if batch_idx % 10 == 0:
                 print(f"Epoch {epoch+1} Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")
            
        scheduler.step()
        epoch_loss /= len(full_dataset)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            # Checkpoint
            torch.save(model.state_dict(), os.path.join(DATA_ROOT, f"coke_tracker_checkpoint_{epoch+1}.pth"))

    # Save Final
    save_path = os.path.join(DATA_ROOT, "coke_tracker_mobilenetv3.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_model()
