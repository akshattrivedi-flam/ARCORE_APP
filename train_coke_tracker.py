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
DATA_DIR = "/home/user/Desktop/ARCORE_APP/video_01_red"
JSON_PATH = os.path.join(DATA_DIR, "annotations.json")
BATCH_SIZE = 16
LR = 0.001
EPOCHS = 80 # Increased for better convergence
IMG_SIZE = 224  # Standard MobileNet Input
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATASET CLASS ---
class CokeDataset(Dataset):
    def __init__(self, json_path, root_dir, transform=None, is_training=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_training = is_training
        
        with open(json_path, 'r') as f:
            self.annotations = json.load(f)
            
        # Filter out frames with low visibility if needed, or valid files
        self.valid_annotations = []
        for ann in self.annotations:
            img_path = os.path.join(root_dir, ann['image'])
            if os.path.exists(img_path):
                self.valid_annotations.append(ann)
            else:
                pass 
                # print(f"Warning: {img_path} missing.")

    def __len__(self):
        return len(self.valid_annotations)

    def __getitem__(self, idx):
        ann = self.valid_annotations[idx]
        img_name = ann['image']
        img_path = os.path.join(self.root_dir, img_name)
        
        # 1. Load Image (OpenCV loads BGR)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. Check Original Dimensions
        h, w, _ = img.shape
        
        # 3. Rotate Image 90 deg Clockwise (Landscape 640x480 -> Portrait 480x640)
        # As per the previous verification step
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        new_h, new_w, _ = img.shape # Should be 640, 480 if original was 480, 640?
        # Wait: cv2.rotate 90 CW:
        # If input is 640(w)x480(h), output is 480(w)x640(h).
        
        # 4. Transform Keypoints (Equation: x_new = 1 - y_old, y_new = x_old)
        raw_kpts = ann['keypoints_2d'] # [x, y, depth] normalized
        transformed_kpts = []
        for kp in raw_kpts:
            x_old, y_old, _ = kp
            x_new = 1.0 - y_old
            y_new = x_old
            transformed_kpts.append(x_new)
            transformed_kpts.append(y_new)
            
        transformed_kpts = np.array(transformed_kpts, dtype=np.float32) # Shape (18,)

        # 5. Image Preprocessing (Resize)
        # We need to resize image to 224x224, so we must scale keypoints?
        # NO. Keypoints are NORMALIZED (0-1), so resizing the image keeps them valid!
        # This is the beauty of normalized coordinates.
        
        pil_img = transforms.ToPILImage()(img)
        
        if self.transform:
            pil_img = self.transform(pil_img)
            
        return pil_img, torch.tensor(transformed_kpts)

# --- MODEL ---
class CokeTrackerReduced(nn.Module):
    def __init__(self):
        super(CokeTrackerReduced, self).__init__()
        # Use MobileNetV3 Small for speed on mobile devices
        self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        
        # Replace Classifier Head
        # Original classifier: Linear(in_features=576, out_features=1000)
        # We need 18 outputs (9 points * 2 coords)
        
        num_features = self.backbone.classifier[0].in_features 
        # MobileNetV3 classifier structure: Sequential(Linear, Hardswish, Dropout, Linear)
        # We strip it down for regression
        
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.Hardswish(),
            nn.Dropout(p=0.3), # Increased Dropout
            nn.Linear(1024, 18), # Output 18 raw values
            nn.Sigmoid() # Force output to 0-1 range (Normalized coordinates)
        )

    def forward(self, x):
        return self.backbone(x)

# --- TRAINING ---
def train_model():
    print(f"Using device: {DEVICE}")

    # Strong Augmentations to force shape recognition, not pixel memorization
    # UPDATED: Heavier Blur and Jitter for "Real World" robustness
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 3.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CokeDataset(JSON_PATH, DATA_DIR, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    model = CokeTrackerReduced().to(DEVICE)
    
    criterion = nn.MSELoss() # L2 Loss for regression
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    print(f"Starting training on {len(dataset)} frames...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, targets in dataloader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        scheduler.step()
        epoch_loss = running_loss / len(dataset)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            # Checkpoint
            torch.save(model.state_dict(), os.path.join(DATA_DIR, f"coke_tracker_checkpoint_{epoch+1}.pth"))

    # Save Final
    save_path = os.path.join(DATA_DIR, "coke_tracker_mobilenetv3.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_model()
