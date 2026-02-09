import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import cv2
import json
import os
import numpy as np
import copy

import datetime

# --- CONFIGURATION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(current_dir, "../data/raw")

# Create Unique Run Directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_DIR = os.path.join(current_dir, f"../models/run_detector_{timestamp}")
os.makedirs(os.path.join(MODEL_DIR, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(MODEL_DIR, "final"), exist_ok=True)

print(f"Detector Run ID: {timestamp}")
print(f"Saving models to: {MODEL_DIR}")

BATCH_SIZE = 16 
LR = 0.001
EPOCHS = 40
IMG_SIZE = 320 # SSD Lite default
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATASET FOR DETECTION ---
# We reuse the logic but now we output (Image, Bounding Box, Label)
class CokeDetectionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = [] 

        # Load ONLY Positives (Red)
        # SSDLite handles negatives implicitly via background class usually, 
        # but pure Object Detection datasets list objects. 
        # Negative frames (Blue/Silver) -> Empty Target.
        
        self._load_category("red", label=1)    # Class 1: Coke
        self._load_category("blue", label=0)   # Class 0: Background (or ignore)
        self._load_category("silver", label=0) # Class 0: Background

        print(f"Total Detection Samples: {len(self.samples)}")

    def _load_category(self, category, label):
        ann_dir = os.path.join(self.root_dir, "annotations", category)
        frames_dir = os.path.join(self.root_dir, "frames", category)
        
        if not os.path.exists(ann_dir): return

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
                
                boxes = []
                if label == 1:
                    # Calculate Bounding Box from 2D Keypoints
                    kpts = ann['keypoints_2d'] # [9, 3] or [9, 2] usually
                    # Extract just X,Y
                    xs = [p[0] for p in kpts]
                    ys = [p[1] for p in kpts]
                    
                    x_min = min(xs)
                    y_min = min(ys)
                    x_max = max(xs)
                    y_max = max(ys)
                    
                    # Store normalized [x1, y1, x2, y2]
                    # We will denormalize to current image size in __getitem__
                    boxes.append([x_min, y_min, x_max, y_max])
                
                self.samples.append((img_path, boxes, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, norm_boxes, label = self.samples[idx]
        
        # Load Image
        img = cv2.imread(img_path)
        if img is None: return self.__getitem__((idx + 1) % len(self))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # ROTATE to Portrait (90 CW) - Critical to match Tracker
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h, w = img.shape[:2]
        
        # Resize to Target Size (320x320)
        # SSDLite Expects resizing usually, but we doing it manually ensures labels match.
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Transform Boxes
        # 1. Rotate Box Coordinates
        # (x,y) -> (1-y, x)
        final_boxes = []
        if label == 1 and len(norm_boxes) > 0:
            for nb in norm_boxes:
                nx1, ny1, nx2, ny2 = nb
                
                # Convert to points to rotate safely
                pts = [(nx1, ny1), (nx2, ny1), (nx2, ny2), (nx1, ny2)]
                rot_xs = []
                rot_ys = []
                for (x,y) in pts:
                    rx = 1.0 - y
                    ry = x
                    rot_xs.append(rx)
                    rot_ys.append(ry)
                
                rx1, ry1 = min(rot_xs), min(rot_ys)
                rx2, ry2 = max(rot_xs), max(rot_ys)
                
                # Scale to IMG_SIZE
                bx1 = rx1 * IMG_SIZE
                by1 = ry1 * IMG_SIZE
                bx2 = rx2 * IMG_SIZE
                by2 = ry2 * IMG_SIZE
                
                # Clip
                bx1 = max(0, min(IMG_SIZE, bx1))
                by1 = max(0, min(IMG_SIZE, by1))
                bx2 = max(0, min(IMG_SIZE, bx2))
                by2 = max(0, min(IMG_SIZE, by2))
                
                # Filter tiny boxes
                if (bx2 - bx1) > 5 and (by2 - by1) > 5:
                    final_boxes.append([bx1, by1, bx2, by2])
        
        # Prepare Targets for PyTorch Detection Models
        target = {}
        if len(final_boxes) > 0:
            target["boxes"] = torch.tensor(final_boxes, dtype=torch.float32)
            target["labels"] = torch.ones((len(final_boxes),), dtype=torch.int64) # Class 1
        else:
            # Negative Image
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            
        # Transform Image
        pil_img = transforms.ToPILImage()(img_resized)
        tensor_img = transforms.ToTensor()(pil_img) # 0-1
        
        return tensor_img, target

def collate_fn(batch):
    return tuple(zip(*batch))

# --- TRAINING ---
def train_detector():
    print(f"Using device: {DEVICE}")
    
    # Model: SSDLite with MobileNetV3 Large Backbone (Pretrained on COCO)
    # We retrain head for 2 classes (Background + Coke)
    model = ssdlite320_mobilenet_v3_large(weights='DEFAULT')
    
    # Replace Head?
    # Actually, SSDLite is complex to replace head manually.
    # Usually safer to fine-tune existing.
    
    model.to(DEVICE)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Dataset
    dataset = CokeDetectionDataset(root_dir=DATA_ROOT, transform=None)
    
    # Split Train/Val?
    # For now use all for training to maximize performance
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=4, collate_fn=collate_fn)
    
    # Resume Logic
    START_EPOCH = 0
    # Check for existing checkpoints
    if os.path.exists(os.path.join(MODEL_DIR, "checkpoints")):
        ckpts = [f for f in os.listdir(os.path.join(MODEL_DIR, "checkpoints")) if "coke_detector" in f]
        if ckpts:
            # Sort by epoch
            ckpts.sort(key=lambda x: int(x.split('_')[-1].replace('.pth','')))
            latest = os.path.join(MODEL_DIR, "checkpoints", ckpts[-1])
            print(f"Resuming Detector from {latest}")
            model.load_state_dict(torch.load(latest))
            START_EPOCH = int(ckpts[-1].split('_')[-1].replace('.pth',''))

    print(f"Starting Detector Training from Epoch {START_EPOCH}...")
    
    for epoch in range(START_EPOCH, EPOCHS):
        model.train()
        epoch_loss = 0
        
        for images, targets in data_loader:
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            
            # Forward pass (Loss is calculated internally by the model if targets provided)
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            
        lr_scheduler.step()
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss/len(data_loader):.4f}")
        
        # Save EVERY epoch
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "checkpoints", f"coke_detector_epoch_{epoch+1}.pth"))
            
    # Save Final
    save_path = os.path.join(MODEL_DIR, "final", "coke_detector.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Detector Model saved to {save_path}")

if __name__ == "__main__":
    train_detector()