import torch
import cv2
import numpy as np
import os
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# --- CONFIG ---
DATA_DIR = "/home/user/Desktop/ARCORE_APP/video_01_red"
COMPOSITED_DIR = os.path.join(DATA_DIR, "composited")
JSON_PATH = os.path.join(DATA_DIR, "annotations.json")
IMG_SIZE = 320

class DebugDataset(Dataset):
    def __init__(self, json_path, root_dir):
        with open(json_path, 'r') as f:
            self.annotations = json.load(f)
        self.valid_annotations = self.annotations

    def __len__(self):
        return len(self.valid_annotations)

    def __getitem__(self, idx):
        ann = self.valid_annotations[idx]
        img_name = ann['image']
        try:
            num = int(img_name.split('_')[1].split('.')[0])
            comp_name = f"frame_{num:06d}.jpg"
        except:
            comp_name = img_name
            
        img_path = os.path.join(COMPOSITED_DIR, comp_name)
        img = cv2.imread(img_path) # BGR
        
        # KEYPOINT LOGIC FROM TRAINER
        raw_kpts = ann['keypoints_2d']
        transformed_kpts = []
        for kp in raw_kpts:
            x_old, y_old, _ = kp
            # Trainer uses: x_new = 1 - y_old, y_new = x_old
            # To match the "Implicit Rotation" of the 640x480 coords to 480x640 image
            x_new = 1.0 - y_old
            y_new = x_old
            transformed_kpts.append([x_new, y_new])
            
        return img, np.array(transformed_kpts), comp_name

def main():
    dataset = DebugDataset(JSON_PATH, DATA_DIR)
    
    if not os.path.exists("debug_loader_output"):
        os.makedirs("debug_loader_output")

    print(f"Visualizing first 10 training samples...")
    
    for i in range(10):
        img, kpts, name = dataset[i]
        h, w = img.shape[:2]
        
        # Resize to 320x320 (What model sees) without distortion? 
        # No, trainer does direct resize. Let's see direct resize impact.
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        for kp in kpts:
            px = int(kp[0] * IMG_SIZE)
            py = int(kp[1] * IMG_SIZE)
            cv2.circle(img_resized, (px, py), 4, (0, 0, 255), -1)
            
        cv2.imwrite(f"debug_loader_output/debug_{name}", img_resized)
        
    print("Debug images saved to debug_loader_output/")

if __name__ == "__main__":
    main()
