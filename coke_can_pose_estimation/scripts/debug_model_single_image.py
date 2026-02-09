import torch
import cv2
import numpy as np
import os
import json
import torch.nn as nn
from torchvision import transforms, models
# import matplotlib.pyplot as plt

# --- CONFIG ---
DEVICE = torch.device("cpu") # CPU is fine for debug
IMG_SIZE = 224
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "../data/raw")

# --- MODEL (Full Definition to Match Checkpoint) ---
class CokeTrackerReduced(nn.Module):
    def __init__(self):
        super(CokeTrackerReduced, self).__init__()
        self.backbone = models.mobilenet_v3_small(weights=None) # No weights needed for loading ckpt
        num_features = self.backbone.classifier[0].in_features
        
        self.regressor_2d = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.Hardswish(),
            nn.Dropout(p=0.2), 
            nn.Linear(512, 18), 
            nn.Sigmoid() 
        )
        self.regressor_3d = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.Hardswish(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 27) 
        )
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
        return kpts_2d

def load_tracker():
    models_root = os.path.join(BASE_DIR, "../models")
    run_folders = [d for d in os.listdir(models_root) if "run_tracker" in d]
    run_folders.sort(reverse=True)
    if not run_folders: return None
    ckpt_path = os.path.join(models_root, run_folders[0], "final", "model_3d.pth")
    print(f"Loading: {ckpt_path}")
    model = CokeTrackerReduced()
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    return model

def main():
    # 1. Load Data
    frame_path = os.path.join(DATA_ROOT, "frames/red/video_01_red/frame_0000.jpg")
    json_path = os.path.join(DATA_ROOT, "annotations/red/video_01_red_annotations.json")
    
    if not os.path.exists(frame_path):
        print(f"Error: {frame_path} not found")
        return
        
    img = cv2.imread(frame_path)
    if img is None: print("Failed to read image"); return
    
    with open(json_path, 'r') as f:
        anns = json.load(f)
    ann = anns[0] # Frame 0
    
    # 2. Extract GT Crop (Replicating train_coke_tracker.py logic exactly)
    raw_kpts_2d = ann['keypoints_2d']
    kpts_2d = []
    for kp in raw_kpts_2d:
        kpts_2d.extend([kp[0], kp[1]]) # x, y
    kpts_2d = np.array(kpts_2d).reshape(9, 2)
    
    # 2a. Rotate 90 CW (Training Logic)
    # Train Script: img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    # And Keypoints: xn = 1.0 - y, yn = x
    img_rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    h_new, w_new = img_rot.shape[:2]
    
    kpts_rot = []
    for kp in kpts_2d:
        x, y = kp
        xn = 1.0 - y
        yn = x
        kpts_rot.append([xn, yn])
    kpts_rot = np.array(kpts_rot)
    
    # 2b. Crop Logic
    xs = kpts_rot[:, 0] * w_new
    ys = kpts_rot[:, 1] * h_new
    x1, y1 = np.min(xs), np.min(ys)
    x2, y2 = np.max(xs), np.max(ys)
    
    cx, cy = (x1+x2)/2, (y1+y2)/2
    w_box, h_box = x2-x1, y2-y1
    crop_size = int(max(w_box, h_box) * 1.5)
    
    bx1 = int(cx - crop_size//2)
    by1 = int(cy - crop_size//2)
    bx2 = bx1 + crop_size
    by2 = by1 + crop_size
    
    # Pad crop
    pad_l = max(0, -bx1); pad_t = max(0, -by1)
    pad_r = max(0, bx2 - w_new); pad_b = max(0, by2 - h_new)
    
    crop = img_rot[max(0, by1):min(h_new, by2), max(0, bx1):min(w_new, bx2)]
    crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0,0,0))
    crop_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    
    # 3. Predict
    model = load_tracker()
    
    # Transform (RGB -> Tensor -> Normalize)
    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    
    # Replicate train_coke_tracker.py: 
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    loader = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    inp = loader(crop_rgb).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        preds = model(inp)[0].numpy().reshape(9, 2)
        
    # 4. Visualize
    vis = crop_resized.copy()
    
    # Draw GT (Blue) - Need to map back to crop space?
    # Actually, verify logic:
    # GT kpts_rot are in Image Space (0-1).
    # Need to map to Crop Space (0-1) then to 224.
    # Map: (img_px - bx1) / crop_size
    
    gt_px = []
    for kp in kpts_rot:
        gx = kp[0] * w_new
        gy = kp[1] * h_new
        lx = (gx - bx1) # local pixel
        ly = (gy - by1)
        nx = lx / crop_size # local normalized
        ny = ly / crop_size
        
        px = int(nx * IMG_SIZE)
        py = int(ny * IMG_SIZE)
        gt_px.append((px, py))
        cv2.circle(vis, (px, py), 3, (255, 0, 0), -1) # Blue = GT
        
    # Draw Pred (Red)
    # Preds are 0-1 relative to 224 crop
    pred_err = 0
    for i, kp in enumerate(preds):
        px = int(kp[0] * IMG_SIZE)
        py = int(kp[1] * IMG_SIZE)
        cv2.circle(vis, (px, py), 3, (0, 0, 255), -1) # Red = Pred
        
        # Calc error
        gt = gt_px[i]
        err = np.sqrt((px-gt[0])**2 + (py-gt[1])**2)
        pred_err += err

    res_path = os.path.join(BASE_DIR, "../inference/results/visualizations/debug_fram0_ovrlay.jpg")
    cv2.imwrite(res_path, vis)
    print(f"Saved: {res_path}")
    print(f"Mean Pixel Error: {pred_err/9.0:.2f}")

    # Check orientation
    # Point 0 is Center. Point 1 match?
    # If Red 1 is far from Blue 1, then orientation is wrong.
    
if __name__ == "__main__":
    main()
