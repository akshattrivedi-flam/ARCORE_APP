import torch
import cv2
import numpy as np
import os
import torch.nn as nn
from torchvision import transforms, models

# --- CONFIG ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Default to video 10 as in inference script
VIDEO_PATH = os.path.join(CURRENT_DIR, "../data/raw/videos/red/video_10_red.mp4")
MODEL_PATH = os.path.join(CURRENT_DIR, "../models/checkpoints/coke_tracker_checkpoint_60.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 320

# --- MODEL DEFINITION (Must Match Training) ---
class CokeTrackerReduced(nn.Module):
    def __init__(self):
        super(CokeTrackerReduced, self).__init__()
        # MobileNetV3 Small Backbone
        self.backbone = models.mobilenet_v3_small(weights=None) 
        
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
        )
        
        self.backbone.classifier = nn.Identity()

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        kpts = self.regressor(x)
        conf = self.classifier(x)
        return kpts, conf

def debug():
    print(f"Debug: Loading model from {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("Error: Model file not found.")
        return

    model = CokeTrackerReduced().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading state dict: {e}")
        return
        
    model.eval()
    print("Model loaded.")

    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video not found at {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    frames_to_check = 20
    
    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("\n--- Processing First 20 Frames ---")
    for i in range(frames_to_check):
        ret, frame = cap.read()
        if not ret: break

        # Rotate to Portrait (matching inference logic)
        frame_rot = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        img_rgb = cv2.cvtColor(frame_rot, cv2.COLOR_BGR2RGB)
        pil_img = transforms.ToPILImage()(img_rgb)
        
        input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            kpts, conf_raw = model(input_tensor)
            conf_sig = torch.sigmoid(conf_raw).item()
            
        print(f"Frame {i}: Conf Raw={conf_raw.item():.4f} | Sigmoid={conf_sig:.4f} | {'DETECTED' if conf_sig > 0.5 else 'NO COKE'}")

    cap.release()
    print("--- End Debug ---")

if __name__ == "__main__":
    debug()
