import os
import json
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from train_objectron import ObjectronMobileNetV3, align_keypoints

# --- CONFIGURATION ---
DATA_DIR = "/home/user/Desktop/ARCORE_APP/seq_120737245"
JSON_FILE = os.path.join(DATA_DIR, "annotations.json")
MODEL_PATH = "/home/user/Desktop/ARCORE_APP/bottle_objectron_overfit.pth"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sanity_check():
    # 1. Load Model
    model = ObjectronMobileNetV3().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. Load Annotations
    with open(JSON_FILE, 'r') as f:
        annotations = json.load(f)
    
    # 3. Process first frame
    entry = annotations[0]
    img_path = os.path.join(DATA_DIR, entry['image'])
    print(f"Testing on Training Frame: {img_path}")
    
    with Image.open(img_path) as img:
        real_w, real_h = img.size
        # Resize for model
        input_img = img.resize((224, 224))
        
    # Preprocess
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(input_img).unsqueeze(0).to(DEVICE)

    # 4. Predict
    with torch.no_grad():
        pred = model(input_tensor).cpu().numpy()[0].reshape(9, 2)
    
    # 5. Get Ground Truth (Aligned)
    kpts_raw = np.array(entry['keypoints_2d'])[:, :2]
    gt_aligned = align_keypoints(kpts_raw, real_w, real_h, 
                                 entry['camera_intrinsics']['image_width'], 
                                 entry['camera_intrinsics']['image_height'])
    
    # 6. Compare
    print("\nComparison (Normalized):")
    print(f"GT Center:   {gt_aligned[0]}")
    print(f"PRED Center: {pred[0]}")
    
    dist = np.linalg.norm(gt_aligned[0] - pred[0])
    print(f"Center Distance Error: {dist:.4f}")
    
    # Visualization
    viz_img = cv2.imread(img_path)
    # Draw GT (Green)
    for p in gt_aligned:
        cv2.circle(viz_img, (int(p[0]*real_w), int(p[1]*real_h)), 8, (0, 255, 0), -1)
    # Draw Pred (Red)
    for p in pred:
        cv2.circle(viz_img, (int(p[0]*real_w), int(p[1]*real_h)), 5, (0, 0, 255), -1)
        
    out_path = "/home/user/Desktop/ARCORE_APP/test_botte_off_ground/sanity_check.png"
    cv2.imwrite(out_path, viz_img)
    print(f"\nSanity check image saved to {out_path}")

if __name__ == "__main__":
    sanity_check()
