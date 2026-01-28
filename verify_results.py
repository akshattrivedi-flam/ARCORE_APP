import os
import json
import torch
import cv2
import numpy as np
from torchvision import transforms, models
from PIL import Image
from train_objectron import ObjectronMobileNetV3

# --- CONFIGURATION ---
DATA_DIR = "/home/user/Desktop/ARCORE_APP/seq_120737245"
JSON_FILE = os.path.join(DATA_DIR, "annotations.json")
MODEL_PATH = "bottle_objectron_overfit.pth"
IMAGE_SIZE = 224
OUTPUT_DIR = "results_viz_final"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def draw_labeled_bbox(img, points, color, label_prefix=""):
    h, w, _ = img.shape
    
    # Draw Center (0)
    cx, cy = int(points[0][0] * w), int(points[0][1] * h)
    cv2.circle(img, (cx, cy), 5, color, -1)
    
    # Vertices 1-8 (Front: 1-4, Back: 5-8)
    # 1: FBL, 2: FBR, 3: FTR, 4: FTL
    # 5: BBL, 6: BBR, 7: BTR, 8: BTL
    front_indices = [1, 2, 3, 4]
    back_indices  = [5, 6, 7, 8]
    
    # Draw points
    for i in range(1, 9):
        px, py = int(points[i][0] * w), int(points[i][1] * h)
        cv2.circle(img, (px, py), 3, color, -1)

    # Draw Faces (Lines)
    def draw_face(indices):
        for i in range(4):
            idx1, idx2 = indices[i], indices[(i+1)%4]
            p1 = points[idx1]; p2 = points[idx2]
            cv2.line(img, (int(p1[0]*w), int(p1[1]*h)), (int(p2[0]*w), int(p2[1]*h)), color, 2)
            
    draw_face(front_indices)
    draw_face(back_indices)
    
    # Connecting Edges (1-5, etc)
    for i in range(4):
        p1 = points[front_indices[i]]
        p2 = points[back_indices[i]]
        cv2.line(img, (int(p1[0]*w), int(p1[1]*h)), (int(p2[0]*w), int(p2[1]*h)), color, 2)

def align_keypoints(kpts, img_w, img_h, json_w=640, json_h=480):
    # Same logic as training to visualize Ground Truth correctly
    # 1. Rotation CW
    kpts_rot = np.zeros_like(kpts)
    kpts_rot[:, 0] = 1.0 - kpts[:, 1]
    kpts_rot[:, 1] = kpts[:, 0]
    
    sens_w_rot, sens_h_rot = json_h, json_w
    
    scale = max(img_w / sens_w_rot, img_h / sens_h_rot)
    proj_w = sens_w_rot * scale
    proj_h = sens_h_rot * scale
    
    off_x = (proj_w - img_w) / 2.0
    off_y = (proj_h - img_h) / 2.0
    
    kpts_screen = np.zeros_like(kpts_rot)
    kpts_screen[:, 0] = ((kpts_rot[:, 0] * proj_w) - off_x) / img_w
    kpts_screen[:, 1] = ((kpts_rot[:, 1] * proj_h) - off_y) / img_h
    
    return kpts_screen

def visualize():
    print(f"Loading model from {MODEL_PATH}...")
    model = ObjectronMobileNetV3().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with open(JSON_FILE, 'r') as f:
        annotations = json.load(f)

    # Get dims from first image
    first_path = os.path.join(DATA_DIR, annotations[0]['image'])
    with Image.open(first_path) as tmp:
        real_w, real_h = tmp.size

    print(f"Generating aligned visualizations for 20 frames...")

    for i, entry in enumerate(annotations[:20]):
        img_name = entry['image']
        img_path = os.path.join(DATA_DIR, img_name)
        orig_img = cv2.imread(img_path)
        
        # 1. Ground Truth (Blue) - Must apply same alignment logic to be valid comparison!
        kpts_raw = np.array(entry['keypoints_2d'])[:, :2]
        kpts_gt = align_keypoints(kpts_raw, real_w, real_h, 
                                  entry['camera_intrinsics']['image_width'], 
                                  entry['camera_intrinsics']['image_height'])
        draw_labeled_bbox(orig_img, kpts_gt, (255, 0, 0), "GT")

        # 2. Prediction (Red) - Model predicts in Aligned Space directly
        pil_img = Image.open(img_path).convert('RGB')
        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred = model(input_tensor).cpu().numpy()[0]
        
        pred_kpts = pred.reshape(9, 2)
        draw_labeled_bbox(orig_img, pred_kpts, (0, 0, 255), "P")

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"final_{img_name}"), orig_img)

    print(f"Final Visualization done in {OUTPUT_DIR}. Check final_*.jpg")

if __name__ == "__main__":
    visualize()
