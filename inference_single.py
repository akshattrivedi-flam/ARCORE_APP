import os
import torch
import cv2
import json
import numpy as np
from torchvision import transforms
from PIL import Image
from train_objectron import ObjectronMobileNetV3

# --- CONFIGURATION ---
MODEL_PATH = "/home/user/Desktop/ARCORE_APP/bottle_objectron_overfit.pth"
IMAGE_PATH = "/home/user/Desktop/ARCORE_APP/IMG_20260128_164324.jpg"
OUTPUT_DIR = "/home/user/Desktop/ARCORE_APP/test_botte_off_ground"
IMAGE_SIZE = 224
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def draw_pinpoints_with_coords(img, points, color):
    """
    Draws 9 pinpoints on the image and labels each with its (x, y) coordinates.
    """
    h, w, _ = img.shape
    
    # Labels for the 9 Objectron keypoints
    labels = ["Center", "FBL", "FBR", "FTR", "FTL", "BackBL", "BackBR", "BackTR", "BackTL"]
    
    for i in range(9):
        # Calculate pixel coordinates
        px_float, py_float = points[i][0] * w, points[i][1] * h
        px, py = int(px_float), int(py_float)
        
        # 1. Draw the pinpoint (solid circle)
        cv2.circle(img, (px, py), 5, color, -1)
        
        # 2. Draw the coordinate text (X, Y) next to the point
        # We show the pixel coordinates for accuracy, or normalized if preferred.
        # Format: "P0: (123, 456)"
        coord_text = f"P{i}: ({px}, {py})"
        
        # Draw background rectangle for text readability
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(coord_text, font, font_scale, thickness)
        cv2.rectangle(img, (px + 8, py - text_h - 5), (px + 8 + text_w, py + 5), (0, 0, 0), -1)
        
        # Draw the text
        cv2.putText(img, coord_text, (px + 8, py), font, font_scale, (255, 255, 255), thickness)

def draw_labeled_bbox(img, points, color):
    """
    Keeps the cuboid wireframe visualization as well.
    """
    h, w, _ = img.shape
    front_indices = [1, 2, 3, 4]
    back_indices  = [5, 6, 7, 8]
    
    def draw_face(indices):
        for i in range(4):
            idx1, idx2 = indices[i], indices[(i+1)%4]
            p1 = points[idx1]; p2 = points[idx2]
            cv2.line(img, (int(p1[0]*w), int(p1[1]*h)), (int(p2[0]*w), int(p2[1]*h)), color, 2)
            
    draw_face(front_indices)
    draw_face(back_indices)
    for i in range(4):
        p1 = points[front_indices[i]]; p2 = points[back_indices[i]]
        cv2.line(img, (int(p1[0]*w), int(p1[1]*h)), (int(p2[0]*w), int(p2[1]*h)), color, 2)

def run_inference():
    # --- 1. PYTORCH INFERENCE ---
    print(f"Loading PTH model...")
    model = ObjectronMobileNetV3().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        pth_result = None
    else:
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        pil_img = Image.open(IMAGE_PATH).convert('RGB')
        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pth_result = model(input_tensor).cpu().numpy()[0].reshape(9, 2)

    # --- 2. DATA EXPORT (JSON) ---
    if pth_result is not None:
        json_data = {
            "image_path": IMAGE_PATH,
            "keypoints_normalized": pth_result.tolist(),
            "units": "normalized (0-1)"
        }
        json_path = os.path.join(OUTPUT_DIR, "results.json")
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        print(f"Results saved to {json_path}")

    # --- 3. VISUALIZATION ---
    orig_img = cv2.imread(IMAGE_PATH)
    if orig_img is None: return

    if pth_result is not None:
        img_viz = orig_img.copy()
        # Draw the wireframe for context (subtle)
        draw_labeled_bbox(img_viz, pth_result, (100, 100, 100)) 
        # Draw the pinpoints and (x, y) coordinates
        draw_pinpoints_with_coords(img_viz, pth_result, (0, 0, 255))
        
        out_path = os.path.join(OUTPUT_DIR, "result_pinpoints.png")
        cv2.imwrite(out_path, img_viz)
        print(f"Visualization saved to {out_path}")
        
        # Optional: Display (if running in a GUI environment)
        # cv2.imshow("Pinpoint Detection", img_viz)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference()


