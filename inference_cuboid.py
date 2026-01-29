import os
import torch
import cv2
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

def draw_labeled_bbox(img, points, color):
    """
    Original Cuboid Drawing Logic:
    Draws a 3D wireframe connecting the 9 Objectron keypoints.
    """
    h, w, _ = img.shape
    
    # Draw Center (0)
    cx, cy = int(points[0][0] * w), int(points[0][1] * h)
    cv2.circle(img, (cx, cy), 8, color, -1)
    
    # Vertices 1-8 (Front: 1-4, Back: 5-8)
    front_indices = [1, 2, 3, 4]
    back_indices  = [5, 6, 7, 8]
    
    # Draw all 8 corner points
    for i in range(1, 9):
        px, py = int(points[i][0] * w), int(points[i][1] * h)
        cv2.circle(img, (px, py), 5, color, -1)

    def draw_face(indices):
        """Helper to draw the rectangular faces."""
        for i in range(4):
            idx1, idx2 = indices[i], indices[(i+1)%4]
            p1 = points[idx1]; p2 = points[idx2]
            cv2.line(img, (int(p1[0]*w), int(p1[1]*h)), (int(p2[0]*w), int(p2[1]*h)), color, 3)
            
    # Draw Front and Back Faces
    draw_face(front_indices)
    draw_face(back_indices)

    # Draw Connecting Edges (Front to Back)
    for i in range(4):
        p1 = points[front_indices[i]]
        p2 = points[back_indices[i]]
        cv2.line(img, (int(p1[0]*w), int(p1[1]*h)), (int(p2[0]*w), int(p2[1]*h)), color, 3)

def run_inference():
    print(f"Loading PTH model from {MODEL_PATH}...")
    model = ObjectronMobileNetV3().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    pil_img = Image.open(IMAGE_PATH).convert('RGB')
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        # Get raw prediction and reshape to 9x2
        pth_pred = model(input_tensor).cpu().numpy()[0]
        pth_result = pth_pred.reshape(9, 2)

    # --- VISUALIZATION ---
    orig_img = cv2.imread(IMAGE_PATH)
    if orig_img is None:
        print(f"Error: Could not read image at {IMAGE_PATH}")
        return

    img_viz = orig_img.copy()
    draw_labeled_bbox(img_viz, pth_result, (0, 0, 255)) # Pure Red Wireframe
    
    out_path = os.path.join(OUTPUT_DIR, "result_cuboid.png")
    cv2.imwrite(out_path, img_viz)
    print(f"Cuboid Result saved to {out_path}")

if __name__ == "__main__":
    run_inference()
