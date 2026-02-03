import torch
import cv2
import numpy as np
import os
import glob
from torchvision import transforms
from train_coke_tracker import CokeTrackerReduced, IMG_SIZE, DEVICE

# --- CONFIG ---
COMPOSITED_DIR = "/home/user/Desktop/ARCORE_APP/video_01_red/composited"
OUTPUT_DIR = "/home/user/Desktop/ARCORE_APP/video_01_red/output_inference_composited"
MODEL_PATH = "/home/user/Desktop/ARCORE_APP/video_01_red/coke_tracker_mobilenetv3.pth"

# Exact Can Dimensions (from calculate_dims.py)
half_w, half_h, half_d = 0.1390/2, 0.0690/2, 0.0630/2
CANONICAL_BOX_3D = np.array([
    [0, 0, 0], 
    [half_w, -half_h, -half_d],   [half_w, -half_h, half_d],    [-half_w, -half_h, half_d],   [-half_w, -half_h, -half_d],
    [half_w, half_h, -half_d],    [half_w, half_h, half_d],     [-half_w, half_h, half_d],    [-half_w, half_h, -half_d]
], dtype=np.float32)

# Camera Matrix (Same as inference_coke.py)
CAMERA_MATRIX = np.array([
    [428.8, 0, 241.7],
    [0, 429.3, 314.4],
    [0, 0, 1]
], dtype=np.float32)
DIST_COEFFS = np.zeros(5)

def get_connections():
    return [
        (1, 2), (2, 3), (3, 4), (4, 1),
        (5, 6), (6, 7), (7, 8), (8, 5),
        (1, 5), (2, 6), (3, 7), (4, 8)
    ]

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load Model
    model = CokeTrackerReduced().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded.")

    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get sorted list of images
    image_files = sorted(glob.glob(os.path.join(COMPOSITED_DIR, "*.jpg")))
    print(f"Found {len(image_files)} images.")

    # State for Tracking (if treating as sequence)
    GRID_ALPHA = 0.3
    prev_gray = None
    prev_box_pts = None
    
    for i, img_path in enumerate(image_files):
        frame = cv2.imread(img_path)
        if frame is None: continue
        
        # ROTATION LOGIC:
        # Training data was rotated 90 CW (Landscape -> Portrait).
        # Composited images are likely 640x480 (Landscape).
        # We MUST rotate them to match what the model expects.
        h, w = frame.shape[:2]
        if w > h: # Landscape
            frame_rot = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        else:
            frame_rot = frame # Already portrait?
            
        # Prepare Input
        img_rgb = cv2.cvtColor(frame_rot, cv2.COLOR_BGR2RGB)
        curr_gray = cv2.cvtColor(frame_rot, cv2.COLOR_BGR2GRAY)
        
        pil_img = transforms.ToPILImage()(img_rgb)
        input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)
        
        # Inference (CNN)
        with torch.no_grad():
            raw_flat = model(input_tensor).cpu().numpy()[0]
            
        cnn_pts = []
        h_rot, w_rot = frame_rot.shape[:2]
        for k in range(0, 18, 2):
            cnn_pts.append([raw_flat[k]*w_rot, raw_flat[k+1]*h_rot])
        cnn_pts = np.array(cnn_pts, dtype=np.float32)

        # Hybrid Tracking Logic
        final_pts = cnn_pts
        if prev_gray is not None and prev_box_pts is not None:
             flow_pts, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_box_pts, None, 
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )
             valid_idx = status.flatten() == 1
             if np.mean(valid_idx) > 0.5:
                 final_pts = (1.0 - GRID_ALPHA) * flow_pts + (GRID_ALPHA) * cnn_pts
        
        # Update State
        prev_gray = curr_gray.copy()
        prev_box_pts = final_pts
        
        # PnP solve
        image_points = final_pts.astype(np.float32)
        success, rvec, tvec = cv2.solvePnP(CANONICAL_BOX_3D, image_points, CAMERA_MATRIX, DIST_COEFFS, flags=cv2.SOLVEPNP_ITERATIVE)
        
        if success:
            projected_points, _ = cv2.projectPoints(CANONICAL_BOX_3D, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS)
            pts = [tuple(map(int, p.ravel())) for p in projected_points]
        else:
            pts = [tuple(map(int, p)) for p in final_pts]

        # Draw
        for idx, p in enumerate(pts):
            color = (0, 0, 255) if idx == 0 else (255, 0, 0)
            cv2.circle(frame_rot, p, 4, color, -1)
            
        for start, end in get_connections():
             if start < len(pts) and end < len(pts):
                cv2.line(frame_rot, pts[start], pts[end], (0, 255, 0), 2)

        # Save
        filename = os.path.basename(img_path)
        save_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(save_path, frame_rot)
        
        if i % 50 == 0:
            print(f"Processed {i}/{len(image_files)}")

    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
