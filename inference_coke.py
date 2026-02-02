import torch
import cv2
import numpy as np
from torchvision import transforms
from train_coke_tracker import CokeTrackerReduced, IMG_SIZE, DEVICE

# --- CONFIG ---
VIDEO_PATH = "/home/user/Desktop/ARCORE_APP/video_01_red/video_raw.mp4"
MODEL_PATH = "/home/user/Desktop/ARCORE_APP/video_01_red/coke_tracker_mobilenetv3.pth"
OUTPUT_PATH = "/home/user/Desktop/ARCORE_APP/video_01_red/inference_result.mp4"

# --- 3D GEOMETRY CONSTANTS ---
# Estimated Coke Can Box (in meters) roughly derived from keypoints
# Centroid is 0,0,0
# Order: Centroid, Front-TL, Front-TR, Front-BR, Front-BL, Back-TL, Back-TR, Back-BR, Back-BL
# Calculated Averages: W=0.1390, H=0.0690, D=0.0630
# NOTE: The axis mapping might be different.
# Based on the aspect ratio (H/W = 0.5), it seems X is longest (Height of can?).
# Let's assign:
# Axis 0 (X): 0.1390 -> Can Height
# Axis 1 (Y): 0.0690 -> Can Width
# Axis 2 (Z): 0.0630 -> Can Depth
half_w, half_h, half_d = 0.1390/2, 0.0690/2, 0.0630/2
CANONICAL_BOX_3D = np.array([
    [0, 0, 0], # Centroid
    [half_w, -half_h, -half_d],   # 1
    [half_w, -half_h, half_d],    # 2
    [-half_w, -half_h, half_d],   # 3
    [-half_w, -half_h, -half_d],  # 4
    [half_w, half_h, -half_d],    # 5
    [half_w, half_h, half_d],     # 6
    [-half_w, half_h, half_d],    # 7
    [-half_w, half_h, -half_d]    # 8
], dtype=np.float32)

# Camera Matrix (Intrinsics from json)
# cx: 325.6, cy: 241.6, fx: 429.3, fy: 428.8
# NOTE: The inference image is ROTATED 90 deg. 
# So fx/fy swap, and cx/cy swap AND flip.
# Original: 640x480. Rotated: 480x640.
# cx_new = cy_old = 241.6
# cy_new = width - cx_old = 640 - 325.6 = 314.4
CAMERA_MATRIX = np.array([
    [428.8, 0, 241.7],
    [0, 429.3, 314.4],
    [0, 0, 1]
], dtype=np.float32)
DIST_COEFFS = np.zeros(5) # Assume zero distortion for simplicity

def get_connections():
    # Same connection scheme as visualization
    return [
        (1, 2), (2, 3), (3, 4), (4, 1), # Front
        (5, 6), (6, 7), (7, 8), (8, 5), # Back
        (1, 5), (2, 6), (3, 7), (4, 8)  # Connectors
    ]

def main():
    # 1. Load Model
    model = CokeTrackerReduced().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print("Model file not found! Training might verify be running.")
        return

    model.eval()
    print("Model loaded successfully.")

    # 2. Open Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # We will output a ROTATED video (Portrait)
    # If input is 640x480, output is 480x640
    out_width, out_height = height, width
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (out_width, out_height))

    # 3. Preprocessing Transform (Same as training, minus augmentation)
    IMG_SIZE = 320 # MATCH TRAINING CONFIG
    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Inference Loop with Smoothing
    # Alpha: 0.0 -> Infinite smoothing (no update), 1.0 -> No smoothing (raw raw)
    # 0.2 is a good balance for handheld objects
    GRID_ALPHA = 0.2 
    smoothed_preds = None

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Rotate to Portrait (90 deg CW) - Matching Training Data
        frame_rot = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        # Prepare for Model
        # Input: RGB
        img_rgb = cv2.cvtColor(frame_rot, cv2.COLOR_BGR2RGB)
        pil_img = transforms.ToPILImage()(img_rgb)
        input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            raw_preds = model(input_tensor).cpu().numpy()[0] # [18,]
            
        # --- TEMPORAL SMOOTHING ---
        if smoothed_preds is None:
            smoothed_preds = raw_preds
        else:
            smoothed_preds = (GRID_ALPHA * raw_preds) + ((1 - GRID_ALPHA) * smoothed_preds)
            
        preds = smoothed_preds
        
        # --- SOLVE PNP (GEOMETRIC CORRECTION) ---
        # 1. Reshape preds to (9, 2)
        h, w = frame_rot.shape[:2]
        image_points = []
        for i in range(0, 18, 2):
            px = preds[i] * w
            py = preds[i+1] * h
            image_points.append([px, py])
        image_points = np.array(image_points, dtype=np.float32)
        
        # 2. Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            CANONICAL_BOX_3D, 
            image_points, 
            CAMERA_MATRIX, 
            DIST_COEFFS,
             
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            # 3. Project Rigid Points back
            projected_points, _ = cv2.projectPoints(CANONICAL_BOX_3D, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS)
            pts = [tuple(map(int, p.ravel())) for p in projected_points]
        else:
            # Fallback to raw predictions if PnP fails
            pts = [tuple(map(int, p)) for p in image_points]

        # Draw points
        for i, p in enumerate(pts):
            color = (0, 0, 255) if i == 0 else (255, 0, 0)
            cv2.circle(frame_rot, p, 4, color, -1)
            
        # Draw Connections (Green)
        conns = get_connections()
        for start, end in conns:
            if start < len(pts) and end < len(pts):
                # Skip centroid connections for cleaner look if desired
                # But connection list handles indices 1-8 primarily
                cv2.line(frame_rot, pts[start], pts[end], (0, 255, 0), 2)
                
        out.write(frame_rot)
        
        if frame_idx % 30 == 0:
            print(f"Processed frame {frame_idx}")
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Inference video saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
