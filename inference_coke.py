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

    # 4. Inference & Tracking Loop
    # Hybrid Strategy: CNN + Optical Flow + RANSAC PnP + Kalman Filter
    
    GRID_ALPHA = 0.3 
    
    prev_gray = None
    prev_box_pts = None 
    
    # --- KALMAN FILTER SETUP ---
    # State: [tx, ty, tz, rx, ry, rz, v_tx, v_ty, v_tz, v_rx, v_ry, v_rz] (12 vars)
    # Measurement: [tx, ty, tz, rx, ry, rz] (6 vars)
    kalman = cv2.KalmanFilter(12, 6)
    kalman.transitionMatrix = np.eye(12, dtype=np.float32)
    # Velocity transfer
    kalman.transitionMatrix[0, 6] = 1.0 # tx += v_tx
    kalman.transitionMatrix[1, 7] = 1.0 # ty += v_ty
    kalman.transitionMatrix[2, 8] = 1.0 # tz += v_tz
    kalman.transitionMatrix[3, 9] = 1.0 # rx += v_rx
    kalman.transitionMatrix[4, 10] = 1.0 # ry += v_ry
    kalman.transitionMatrix[5, 11] = 1.0 # rz += v_rz
    
    kalman.measurementMatrix = np.eye(6, 12, dtype=np.float32)
    kalman.processNoiseCov = np.eye(12, dtype=np.float32) * 1e-4
    kalman.measurementNoiseCov = np.eye(6, dtype=np.float32) * 1e-2
    kalman.errorCovPost = np.eye(12, dtype=np.float32)
    
    # Initialize with default pose
    kalman.statePost = np.zeros((12, 1), dtype=np.float32)
    kalman.statePost[2] = 0.5 # Default Z depth 0.5m

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
        
        # Prepare for Optical Flow (Gray)
        curr_gray = cv2.cvtColor(frame_rot, cv2.COLOR_BGR2GRAY)
        
        # Inference (CNN)
        with torch.no_grad():
            raw_flat = model(input_tensor).cpu().numpy()[0] # [18,]
        
        # Reshape CNN result to (9, 2)
        cnn_pts = []
        h, w = frame_rot.shape[:2]
        for i in range(0, 18, 2):
            cnn_pts.append([raw_flat[i]*w, raw_flat[i+1]*h])
        cnn_pts = np.array(cnn_pts, dtype=np.float32)

        # --- HYBRID TRACKING ---
        final_pts = cnn_pts # Default to CNN
        
        if prev_gray is not None and prev_box_pts is not None:
            # 1. OPTICAL FLOW: Track points from last frame
            flow_pts, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_box_pts, None, 
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )
            
            # 2. VALIDATE FLOW directly
            # Check if flow diverged too far from CNN (Drift check)
            # If flow is good, we fuse.
            
            # Simple fuse: Avg(Flow, CNN)
            # Actually, to get "Perfect Fit", we trust Flow almost 100% for short term
            # But we drag it back to CNN to prevent long term drift.
            
            valid_idx = status.flatten() == 1
            if np.mean(valid_idx) > 0.5: # If tracking succeeded for most points
                # Weighted Fusion
                fused = (1.0 - GRID_ALPHA) * flow_pts + (GRID_ALPHA) * cnn_pts
                final_pts = fused
            else:
                final_pts = cnn_pts # Lost tracking, reset to CNN

        # Update State
        prev_gray = curr_gray.copy()
        prev_box_pts = final_pts
        
        # Flatten for drawing & PnP logic
        preds = final_pts.flatten()
        
        # Normalize back to 0-1 for PnP logic (which expects denormalized, wait)
        # The code below expects `preds` to be 0-1 normalized flat array?
        # Let's check: "px = preds[i] * w". Yes.
        # But `final_pts` is already in pixels.
        # So we must avoid re-multiplying by W/H below.
        
        # --- SOLVE PNP RANSAC (ROBUST GEOMETRY) ---
        # RANSAC will ignore outlier points that don't fit the rigid 3D model
        image_points = final_pts.astype(np.float32)
        
        # We need at least 4 points for PnP
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            CANONICAL_BOX_3D, 
            image_points, 
            CAMERA_MATRIX, 
            DIST_COEFFS, 
            iterationsCount=100,
            reprojectionError=8.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        # --- KALMAN UPDATE STEP ---
        # Predict next state
        prediction = kalman.predict()
        
        if success:
            # Update Kalman with new measurement
            measurement = np.array([
                [tvec[0][0]], [tvec[1][0]], [tvec[2][0]],
                [rvec[0][0]], [rvec[1][0]], [rvec[2][0]]
            ], dtype=np.float32)
            kalman.correct(measurement)
            
            # Use Corrected State
            final_state = kalman.statePost
        else:
            # Use Predicted State (Coast through occlusion)
            final_state = prediction

        # Extract Smooth Pose
        smooth_tvec = final_state[0:3]
        smooth_rvec = final_state[3:6]

        # Project Rigid Points back using SMOOTHED pose
        projected_points, _ = cv2.projectPoints(CANONICAL_BOX_3D, smooth_rvec, smooth_tvec, CAMERA_MATRIX, DIST_COEFFS)
        pts = [tuple(map(int, p.ravel())) for p in projected_points]

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
