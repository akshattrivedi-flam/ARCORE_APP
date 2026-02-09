import torch
import cv2
import numpy as np
import torch.nn as nn
from torchvision import transforms, models
# from train_coke_tracker import CokeTrackerReduced, IMG_SIZE, DEVICE # Redefine locally to ensure sync
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL DEFINITION (Must Match Training) ---
class CokeTrackerReduced(nn.Module):
    def __init__(self):
        super(CokeTrackerReduced, self).__init__()
        # MobileNetV3 Small Backbone
        self.backbone = models.mobilenet_v3_small(pretrained=False) # Weights loaded manually later
        
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

# --- CONFIG ---
VIDEO_PATH = "/home/user/Desktop/ARCORE_APP/DATASET_TRAINING/red/video_10_red/video_raw.mp4"
MODEL_PATH = "/home/user/Desktop/ARCORE_APP/DATASET_TRAINING/coke_tracker_checkpoint_60.pth"
OUTPUT_PATH = "/home/user/Desktop/ARCORE_APP/DATASET_TRAINING/red/video_10_red/inference_check_60.mp4"

# --- 3D GEOMETRY CONSTANTS ---
# Estimated Coke Can Box (in meters) derived from annotations.json
# Analysis of Model Matrix and Keypoints indicates:
# Y-axis is Height (Long axis ~15.5cm)
# X and Z are Width/Depth (~6.5cm)
# Dimensions: W=0.065, H=0.155, D=0.065
half_w = 0.065 / 2.0  # X
half_h = 0.155 / 2.0  # Y (Height)
half_d = 0.065 / 2.0  # Z

# Vertices 1-8
# Analysis of annotations.json (and implicit Objectron convention):
# 1-4: Front Face. 5-8: Back Face.
# 1: Top-Right, 2: Bottom-Right, 3: Bottom-Left, 4: Top-Left
# Y is Height (Up/Down). X is Width (Right/Left). Z is Depth (Front/Back).
CANONICAL_BOX_3D = np.array([
    [0, 0, 0], # Centroid
    [half_w, half_h, -half_d],   # 1: Top-Right-Front
    [half_w, -half_h, -half_d],  # 2: Bottom-Right-Front
    [-half_w, -half_h, -half_d], # 3: Bottom-Left-Front
    [-half_w, half_h, -half_d],  # 4: Top-Left-Front
    [half_w, half_h, half_d],    # 5: Top-Right-Back
    [half_w, -half_h, half_d],   # 6: Bottom-Right-Back
    [-half_w, -half_h, half_d],  # 7: Bottom-Left-Back
    [-half_w, half_h, half_d]    # 8: Top-Left-Back
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
    
    # We will output in ORIGINAL Resolution (Landscape)
    # If input is 640x480, output is 640x480
    out_width, out_height = width, height
    
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
            kpts_raw, conf_raw = model(input_tensor)
            
            # 1. Check Confidence
            conf_score = torch.sigmoid(conf_raw).item()
            
            if conf_score < 0.5:
                # NEGATIVE DETECTED (Background, Blue, Silver)
                # Skip Drawing, just write ORIGINAL frame
                out.write(frame)
                
                # Reset Tracking State if confidence drops
                prev_gray = None
                prev_box_pts = None
                
                if frame_idx % 30 == 0:
                   print(f"Frame {frame_idx}: No Object (Conf: {conf_score:.4f})")
                frame_idx += 1
                continue

            # OBJECT FOUND
            raw_flat = kpts_raw.cpu().numpy()[0] # [18,]
        
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
        
        # --- SOLVE PNP RANSAC (ROBUST GEOMETRY) ---
        # Scale Intrinsics
        base_w, base_h = 480, 640
        h, w = frame_rot.shape[:2]
        scale_x = w / base_w
        scale_y = h / base_h
        
        scaled_matrix = CAMERA_MATRIX.copy()
        scaled_matrix[0, 0] *= scale_x # fx
        scaled_matrix[1, 1] *= scale_y # fy
        scaled_matrix[0, 2] *= scale_x # cx
        scaled_matrix[1, 2] *= scale_y # cy

        image_points = final_pts.astype(np.float32)
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            CANONICAL_BOX_3D, 
            image_points, 
            scaled_matrix, 
            DIST_COEFFS, 
            iterationsCount=100,
            reprojectionError=10.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        # --- KALMAN UPDATE STEP ---
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
        projected_points, _ = cv2.projectPoints(CANONICAL_BOX_3D, smooth_rvec, smooth_tvec, scaled_matrix, DIST_COEFFS)
        pts = [tuple(map(int, p.ravel())) for p in projected_points]

        # Draw points
        for i, p in enumerate(pts):
            color = (0, 0, 255) if i == 0 else (255, 0, 0)
            cv2.circle(frame_rot, p, 4, color, -1)
                
        # Draw Connections (Green)
        conns = get_connections()
        for start, end in conns:
            if start < len(pts) and end < len(pts):
                cv2.line(frame_rot, pts[start], pts[end], (0, 255, 0), 2)

        # Rotate BACK to Original Orientation (Landscape / 90 CCW)
        frame_final = cv2.rotate(frame_rot, cv2.ROTATE_90_COUNTERCLOCKWISE)
        out.write(frame_final)
        
        if frame_idx % 30 == 0:
            print(f"Processed frame {frame_idx}")
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Inference video saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
