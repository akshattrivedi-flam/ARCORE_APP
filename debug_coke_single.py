import torch
import cv2
import numpy as np
import os
from torchvision import transforms
from inference_coke import CokeTrackerReduced, CANONICAL_BOX_3D, CAMERA_MATRIX, DIST_COEFFS

# Config
IMG_PATH = "/home/user/Desktop/ARCORE_APP/DATASET_TRAINING/red/video_10_red/frame_0000.jpg"
MODEL_PATH = "/home/user/Desktop/ARCORE_APP/DATASET_TRAINING/coke_tracker_checkpoint_60.pth"
OUTPUT_PATH = "debug_coke_static.jpg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"Loading model from {MODEL_PATH}")
    model = CokeTrackerReduced().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Load Image
    img_bgr = cv2.imread(IMG_PATH)
    if img_bgr is None:
        print(f"Error reading {IMG_PATH}")
        return

    # Rotate 90 CW (Landscape -> Portrait)
    img_rot = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    h, w = img_rot.shape[:2]

    # Preprocess
    preprocess = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_rgb = cv2.cvtColor(img_rot, cv2.COLOR_BGR2RGB)
    pil_img = transforms.ToPILImage()(img_rgb)
    input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        kpts_raw, conf = model(input_tensor)
    
    # Process Points
    raw_flat = kpts_raw.cpu().numpy()[0]
    cnn_pts = []
    print("\n--- PREDICTED KEYPOINTS ---")
    img_h, img_w = img_rot.shape[:2]
    
    for i in range(0, 18, 2):
        x = raw_flat[i] * w
        y = raw_flat[i+1] * h
        cnn_pts.append([x, y])
        print(f"Pt {i//2}: ({x:.2f}, {y:.2f}) [Norm: {raw_flat[i]:.2f}, {raw_flat[i+1]:.2f}]")
        
        if x < 0 or x > img_w or y < 0 or y > img_h:
            print(f"  WARNING: Point {i//2} out of bounds!")

    cnn_pts = np.array(cnn_pts, dtype=np.float32)

    # Draw CNN Points (BLUE)
    viz_img = img_rot.copy()
    for i, p in enumerate(cnn_pts):
        cv2.circle(viz_img, (int(p[0]), int(p[1])), 6, (255, 0, 0), -1) # Blue (BGR)
        cv2.putText(viz_img, str(i), (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Scale Intrinsics to match Image Resolution
    # Base Intrinsics are for 480x640 (Portrait)
    # Current Image is h x w
    base_w, base_h = 480, 640
    scale_x = w / base_w
    scale_y = h / base_h
    
    print(f"Image Res: {w}x{h}. Base Res: {base_w}x{base_h}. Scale: {scale_x:.2f}, {scale_y:.2f}")

    scaled_matrix = CAMERA_MATRIX.copy()
    scaled_matrix[0, 0] *= scale_x # fx
    scaled_matrix[1, 1] *= scale_y # fy
    scaled_matrix[0, 2] *= scale_x # cx
    scaled_matrix[1, 2] *= scale_y # cy

    # PnP
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        CANONICAL_BOX_3D, cnn_pts, scaled_matrix, DIST_COEFFS,
        iterationsCount=100, reprojectionError=10.0, confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE
    )

    if success:
        print(f"PnP Success! Translation: {tvec.flatten()}")
        # Project Box
        projected_points, _ = cv2.projectPoints(CANONICAL_BOX_3D, rvec, tvec, scaled_matrix, DIST_COEFFS)
        pts = [tuple(map(int, p.ravel())) for p in projected_points]

        # Draw Box (Green lines, Red corners)
        connections = [
            (1, 2), (2, 3), (3, 4), (4, 1), # Front
            (5, 6), (6, 7), (7, 8), (8, 5), # Back
            (1, 5), (2, 6), (3, 7), (4, 8)  # Ribs
        ]
        
        for p in pts:
            cv2.circle(viz_img, p, 3, (0, 0, 255), -1)
            
        for s, e in connections:
            if s < len(pts) and e < len(pts):
                cv2.line(viz_img, pts[s], pts[e], (0, 255, 0), 2)
    else:
        print("PnP Failed: Could not solve pose.")

    # Rotate Back for view
    final_img = cv2.rotate(viz_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(OUTPUT_PATH, final_img)
    print(f"Saved debug image to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
