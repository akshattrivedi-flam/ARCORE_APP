import torch
import cv2
import numpy as np
from torchvision import transforms
from train_coke_tracker import CokeTrackerReduced, IMG_SIZE, DEVICE

# --- CONFIG ---
VIDEO_PATH = "/home/user/Desktop/ARCORE_APP/video_01_red/video_raw.mp4"
MODEL_PATH = "/home/user/Desktop/ARCORE_APP/video_01_red/coke_tracker_mobilenetv3.pth"
OUTPUT_PATH = "/home/user/Desktop/ARCORE_APP/video_01_red/inference_result.mp4"

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
    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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
            preds = model(input_tensor) # [1, 18]
            
        preds = preds.cpu().numpy()[0] # [18,]
        
        # Draw
        h, w = frame_rot.shape[:2]
        pts = []
        for i in range(0, 18, 2):
            # Denormalize
            px = int(preds[i] * w)
            py = int(preds[i+1] * h)
            pts.append((px, py))
            
            # Draw point (Red for centroid, Blue for others)
            color = (0, 0, 255) if i == 0 else (255, 0, 0)
            cv2.circle(frame_rot, (px, py), 4, color, -1)
            
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
