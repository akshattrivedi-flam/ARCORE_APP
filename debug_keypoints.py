import cv2
import json
import os
import numpy as np

# Path to data
JSON_PATH = "/home/user/Desktop/ARCORE_APP/DATASET_TRAINING/red/video_10_red/annotations.json"
IMG_PATH = "/home/user/Desktop/ARCORE_APP/DATASET_TRAINING/red/video_10_red/frame_0000.jpg"
OUTPUT_PATH = "debug_keypoints_order.jpg"

def main():
    with open(JSON_PATH, 'r') as f:
        anns = json.load(f)
        
    # Get frame 0
    ann = anns[0]
    
    img = cv2.imread(IMG_PATH)
    if img is None:
        print(f"Failed to load {IMG_PATH}")
        # Try local path if partial path provided
        return

    h, w = img.shape[:2]
    kpts = ann['keypoints_2d']
    
    # Draw points with indices
    for i, kp in enumerate(kpts):
        x = int(kp[0] * w)
        y = int(kp[1] * h)
        
        # Color: 0=Red, 1-8=Green
        color = (0, 0, 255) if i == 0 else (0, 255, 0)
        
        cv2.circle(img, (x, y), 5, color, -1)
        cv2.putText(img, str(i), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.0, (255, 255, 0), 2)

    cv2.imwrite(OUTPUT_PATH, img)
    print(f"Saved visualization to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
