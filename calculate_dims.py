import json
import numpy as np
import os

JSON_PATH = "/home/user/Desktop/ARCORE_APP/video_01_red/annotations.json"

def calculate_dimensions():
    with open(JSON_PATH, 'r') as f:
        annotations = json.load(f)

    widths = []
    heights = []
    depths = []

    for ann in annotations:
        kpts_3d = np.array(ann['keypoints_3d']) # 9 points: Centroid + 8 corners
        if len(kpts_3d) != 9:
            continue
            
        # Order in Objectron:
        # 0: Centroid
        # 1-4: Front Face
        # 5-8: Back Face
        
        # Width (X): dist(1, 4) or dist(2, 3)
        w1 = np.linalg.norm(kpts_3d[1] - kpts_3d[4])
        w2 = np.linalg.norm(kpts_3d[2] - kpts_3d[3])
        widths.append((w1 + w2) / 2)

        # Height (Y): dist(1, 2) or dist(4, 3)
        h1 = np.linalg.norm(kpts_3d[1] - kpts_3d[2])
        h2 = np.linalg.norm(kpts_3d[4] - kpts_3d[3])
        heights.append((h1 + h2) / 2)
        
        # Depth (Z): dist(1, 5) or dist(2, 6)
        d1 = np.linalg.norm(kpts_3d[1] - kpts_3d[5])
        d2 = np.linalg.norm(kpts_3d[2] - kpts_3d[6])
        depths.append((d1 + d2) / 2)

    avg_w = np.mean(widths)
    avg_h = np.mean(heights)
    avg_d = np.mean(depths)

    print(f"Calculated Dimensions (Meters):")
    print(f"Width (X): {avg_w:.4f}")
    print(f"Height (Y): {avg_h:.4f}")
    print(f"Depth (Z): {avg_d:.4f}")
    
    # Check aspect ratio
    print(f"Aspect Ratio H/W: {avg_h/avg_w:.2f}")

if __name__ == "__main__":
    calculate_dimensions()
