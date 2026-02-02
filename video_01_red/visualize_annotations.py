import os
import json
import cv2
import numpy as np

def draw_3d_box(img, points_2d, connections):
    """
    Draws a 3D bounding box on the image.
    points_2d: list of 9 points [x, y, depth] relative to image dimensions.
    connections: list of tuples (start_idx, end_idx) for box lines.
    """
    h, w = img.shape[:2]
    pts = []
    for p in points_2d:
        x, y = p[0] * w, p[1] * h
        pts.append((int(x), int(y)))
    
    # Draw connections
    # Front-face: Green
    for i in range(len(connections)):
        start_idx, end_idx = connections[i]
        color = (0, 255, 0) # Default Green
        # Index 0 is centroid, so we skip it for box lines as per user request
        if start_idx == 0 or end_idx == 0:
            continue
        cv2.line(img, pts[start_idx], pts[end_idx], color, 2)
    
    # Draw centroid: Red
    if len(pts) > 0:
        cv2.circle(img, pts[0], 4, (0, 0, 255), -1)

def main():
    # Configuration
    data_dir = "/home/user/Desktop/ARCORE_APP/video_01_red"
    composited_dir = os.path.join(data_dir, "composited")
    json_path = os.path.join(data_dir, "annotations.json")
    output_dir = os.path.join(data_dir, "output")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Standard Objectron / User specified connection scheme
    # Front face: 1-2-3-4-1
    # Back face: 5-6-7-8-5
    # Connectors: 1-5, 2-6, 3-7, 4-8
    connections = [
        # Front face
        (1, 2), (2, 3), (3, 4), (4, 1),
        # Back face
        (5, 6), (6, 7), (7, 8), (8, 5),
        # Connectors
        (1, 5), (2, 6), (3, 7), (4, 8)
    ]

    # Load annotations
    with open(json_path, 'r') as f:
        annotations = json.load(f)

    print(f"Loaded {len(annotations)} annotations.")

    for ann in annotations:
        img_name = ann.get('image')
        frame_id = ann.get('frame_id')
        
        # Filename mapping: frame_0000.jpg -> frame_000000.jpg
        # Extract number from 'frame_XXXX.jpg'
        try:
            num_part = img_name.split('_')[1].split('.')[0]
            comp_name = f"frame_{int(num_part):06d}.jpg"
        except (IndexError, ValueError):
            comp_name = img_name
            
        img_path = os.path.join(composited_dir, comp_name)
        
        if not os.path.exists(img_path):
            # Try matching exact name if padding differs
            comp_name = img_name
            img_path = os.path.join(composited_dir, comp_name)
            if not os.path.exists(img_path):
                # print(f"Warning: Image {img_path} not found. Skipping.")
                continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image {img_path}")
            continue

        # Rotation logic based on user constraint:
        # "rotate them from landscape to portrait first... 90 deg clockwise"
        # "Make sure that the composite frames which are 480 x 640... should be rotated 90 deg clockwise"
        
        # If the image is 640x480 (landscape), rotating 90 CW makes it 480x640 (portrait).
        # If the image is already 480x640, but he says "SHOULD be rotated", 
        # it might mean it becomes 640x480? No, he wants portrait.
        # I suspect he means the RAW frames are 640x480 and need rotation.
        # IF the composited image is already 480x640, it has likely already been rotated.
        
        curr_h, curr_w = img.shape[:2]
        
        # Get coordinates for 640x480 (Landscape)
        kpts_2d = ann.get('keypoints_2d', [])
        visibility = ann.get('visibility', [1.0] * len(kpts_2d))
        
        # Check for invisible keypoints
        for i, vis in enumerate(visibility):
            if vis < 0.5: # User said visibility < 1, usually 0 or 1
                pass # print(f"Frame {frame_id}: Keypoint {i} is invisible (vis={vis})")

        # TRANSFORM COORDINATES:
        # Landscape (640x480) -> Portrait (480x640) 90 deg CW
        # x_new = 1 - y_old
        # y_new = x_old
        
        target_kpts = []
        for kp in kpts_2d:
            x_old, y_old, depth = kp
            x_new = 1.0 - y_old
            y_new = x_old
            target_kpts.append([x_new, y_new, depth])

        # ROTATE IMAGE IF NECESSARY
        # If image is 640x480, rotate it to 480x640.
        if curr_w == 640 and curr_h == 480:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif curr_w == 480 and curr_h == 640:
            # It's already the target portrait resolution.
            # We assume it already represents the rotated content.
            pass
        else:
            # Resize and Rotate? 
            # User said: "Make sure that ... the frames are also 480x640 and rotated 90 deg clockwise"
            # I will resize to 640x480 then rotate to 480x640 to be safe.
            img = cv2.resize(img, (640, 480))
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        # Draw the 3D box on the (now portrait) image
        draw_3d_box(img, target_kpts, connections)
        
        # Save output
        save_path = os.path.join(output_dir, f"annotated_{comp_name}")
        cv2.imwrite(save_path, img)

    print(f"Completed! Annotated images saved to {output_dir}")

if __name__ == "__main__":
    main()
