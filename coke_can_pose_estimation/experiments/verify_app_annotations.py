import os
import json
import cv2
import numpy as np
import argparse

def draw_labeled_bbox(img, points, color):
    h, w, _ = img.shape

    # Vertices 1-8 (Front: 1-4, Back: 5-8)
    # 0: Center, 1: FBL, 2: FBR, 3: FTR, 4: FTL, 5: BBL, 6: BBR, 7: BTR, 8: BTL
    front_indices = [1, 2, 3, 4]
    back_indices  = [5, 6, 7, 8]

    # Draw Faces (Lines)
    def draw_face(indices):
        for i in range(4):
            idx1, idx2 = indices[i], indices[(i+1)%4]
            p1 = points[idx1]; p2 = points[idx2]
            cv2.line(img, (int(p1[0]*w), int(p1[1]*h)), (int(p2[0]*w), int(p2[1]*h)), color, 2)

    draw_face(front_indices)
    draw_face(back_indices)

    # Connecting Edges (1-5, etc)
    for i in range(4):
        p1 = points[front_indices[i]]
        p2 = points[back_indices[i]]
        cv2.line(img, (int(p1[0]*w), int(p1[1]*h)), (int(p2[0]*w), int(p2[1]*h)), color, 2)

    # Draw Center (0)
    cx, cy = int(points[0][0] * w), int(points[0][1] * h)
    cv2.circle(img, (cx, cy), 5, (255, 255, 255), -1)

def align_keypoints(kpts, img_w, img_h, json_w=640, json_h=480):
    # 1. Rotation CW (90 degrees)
    # The app logic seems to capture in 640x480 (Landscape)
    # But images are saved in Portrait (1440x2617 or similar)
    kpts_rot = np.zeros_like(kpts)
    kpts_rot[:, 0] = 1.0 - kpts[:, 1]
    kpts_rot[:, 1] = kpts[:, 0]

    sens_w_rot, sens_h_rot = json_h, json_w # 480, 640

    # 2. Scale and Offset (Letterboxing/Crop handling)
    scale = max(img_w / sens_w_rot, img_h / sens_h_rot)
    proj_w = sens_w_rot * scale
    proj_h = sens_h_rot * scale

    off_x = (proj_w - img_w) / 2.0
    off_y = (proj_h - img_h) / 2.0

    kpts_screen = np.zeros_like(kpts_rot)
    kpts_screen[:, 0] = ((kpts_rot[:, 0] * proj_w) - off_x) / img_w
    kpts_screen[:, 1] = ((kpts_rot[:, 1] * proj_h) - off_y) / img_h

    return kpts_screen

def verify(annotations_file, images_dir, output_dir, num_frames=20):
    if not os.path.exists(annotations_file):
        print(f"Error: {annotations_file} not found")
        return

    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Verifying {min(len(annotations), num_frames)} frames from {images_dir} using {annotations_file}...")

    # Sort annotations by frame number/image name if needed, but usually they are sequential
    # or we just iterate through them.

    count = 0
    for i, entry in enumerate(annotations):
        if count >= num_frames:
            break
            
        img_name = entry.get('image_path') or entry.get('image') # handle different json keys
        
        # If image path is full path, extract basename, or join with images_dir
        if not img_name:
             # Fallback if image name is not in json, assume frame_XXXX.jpg based on index?? 
             # But usually it is there. Let's check the structure if this fails.
             print(f"Warning: No image name in annotation entry {i}")
             continue

        img_basename = os.path.basename(img_name)
        img_path = os.path.join(images_dir, img_basename)
        
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Failed to load {img_path}")
            continue

        real_h, real_w, _ = img.shape

        # Support both 'keypoints_2d' and 'keypoints' keys
        kpts_raw = entry.get('keypoints_2d') or entry.get('keypoints')
        if not kpts_raw:
             print(f"Warning: No keypoints in annotation entry {i}")
             continue
             
        kpts_raw = np.array(kpts_raw)[:, :2]

        # Apply alignment logic
        # Check if intrinsics exist
        intrinsics = entry.get('camera_intrinsics', {})
        w = intrinsics.get('image_width', 640) # default if missing
        h = intrinsics.get('image_height', 480)

        kpts_aligned = align_keypoints(
            kpts_raw, real_w, real_h,
            w, h
        )

        draw_labeled_bbox(img, kpts_aligned, (0, 255, 0)) # Green for GT
        
        # Draw raw points too for debug (red)
        # Scale raw points to image dims directly to see if they match without rot
        # parsed_raw = []
        # for p in kpts_raw:
        #     parsed_raw.append([p[0] * real_w, p[1] * real_h])
        # draw_labeled_bbox(img, np.array(parsed_raw)/[real_w, real_h], (0, 0, 255))

        output_path = os.path.join(output_dir, f"verify_{img_basename}")
        cv2.imwrite(output_path, img)
        count += 1

    print(f"Verification images saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_file", type=str, required=True, help="Path to the annotations .json file")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to the directory containing image frames")
    parser.add_argument("--output_dir", type=str, default="verification_results")
    parser.add_argument("--num_frames", type=int, default=20)
    args = parser.parse_args()

    verify(args.annotations_file, args.images_dir, args.output_dir, args.num_frames)