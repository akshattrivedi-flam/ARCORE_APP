import os
import json
from PIL import Image

# Config
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(CURRENT_DIR, "../data/raw")
CATEGORIES = ["red", "blue", "silver"]

def sanity_check():
    print("--- SANITY CHECK STARTING ---")
    print(f"Data Root: {DATA_ROOT}")
    
    if not os.path.exists(DATA_ROOT):
        print(f"ERROR: Data Root not found!")
        return

    # Check Directory Structure
    for cat in CATEGORIES:
        for dtype in ["videos", "frames", "annotations"]:
            path = os.path.join(DATA_ROOT, dtype, cat)
            if not os.path.exists(path):
                print(f"ERROR: Missing directory: {path}")
            else:
                count = len(os.listdir(path))
                print(f"OK: {dtype}/{cat} exists ({count} items)")

    # Check Annotations (Red Category)
    ann_dir = os.path.join(DATA_ROOT, "annotations", "red")
    frame_dir = os.path.join(DATA_ROOT, "frames", "red")
    
    json_files = [f for f in os.listdir(ann_dir) if f.endswith(".json")]
    if not json_files:
        print("ERROR: No JSON annotations found in red category.")
        return

    # Check first annotation file
    sample_json = json_files[0]
    sample_path = os.path.join(ann_dir, sample_json)
    print(f"\nChecking sample annotation: {sample_json}")
    
    with open(sample_path, 'r') as f:
        data = json.load(f)
        
    print(f"Found {len(data)} frames in JSON.")
    
    # Check first 5 frames
    video_id = sample_json.replace("_annotations.json", "")
    
    for i in range(min(5, len(data))):
        entry = data[i]
        img_name = entry['image']
        
        # Construct expected frame path
        # frames/red/video_id/frame_name
        img_path = os.path.join(frame_dir, video_id, img_name)
        
        if not os.path.exists(img_path):
             print(f"ERROR: Image not found: {img_path}")
        else:
             print(f"OK: Frame {i} found at {img_path}")
             
        # Check Keypoints
        if 'keypoints_2d' in entry:
            kpts = entry['keypoints_2d']
            if len(kpts) == 9:
                print(f"OK: Frame {i} has 9 keypoints.")
            else:
                print(f"WARNING: Frame {i} has {len(kpts)} keypoints.")

    print("\n--- SANITY CHECK COMPLETE ---")

if __name__ == "__main__":
    sanity_check()
