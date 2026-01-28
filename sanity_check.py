import os
import json
from PIL import Image
import torch

DATA_DIR = "/home/user/Desktop/ARCORE_APP/seq_120737245"
JSON_FILE = os.path.join(DATA_DIR, "annotations.json")

def sanity_check():
    print("--- SANITY CHECK STARTING ---")
    
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Data directory {DATA_DIR} not found.")
        return
    
    if not os.path.exists(JSON_FILE):
        print(f"ERROR: JSON annotation file {JSON_FILE} not found.")
        return
    
    print(f"SUCCESS: Data directory and JSON file found.")

    with open(JSON_FILE, 'r') as f:
        try:
            annotations = json.load(f)
            print(f"SUCCESS: JSON file parsed. Found {len(annotations)} frames.")
        except Exception as e:
            print(f"ERROR: Failed to parse JSON: {e}")
            return

    if len(annotations) == 0:
        print("ERROR: No annotations found in JSON.")
        return

    # Check first 5 items
    for i in range(min(5, len(annotations))):
        entry = annotations[i]
        img_name = entry.get('image')
        if not img_name:
            print(f"ERROR: Entry {i} missing image name.")
            continue
        
        img_path = os.path.join(DATA_DIR, img_name)
        if not os.path.exists(img_path):
            print(f"ERROR: Image not found at {img_path}")
            continue
        
        try:
            with Image.open(img_path) as img:
                img.verify()
            print(f"SUCCESS: Image {img_name} is valid and can be opened.")
        except Exception as e:
            print(f"ERROR: Image {img_name} is corrupted or invalid: {e}")
            continue

        kpts = entry.get('keypoints_2d')
        if not kpts or len(kpts) != 9:
            print(f"ERROR: Entry {i} has invalid keypoints count: {len(kpts) if kpts else 0}")
        else:
            print(f"SUCCESS: Entry {i} has correct number of keypoints (9).")

    print("--- SANITY CHECK COMPLETE ---")

if __name__ == "__main__":
    sanity_check()
