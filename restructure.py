import os
import shutil
from pathlib import Path

# Config
SOURCE_ROOT = "/home/user/Desktop/ARCORE_APP"
DATASET_SOURCE = os.path.join(SOURCE_ROOT, "DATASET_TRAINING")
TARGET_ROOT = os.path.join(SOURCE_ROOT, "coke_can_pose_estimation")

# Structure definition
DIRS_TO_CREATE = [
    "data/raw/videos/red",
    "data/raw/videos/blue",
    "data/raw/videos/silver",
    "data/raw/frames/red",
    "data/raw/frames/blue",
    "data/raw/frames/silver",
    "data/raw/annotations/red",
    "data/raw/annotations/blue",
    "data/raw/annotations/silver",
    "data/processed/train",
    "data/processed/val",
    "data/processed/test",
    "data/metadata",
    "models/checkpoints",
    "models/final",
    "inference/images",
    "inference/videos",
    "inference/results/visualizations",
    "scripts",
    "configs",
    "experiments"
]

def create_structure():
    if os.path.exists(TARGET_ROOT):
        print(f"Target directory {TARGET_ROOT} already exists. Merging/Overwriting...")
    
    for d in DIRS_TO_CREATE:
        os.makedirs(os.path.join(TARGET_ROOT, d), exist_ok=True)
    
    # Touch files
    Path(os.path.join(TARGET_ROOT, "README.md")).touch()
    Path(os.path.join(TARGET_ROOT, "requirements.txt")).touch()

def move_dataset_category(category):
    src_cat_dir = os.path.join(DATASET_SOURCE, category)
    if not os.path.exists(src_cat_dir):
        print(f"Category {category} not found in {DATASET_SOURCE}")
        return

    print(f"Processing category: {category}...")
    
    # Iterate over video directories (e.g., video_01_red)
    for video_folder in os.listdir(src_cat_dir):
        src_video_path = os.path.join(src_cat_dir, video_folder)
        if not os.path.isdir(src_video_path):
            continue
            
        # Target naming
        # video_folder is like "video_10_red"
        # We want to keep it as the identifier
        
        # 1. Move Video
        src_vid_file = os.path.join(src_video_path, "video_raw.mp4")
        if os.path.exists(src_vid_file):
            tgt_vid_path = os.path.join(TARGET_ROOT, f"data/raw/videos/{category}/{video_folder}.mp4")
            shutil.copy2(src_vid_file, tgt_vid_path)
            
        # 2. Move Annotations
        src_ann_file = os.path.join(src_video_path, "annotations.json")
        if os.path.exists(src_ann_file):
            tgt_ann_path = os.path.join(TARGET_ROOT, f"data/raw/annotations/{category}/{video_folder}_annotations.json")
            shutil.copy2(src_ann_file, tgt_ann_path)
            
        # 3. Move Frames
        # Create dedicated frame folder
        tgt_frames_dir = os.path.join(TARGET_ROOT, f"data/raw/frames/{category}/{video_folder}")
        os.makedirs(tgt_frames_dir, exist_ok=True)
        
        # Copy all jpgs
        jpgs = [f for f in os.listdir(src_video_path) if f.endswith(".jpg")]
        for jpg in jpgs:
            shutil.copy2(os.path.join(src_video_path, jpg), os.path.join(tgt_frames_dir, jpg))
            
    print(f"Finished split for {category}.")

def move_models_and_scripts():
    print("Moving models and scripts...")
    
    # Checkpoints
    for f in os.listdir(DATASET_SOURCE):
        if f.startswith("coke_tracker_checkpoint") and f.endswith(".pth"):
            shutil.copy2(os.path.join(DATASET_SOURCE, f), os.path.join(TARGET_ROOT, "models/checkpoints", f))
            
    # Final Model
    final_model = os.path.join(DATASET_SOURCE, "coke_tracker_mobilenetv3.pth")
    if os.path.exists(final_model):
        shutil.copy2(final_model, os.path.join(TARGET_ROOT, "models/final/model.pth"))
        
    # Scripts
    scripts_map = {
        "train_coke_tracker.py": "scripts/train_coke_tracker.py",
        "inference_coke.py": "scripts/inference_coke.py",
        "verify_results.py": "scripts/verify_results.py",
        "sanity_check.py": "scripts/sanity_check.py"
    }
    
    for src, dest in scripts_map.items():
        src_path = os.path.join(SOURCE_ROOT, src)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(TARGET_ROOT, dest))

def list_contents():
    print("New Structure Summary:")
    for root, dirs, files in os.walk(TARGET_ROOT):
        level = root.replace(TARGET_ROOT, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        # Limit file print
        if len(files) > 5:
             print('{}{} files...'.format(subindent, len(files)))
        else:
            for f in files:
                print('{}{}'.format(subindent, f))

if __name__ == "__main__":
    create_structure()
    move_dataset_category("red")
    move_dataset_category("blue")
    move_dataset_category("silver")
    move_models_and_scripts()
    # list_contents()
    print("Restructuring Complete.")
