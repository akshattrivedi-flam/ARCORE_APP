import os
import torch
import cv2
import json
import numpy as np
from torchvision import transforms
from PIL import Image
from train_objectron import ObjectronMobileNetV3

# --- CONFIGURATION ---
MODEL_PATH = "/home/user/Desktop/ARCORE_APP/bottle_objectron_overfit.pth"
IMAGE_PATH = "/home/user/Desktop/ARCORE_APP/IMG_20260128_164324.jpg"
OUTPUT_DIR = "/home/user/Desktop/ARCORE_APP/test_botte_off_ground"
IMAGE_SIZE = 224
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def draw_pinpoints_only(img, points, color):
    """
    Specialized Pinpoint Visualization:
    Draws 9 dots and labels them with their (x, y) pixel coordinates.
    """
    h, w, _ = img.shape
    
    for i in range(9):
        # Calculate pixel coordinates
        px_float, py_float = points[i][0] * w, points[i][1] * h
        px, py = int(px_float), int(py_float)
        
        # 1. Draw the pinpoint
        cv2.circle(img, (px, py), 6, color, -1)
        
        # 2. Draw the coordinate text (X, Y)
        coord_text = f"P{i}: ({px}, {py})"
        
        # Text background for readability
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        (text_w, text_h), _ = cv2.getTextSize(coord_text, font, font_scale, 1)
        cv2.rectangle(img, (px + 10, py - text_h - 5), (px + 10 + text_w, py + 5), (0, 0, 0), -1)
        cv2.putText(img, coord_text, (px + 10, py), font, font_scale, (255, 255, 255), 1)

def run_inference():
    print(f"Loading PTH model for pinpoints (with ARCore Alignment)...")
    model = ObjectronMobileNetV3().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
    except Exception as e:
        print(f"Error: {e}")
        return

    # --- IMAGE ALIGNMENT (ARCore Emulation) ---
    pil_img = Image.open(IMAGE_PATH).convert('RGB')
    orig_w, orig_h = pil_img.size
    
    # Target Aspect Ratio from Training (1080 / 2023)
    target_ratio = 1080 / 2023
    current_ratio = orig_w / orig_h
    
    if current_ratio > target_ratio:
        # Image is too wide, crop sides
        new_w = int(orig_h * target_ratio)
        left = (orig_w - new_w) // 2
        right = left + new_w
        top, bottom = 0, orig_h
    else:
        # Image is too tall, crop top/bottom
        new_h = int(orig_w / target_ratio)
        top = (orig_h - new_h) // 2
        bottom = top + new_h
        left, right = 0, orig_w
    
    # Crop to match training viewport "shape"
    cropped_img = pil_img.crop((left, top, right, bottom))
    crop_w, crop_h = cropped_img.size
    
    # Preprocess the cropped version
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(cropped_img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        pth_pred = model(input_tensor).cpu().numpy()[0]
        pth_result_cropped = pth_pred.reshape(9, 2)

    # --- INVERSE MAPPING ---
    # Convert normalized coords (0-1) of the CROP to normalized coords of the ORIGINAL
    pth_result_orig = np.zeros_like(pth_result_cropped)
    for i in range(9):
        # x_orig = (x_crop * crop_w + offset_left) / orig_w
        pth_result_orig[i, 0] = (pth_result_cropped[i, 0] * crop_w + left) / orig_w
        pth_result_orig[i, 1] = (pth_result_cropped[i, 1] * crop_h + top) / orig_h

    # --- SAVE JSON ---
    json_path = os.path.join(OUTPUT_DIR, "keypoints_aligned.json")
    with open(json_path, 'w') as f:
        json.dump({"points_normalized": pth_result_orig.tolist()}, f, indent=4)

    # --- VISUALIZATION ---
    orig_img = cv2.imread(IMAGE_PATH)
    if orig_img is None: return

    img_viz = orig_img.copy()
    # Use the Inverse Mapped coordinates to draw on the original uncropped image
    draw_pinpoints_only(img_viz, pth_result_orig, (0, 255, 0)) # Green
    
    # Save results
    out_path = os.path.join(OUTPUT_DIR, "result_pinpoints_aligned.png")
    cv2.imwrite(out_path, img_viz)
    print(f"Aligned Result saved to {out_path}")


if __name__ == "__main__":
    run_inference()
