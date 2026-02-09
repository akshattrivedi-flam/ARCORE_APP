import torch
import cv2
import numpy as np
import os
from train_coke_tracker import CokeDataset, DATA_ROOT, IMG_SIZE

# Function to inverse normalize and draw keypoints
def visualize_crop(img_tensor, kpts_tensor, label):
    # Denormalize Image (approximate for vis)
    img = img_tensor.permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8).copy()
    
    # Draw Keypoints (Green for Positive, Red/None for Negative)
    if label.item() > 0.5:
        kpts = kpts_tensor.numpy()
        h, w = img.shape[:2]
        for i in range(0, 18, 2):
            x = int(kpts[i] * w)
            y = int(kpts[i+1] * h)
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
            
        # Draw Box
        xs = kpts[0::2]
        ys = kpts[1::2]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        cv2.rectangle(img, 
                     (int(min_x*w), int(min_y*h)), 
                     (int(max_x*w), int(max_y*h)), 
                     (255, 0, 0), 1)
                     
    return img

def main():
    print("Verifying Crops...")
    dataset = CokeDataset(root_dir=DATA_ROOT, transform=None)
    
    # We apply the transforms manually to visualize the raw crop output first, 
    # but the Dataset class already does resizing and everything.
    # Wait, the Dataset class returns a PIL image.
    
    # Let's instantiate proper transforms as per training
    from torchvision import transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Retarget dataset transform
    dataset.transform = train_transform
    
    if not os.path.exists("debug_crops"):
        os.makedirs("debug_crops")
        
    print(f"Dataset Size: {len(dataset)}")
    
    # Save 10 Positives and 5 Negatives
    pos_count = 0
    neg_count = 0
    
    indices = np.random.permutation(len(dataset))
    
    for i in indices:
        img, kpts, label = dataset[i]
        
        vis = visualize_crop(img, kpts, label)
        
        if label.item() > 0.5 and pos_count < 10:
            cv2.imwrite(f"debug_crops/pos_{pos_count}.jpg", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            pos_count += 1
            print(f"Saved Positive {pos_count}")
        elif label.item() < 0.5 and neg_count < 5:
            cv2.imwrite(f"debug_crops/neg_{neg_count}.jpg", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            neg_count += 1
            print(f"Saved Negative {neg_count}")
            
        if pos_count >= 10 and neg_count >= 5:
            break
            
    print("Verification Done. Check 'debug_crops' folder.")

if __name__ == "__main__":
    main()
