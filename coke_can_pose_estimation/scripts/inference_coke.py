import torch
import cv2
import numpy as np
import os
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

# --------------------------------------------------
# MODEL DEFINITIONS
# --------------------------------------------------

class CokeTrackerReduced(nn.Module):
    def __init__(self):
        super(CokeTrackerReduced, self).__init__()
        self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        num_features = self.backbone.classifier[0].in_features
        
        self.regressor_2d = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.Hardswish(),
            nn.Dropout(p=0.2), 
            nn.Linear(512, 18), 
            nn.Sigmoid() 
        )
        
        self.regressor_3d = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.Hardswish(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 27) 
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.Hardswish(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1)
        )
        self.backbone.classifier = nn.Identity()

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        kpts_2d = self.regressor_2d(x)
        kpts_3d = self.regressor_3d(x)
        conf = self.classifier(x)
        return kpts_2d, kpts_3d, conf

# --------------------------------------------------
# LOADING UTILS
# --------------------------------------------------

def load_detector(models_root):
    run_folders = [d for d in os.listdir(models_root) if "run_detector" in d]
    run_folders.sort(reverse=True)
    for run in run_folders:
        ckpt_path = os.path.join(models_root, run, "final", "coke_detector.pth")
        if os.path.exists(ckpt_path):
            print(f"Loading Detector: {ckpt_path}")
            model = ssdlite320_mobilenet_v3_large(weights="DEFAULT")
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            model.to(DEVICE).eval()
            return model
    return None

def load_tracker(models_root):
    run_folders = [d for d in os.listdir(models_root) if "run_tracker" in d]
    run_folders.sort(reverse=True)
    for run in run_folders:
        ckpt_path = os.path.join(models_root, run, "final", "model_3d.pth")
        if os.path.exists(ckpt_path):
            print(f"Loading Tracker: {ckpt_path}")
            model = CokeTrackerReduced()
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            model.to(DEVICE).eval()
            return model
    return None

# --------------------------------------------------
# GEOMETRY UTILS
# --------------------------------------------------

half_w = 0.033
half_h = 0.0615
half_d = 0.033

CANONICAL_BOX_3D = np.array([
    [0, 0, 0],                  # 0: Center
    [half_w, half_h, half_d],   # 1: Front Top Right
    [half_w, -half_h, half_d],  # 2: Front Bottom Right
    [-half_w, -half_h, half_d], # 3: Front Bottom Left
    [-half_w, half_h, half_d],  # 4: Front Top Left
    [half_w, half_h, -half_d],  # 5: Back Top Right
    [half_w, -half_h, -half_d], # 6: Back Bottom Right
    [-half_w, -half_h, -half_d],# 7: Back Bottom Left
    [-half_w, half_h, -half_d]  # 8: Back Top Left
], dtype=np.float32)

CAMERA_MATRIX = np.array([[430.0, 0, 240.0], [0, 430.0, 320.0], [0, 0, 1]], dtype=np.float32)
DIST_COEFFS = np.zeros(5)

class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.current = None
    def update(self, target):
        if self.current is None: self.current = target
        else: self.current = self.alpha * target + (1 - self.alpha) * self.current
        return self.current

def solve_pnp(kpts_2d_global, frame_size):
    base_w, base_h = 480, 640
    curr_w, curr_h = frame_size
    scale_x = curr_w / base_w
    scale_y = curr_h / base_h
    
    cam_mat = CAMERA_MATRIX.copy()
    cam_mat[0, 0] *= scale_x; cam_mat[1, 1] *= scale_y
    cam_mat[0, 2] *= scale_x; cam_mat[1, 2] *= scale_y
    
    try:
        success, rvec, tvec = cv2.solvePnP(
            CANONICAL_BOX_3D, kpts_2d_global, cam_mat, DIST_COEFFS, flags=cv2.SOLVEPNP_ITERATIVE
        )
        return success, rvec, tvec, cam_mat, None
    except Exception:
        return False, None, None, None, None

def draw_3d_box(img, rvec, tvec, cam_mat):
    points_3d = CANONICAL_BOX_3D[1:] # Corners 1-8
    projected, _ = cv2.projectPoints(points_3d, rvec, tvec, cam_mat, DIST_COEFFS)
    pts = [tuple(map(int, p.ravel())) for p in projected]
    
    # 1-4 Front (Green)
    for i in range(4):
        cv2.line(img, pts[i], pts[(i+1)%4], (0, 255, 0), 2)
    # 5-8 Back (Red)
    for i in range(4):
        cv2.line(img, pts[i+4], pts[((i+1)%4)+4], (0, 0, 255), 2)
    # Connect (Blue)
    for i in range(4):
        cv2.line(img, pts[i], pts[i+4], (255, 0, 0), 2)
        
    cv2.drawFrameAxes(img, cam_mat, DIST_COEFFS, rvec, tvec, 0.08) 

def extract_crop(img, center, size):
    h, w = img.shape[:2]
    cx, cy = center
    radius = size / 2
    x1 = int(cx - radius); y1 = int(cy - radius)
    x2 = int(cx + radius); y2 = int(cy + radius)
    
    pad_l = max(0, -x1); pad_t = max(0, -y1)
    pad_r = max(0, x2 - w); pad_b = max(0, y2 - h)
    
    bx1 = max(0, x1); by1 = max(0, y1)
    bx2 = min(w, x2); by2 = min(h, y2)
    crop = img[by1:by2, bx1:bx2]
    
    if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0,0,0))
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE)), (x1, y1), size

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "../models")
    video_path = os.path.join(base_dir, "../inference/videos/testing_detector.mp4")
    output_path = os.path.join(base_dir, "../inference/results/visualizations/inference_test_final_landscape.mp4")
    
    detector = load_detector(models_dir)
    tracker = load_tracker(models_dir)
    if detector is None or tracker is None: return

    cap = cv2.VideoCapture(video_path)
    # Original Dimensions (Landscape 1280x720)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # OUTPUT: Landscape (width, height)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    tracker_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    STATE = "SEARCH"
    roi_center = np.array([0., 0.]); roi_size = 0.
    lost_count = 0
    
    # Strong Smoothing
    f_roi_center = LowPassFilter(0.1)
    f_roi_size = LowPassFilter(0.1)
    f_kpts = LowPassFilter(0.1)
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 1. PROCESS: Rotate CW (Portrait) for Model
        frame_rot = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        h, w = frame_rot.shape[:2]
        vis_frame = frame_rot.copy()
        
        if STATE == "SEARCH":
            f_roi_center.current = None; f_roi_size.current = None; f_kpts.current = None
            
            img_rgb = cv2.cvtColor(frame_rot, cv2.COLOR_BGR2RGB)
            inp = torch.from_numpy(img_rgb).permute(2,0,1).float().div(255.0).unsqueeze(0).to(DEVICE)
            with torch.no_grad(): dets = detector(inp)[0]
            
            best_score = 0; best_box = None
            scan_radius = min(h,w) * 0.45; mh, mw = h//2, w//2
            
            for i, score in enumerate(dets['scores']):
                if score > 0.15 and dets['labels'][i] == 1:
                    box = dets['boxes'][i].cpu().numpy()
                    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                    if np.sqrt((cx-mw)**2 + (cy-mh)**2) < scan_radius:
                        if score > best_score: best_score = score; best_box = box
            
            if best_box is not None:
                x1, y1, x2, y2 = best_box
                box_w, box_h = x2-x1, y2-y1
                
                init_cx = (x1+x2)/2; init_cy = (y1+y2)/2
                init_size = max(box_w, box_h) * 1.5
                
                roi_center = f_roi_center.update(np.array([init_cx, init_cy]))
                roi_size = f_roi_size.update(init_size)
                STATE = "TRACK"; lost_count = 0
                cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            else:
                cv2.circle(vis_frame, (mh, mw), int(scan_radius), (0, 0, 255), 1)
                
        elif STATE == "TRACK":
            crop, (off_x, off_y), scale = extract_crop(frame_rot, roi_center, roi_size)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            inp = tracker_norm(crop_rgb).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                kpts_2d_n, _, conf_logits = tracker(inp)
                conf = torch.sigmoid(conf_logits).item()
                
            if conf > 0.1: # Forced low tolerance
                lost_count = 0
                kpts_2d = kpts_2d_n[0].cpu().numpy().reshape(9, 2)
                
                if not np.isnan(kpts_2d).any(): 
                    kpts_global = []
                    for kp in kpts_2d:
                        factor = scale / float(IMG_SIZE)
                        gx = off_x + (kp[0] * IMG_SIZE) * factor
                        gy = off_y + (kp[1] * IMG_SIZE) * factor
                        kpts_global.append([gx, gy])
                    
                    raw_kpts = np.array(kpts_global, dtype=np.float32)
                    kpts_global = f_kpts.update(raw_kpts)
                    
                    min_x, max_x = np.min(kpts_global[:,0]), np.max(kpts_global[:,0])
                    min_y, max_y = np.min(kpts_global[:,1]), np.max(kpts_global[:,1])
                    obj_cx = (min_x+max_x)/2; obj_cy = (min_y+max_y)/2
                    obj_size = max(max_x-min_x, max_y-min_y) * 1.8
                    
                    roi_center = f_roi_center.update(np.array([obj_cx, obj_cy]))
                    roi_size = f_roi_size.update(obj_size)
                    
                    for idx, kp in enumerate(kpts_global):
                        cv2.circle(vis_frame, (int(kp[0]), int(kp[1])), 3, (0, 255, 255), -1)

                    success, rvec, tvec, cam_mat, inliers = solve_pnp(kpts_global, (w, h))
                    if success:
                        draw_3d_box(vis_frame, rvec, tvec, cam_mat)
                        if frame_idx % 30 == 0: print(f"Frame {frame_idx}: PnP OK")
            else:
                lost_count += 1
                if lost_count > 30: STATE = "SEARCH"; print("Lost tracking.")
            
            if roi_center is not None:
                tl = (int(roi_center[0]-roi_size/2), int(roi_center[1]-roi_size/2))
                br = (int(roi_center[0]+roi_size/2), int(roi_center[1]+roi_size/2))
                cv2.rectangle(vis_frame, tl, br, (255,0,0), 1)

        # 2. FINALIZE: Rotate Back to Landscape (CCW) so output matches original video
        vis_frame_final = cv2.rotate(vis_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        out.write(vis_frame_final)
        frame_idx += 1
    
    cap.release(); out.release()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()
