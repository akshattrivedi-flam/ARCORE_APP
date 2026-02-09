import torch
import cv2
import numpy as np
import os
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))

VIDEO_PATH = os.path.join(
    current_dir, "../data/raw/videos/red/video_21_red.mp4"
)
OUTPUT_PATH = os.path.join(
    current_dir,
    "../inference/results/visualizations/inference_ssd_red_box_only_21_final.mp4"
)
MODELS_ROOT = os.path.join(current_dir, "../models")


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------
def load_latest_detector():
    if not os.path.exists(MODELS_ROOT):
        print("Models directory not found.")
        return None

    run_folders = [
        d for d in os.listdir(MODELS_ROOT)
        if os.path.isdir(os.path.join(MODELS_ROOT, d))
        and "run_detector" in d
    ]

    if not run_folders:
        print("No detector run folders found.")
        return None

    run_folders.sort(reverse=True)

    # Priority: Check FINAL model first in latest runs
    for run in run_folders:
        final_dir = os.path.join(MODELS_ROOT, run, "final")
        final_path = os.path.join(final_dir, "coke_detector.pth")
        
        if os.path.exists(final_path):
            print(f"Using FINAL model: {final_path}")
             # Load
            try:
                model = ssdlite320_mobilenet_v3_large(weights="DEFAULT")
                state_dict = torch.load(final_path, map_location=DEVICE)
                model.load_state_dict(state_dict)
                model.to(DEVICE)
                model.eval()
                return model
            except Exception as e:
                 print(f"Error loading final model: {e}")
                 # Fallback to checkpoints continues below if this fails

    # Fallback: Checkpoints
    for run in run_folders:
        ckpt_dir = os.path.join(MODELS_ROOT, run, "checkpoints")
        if not os.path.exists(ckpt_dir):
            continue

        ckpts = [f for f in os.listdir(ckpt_dir) if "coke_detector" in f]
        if not ckpts:
            continue

        ckpts.sort(key=lambda x: int(x.split("_")[-1].replace(".pth", "")))
        latest_checkpoint = os.path.join(ckpt_dir, ckpts[-1])
        print(f"Using checkpoint: {ckpts[-1]}")

        model = ssdlite320_mobilenet_v3_large(weights="DEFAULT")
        state_dict = torch.load(latest_checkpoint, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        return model

    print("No checkpoints found.")
    return None


# ---------------------------------------------------------
# MAIN INFERENCE
# ---------------------------------------------------------
def main():
    detector = load_latest_detector()
    if detector is None:
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Read first frame to get output size (after rotation)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video.")
        return

    frame_rot = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    h_rot, w_rot = frame_rot.shape[:2]

    out = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w_rot, h_rot)
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Rotate to portrait (unchanged behavior)
        frame_rot = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Preprocess
        img_rgb = cv2.cvtColor(frame_rot, cv2.COLOR_BGR2RGB)
        img_tensor = (
            torch.from_numpy(img_rgb)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(DEVICE)
        )

        # Inference
        with torch.no_grad():
            detections = detector(img_tensor)[0]

        boxes = detections["boxes"].cpu().numpy()
        scores = detections["scores"].cpu().numpy()
        labels = detections["labels"].cpu().numpy()

        # -------------------------------------------------
        # 🖤 PURE BLACK CANVAS
        # -------------------------------------------------
        drawn_frame = np.zeros_like(frame_rot)

        candidates = []
        for i, score in enumerate(scores):
            if score > 0.3 and labels[i] == 1:
                candidates.append((score, boxes[i]))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_score, best_box = candidates[0]

            x1, y1, x2, y2 = best_box.astype(int)

            # Clamp
            h, w = frame_rot.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # 🟩 DRAW ONLY THE BOX
            cv2.rectangle(
                drawn_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3,
            )

            cv2.putText(
                drawn_frame,
                f"COKE {best_score:.2f}",
                (x1, max(y1 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        out.write(drawn_frame)

        if frame_idx % 30 == 0:
            print(f"Frame {frame_idx} processed")

        frame_idx += 1

    cap.release()
    out.release()
    print(f"Black-canvas SSD output saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
