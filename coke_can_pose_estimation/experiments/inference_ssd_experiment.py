import torch
import cv2
import numpy as np
import os
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))

VIDEO_PATH = os.path.join(
    current_dir, "../inference/videos/testing_detector.mp4"
)
OUTPUT_PATH = os.path.join(
    current_dir,
    "../inference/videos/inference_ssd_experiment_testing_detector.mp4"
)
MODELS_ROOT = os.path.join(current_dir, "../models")

TARGET_CHECKPOINT = "coke_detector.pth"


# --------------------------------------------------
# LOAD FINAL DETECTOR
# --------------------------------------------------
def load_detector_final():
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

    # Load from the latest run (assuming that's the one that finished)
    run_folders.sort(reverse=True)
    
    for run_folder in run_folders:
        # Check 'final' folder first
        ckpt_dir = os.path.join(MODELS_ROOT, run_folder, "final")
        if not os.path.exists(ckpt_dir):
            continue

        ckpt_path = os.path.join(ckpt_dir, TARGET_CHECKPOINT)
        if os.path.exists(ckpt_path):
            print(f"Loading FINAL model: {ckpt_path}")

            try:
                model = ssdlite320_mobilenet_v3_large(weights="DEFAULT")

                state_dict = torch.load(ckpt_path, map_location=DEVICE)
                model.load_state_dict(state_dict)

                model.to(DEVICE)
                model.eval()
                return model

            except Exception as e:
                print(f"Error loading model: {e}")
                return None

    print(f"Final model '{TARGET_CHECKPOINT}' not found in any run_detector folder.")
    return None


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    print("Initializing SSD Experiment (Final Model)...")

    detector = load_detector_final()
    if detector is None:
        print("Failed to load detector.")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (height, width), # Portrait
    )

    # ROI Configuration (center scan region)
    roi_margin = 0.5
    roi_x1 = int(width * (0.5 - roi_margin / 2))
    roi_y1 = int(height * (0.5 - roi_margin / 2))
    roi_x2 = int(width * (0.5 + roi_margin / 2))
    roi_y2 = int(height * (0.5 + roi_margin / 2))

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Rotate to Portrait (match training - TRY COUNTERCLOCKWISE if upside down)
        frame_rot = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # 2. Preprocess
        img_rgb = cv2.cvtColor(frame_rot, cv2.COLOR_BGR2RGB)
        img_tensor = (
            torch.from_numpy(img_rgb)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(DEVICE)
        )

        # 3. Inference
        with torch.no_grad():
            detections = detector(img_tensor)[0]

        boxes = detections["boxes"].cpu().numpy()
        scores = detections["scores"].cpu().numpy()
        labels = detections["labels"].cpu().numpy()

        drawn_frame = frame_rot.copy()
        found = False

        # ROI in portrait coordinates
        h_rot, w_rot = drawn_frame.shape[:2]
        roi_w = w_rot * roi_margin
        roi_h = h_rot * roi_margin
        proi_x1 = int((w_rot - roi_w) / 2)
        proi_y1 = int((h_rot - roi_h) / 2)
        proi_x2 = int((w_rot + roi_w) / 2)
        proi_y2 = int((h_rot + roi_h) / 2)

        # cv2.rectangle(drawn_frame, (proi_x1, proi_y1), (proi_x2, proi_y2), (255, 0, 0), 1)

        # Collect valid candidates
        candidates = []
        for i, score in enumerate(scores):
            if score > 0.15 and labels[i] == 1:
                x1, y1, x2, y2 = boxes[i].astype(int)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                if proi_x1 < cx < proi_x2 and proi_y1 < cy < proi_y2:
                    candidates.append((score, boxes[i]))

        # Pick best candidate
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_score, best_box = candidates[0]

            x1, y1, x2, y2 = best_box.astype(int)

            cv2.rectangle(
                drawn_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3,
            )

            cv2.putText(
                drawn_frame,
                f"COKE: {best_score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            found = True

        # 5. Write Portrait Frame directly
        out.write(drawn_frame)

        if frame_idx % 30 == 0:
            print(f"Frame {frame_idx}: Detected? {found}")

        frame_idx += 1

    cap.release()
    out.release()
    print(f"SSD Epoch-32 experiment saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
