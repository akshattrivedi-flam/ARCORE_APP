import cv2
import time

video1_path = "/home/user/Desktop/ARCORE_APP/coke_can_pose_estimation/inference/results/visualizations/inference_ssd_experiment_red_01_final.mp4"
video2_path = "/home/user/Desktop/ARCORE_APP/coke_can_pose_estimation/inference/results/visualizations/inference_ssd_red_box_only_01_final.mp4"

cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

if not cap1.isOpened() or not cap2.isOpened():
    raise RuntimeError("Error opening one of the videos")

# Metadata resolution
w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps1 = cap1.get(cv2.CAP_PROP_FPS)

w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps2 = cap2.get(cv2.CAP_PROP_FPS)

print("=== VIDEO METADATA ===")
print(f"Video 1: {w1} x {h1} @ {fps1:.2f} FPS")
print(f"Video 2: {w2} x {h2} @ {fps2:.2f} FPS")

fps = min(fps1, fps2)
delay = 1.0 / fps

paused = False
frame1 = None
frame2 = None
first_frame = True

while True:
    if not paused:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        if first_frame:
            h1f, w1f = frame1.shape[:2]
            h2f, w2f = frame2.shape[:2]

            print("\n=== ACTUAL FRAME SHAPE ===")
            print(f"Video 1 frame: {w1f} x {h1f}")
            print(f"Video 2 frame: {w2f} x {h2f}")
            print("=========================\n")

            first_frame = False

    cv2.imshow("Video 1", frame1)
    cv2.imshow("Video 2", frame2)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused
        time.sleep(0.2)

    if not paused:
        time.sleep(delay)

cap1.release()
cap2.release()
cv2.destroyAllWindows()
