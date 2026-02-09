import subprocess
import json

video_path = "/home/user/Desktop/ARCORE_APP/coke_can_pose_estimation/experiments/video_raw.mp4"

cmd = [
    "ffprobe",
    "-v", "error",
    "-select_streams", "v:0",
    "-show_entries", "stream=width,height",
    "-of", "json",
    video_path
]

result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
info = json.loads(result.stdout)

width = info["streams"][0]["width"]
height = info["streams"][0]["height"]

print(f"Resolution: {width} x {height}")