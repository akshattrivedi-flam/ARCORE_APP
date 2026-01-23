# ARCore Dataset Capture App

This application is designed to capture annotated datasets for 3D object detection (Objectron format), specifically for beverage cans.

## Features
- **Ground Plane Detection**: Uses ARCore to find surfaces.
- **Adjustable Bounding Box**: Control scale(X,Y,Z) and rotation(Yaw) to fit the physical object.
- **Dataset Export**: Saves frames as JPEG and annotations as `annotations.json`.
- **Objectron Compatible**: Exports 9 keypoints (1 center + 8 corners) in 2D and 3D.

## Coordinate Systems
1. **Local Space**: The unit cube is defined from -0.5 to 0.5 in all axes.
2. **World Space**: Set by ARCore's internal tracking.
3. **Camera Space**: Transformed using the ARCore View Matrix. `keypoints_3d` are stored in this space (meters).
4. **Image Space**: Projected using Camera Intrinsics. `keypoints_2d` are stored as `[x_norm, y_norm, depth]`.

## Annotation Format
The `annotations.json` file contains a list of frame objects:
```json
{
  "frame_id": 0,
  "image": "frame_000.jpg",
  "keypoints_2d": [[x_norm, y_norm, depth], ...],
  "keypoints_3d": [[x_cam, y_cam, z_cam], ...],
  "visibility": [1.0, ...],
  "camera_intrinsics": { "fx": ..., "fy": ..., ... },
  "view_matrix": [...],
  "model_matrix": [...],
  "timestamp": ...
}
```

## Keypoint Order (9 Points)
- 0: Center (0,0,0)
- 1-4: Front face corners
- 5-8: Back face corners

## Math & Pipeline
1. **Model Matrix**: $M = T \cdot R \cdot S$
2. **Keypoint World**: $P_{world} = M \cdot P_{local}$
3. **Keypoint Camera**: $P_{camera} = V \cdot P_{world}$
4. **Keypoint Image**: $x_{pixel} = fx \cdot (x_{cam} / -z_{cam}) + cx$

## Usage
1. Open the app and scan the floor until dots appear.
2. Tap on the floor to place the bounding box.
3. Use the **- / + buttons** to adjust Scale, Rotation, and Translation in all 3 axes (X, Y, Z) to match your beverage can precisely.
4. Tap **"START RECORDING"**.
5. Move the phone slowly around the can (360 degrees, different heights) to capture all angles.
6. The app captures frames at up to 60fps with synchronization between images and matrices.
7. Tap **"STOP RECORDING"** when finished (suggested 12-15 seconds).
8. Tap **"EXPORT ZIP"** to bundle the dataset.
9. Pull the `.zip` from `/Android/data/com.example.arcoreapp/files/Pictures/`.
