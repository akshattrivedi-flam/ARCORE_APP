# Coca-Cola Can Pose Estimation & Object Detection Dataset

This repository contains the dataset, annotations, trained models, and inference outputs for
pose estimation, object detection, and tracking of Coca-Cola cans.

The project focuses on:
- Red Coca-Cola can (target class)
- Blue Thums Up can (negative class)
- Silver Coca-Cola can (negative class)

The dataset consists of raw videos, extracted frames, bounding-box annotations, and pose data
used for training and evaluation.

---

## Directory Overview

### data/

Contains all dataset-related files.

#### data/raw/
Unmodified source data.

- **videos/**
  Raw videos captured during annotation.
  Organized by class (red, blue, silver).

- **frames/**
  Extracted frames from raw videos.
  Each video has its own subdirectory.

- **annotations/**
  JSON annotation files per video.
  Includes bounding boxes, pose keypoints, and metadata.

#### data/processed/
Cleaned and formatted data used directly by training pipelines.
Includes train/validation/test splits.

#### data/metadata/
Auxiliary dataset information:
- `class_map.json`: Label-to-class mapping
- `dataset_stats.json`: Class distribution and frame counts
- `splits.json`: Train/val/test split definitions

---

### models/

Stores trained models and checkpoints.

- **checkpoints/**
  Intermediate model checkpoints saved during training.

- **final/**
  Final trained models ready for inference or deployment.
  May include ONNX or TensorRT exports.

---

### inference/

Inference inputs and outputs.

- **images/** and **videos/**
  Media used for inference testing.

- **results/**
  Inference outputs including:
  - Visualized detections and pose overlays
  - Metrics and logs

---

### scripts/

Utility scripts for:
- Frame extraction
- Training
- Evaluation
- Inference

---

### configs/

Configuration files controlling training, augmentation, and inference behavior.

---

### experiments/

Isolated experiment runs with separate configs, logs, and outputs.
Useful for ablation studies and benchmarking.

---

## Dataset Summary

| Class  | Videos |
|-------|--------|
| Red   | 26     |
| Blue  | 10     |
| Silver| 10     |

---

## Best Practices Followed

- Raw data is immutable and never modified
- Processed data is reproducible from raw data
- Models and checkpoints are versioned
- Experiments are isolated for reproducibility
- Clear separation between training and inference artifacts

---

## Future Extensions

- Multi-object tracking benchmarks
- 6D pose estimation labels
- Synthetic data augmentation
- Domain randomization support
