# GeoVision Pakistan 🛰️

**Deep Learning for Satellite Imagery Analysis of Pakistan**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-red)](https://pytorch.org)
[![Ultralytics](https://img.shields.io/badge/YOLOv8-8.4.72-green)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

GeoVision Pakistan applies state-of-the-art deep learning to multispectral Sentinel-2 satellite imagery for geospatial analysis of Pakistan. The project covers two tasks:

| Task | Problem | Architecture | Result |
|------|---------|-------------|--------|
| **Task A** | Land Use / Land Cover (LULC) Segmentation | Swin-UNet | **mIoU = 0.456** |
| **Task B** | Graveyard Detection | YOLOv8s | **mAP50 = 0.995** |

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Team](#team)
- [References](#references)

---

## Overview

All satellite data was sourced from ESA's **Sentinel-2** constellation via **Google Earth Engine (GEE)**:

- **Task A**: 6-band multispectral imagery at 100 m/px, covering all of Pakistan. Ground truth from ESA WorldCover v200 (7 classes).
- **Task B**: 3-band RGB imagery at 10 m/px for Lahore, Islamabad, Peshawar, and Faisalabad. Ground truth from OpenStreetMap (OSM) cemetery polygons.

The entire pipeline — from raw GEE export to trained model evaluation — was built from scratch by the team.

---

## Dataset

### Task A — LULC Dataset

| Property | Value |
|----------|-------|
| Coverage | All of Pakistan (60.87°E–77.84°E, 23.63°N–37.09°N) |
| Source | Sentinel-2 SR (2024 median composite) + ESA WorldCover v200 |
| Bands | B2, B3, B4, B8, B11, B12 (6 bands) |
| Resolution | 100 m/px |
| Tile size | 224 × 224 px |
| Total tiles (raw) | 19,912 |
| Final curated set | 2,500 (float32, NaN-free) |
| Split | 70% train / 10% val / 20% test |
| Classes | Built-up, Cropland, Forest, Water, Bare, Desert, Wetland |

**Cleaning pipeline:**
1. Orphan removal — 1 image without a mask removed (`tile_011305.npy`)
2. NaN audit — 10 tiles (0.4%) with nodata pixels excluded
3. dtype cast — float64 → float32 (50% storage reduction)
4. Reproducible split saved to `splits.json` (seed 42)

### Task B — Graveyard Detection Dataset

| Property | Value |
|----------|-------|
| Cities | Lahore, Islamabad, Peshawar, Faisalabad |
| Source | Sentinel-2 RGB (10 m/px, 2022 scenes) + OpenStreetMap |
| Tile size | 256 × 256 px (no overlap) |
| Total tiles | 3,356 |
| Split | Train 2,349 / Val 335 / Test 672 |
| Class | `graveyard` (1 class) |
| Format | YOLO (images + labels + dataset.yaml) |

---

## Model Architectures

### Task A — Swin-UNet

```
Input (6-ch) → 1×1 Conv Projection (6→3) → Swin-Tiny Encoder → Skip Connections → Convolutional Decoder → 7-class output
```

- **Encoder**: `swin_tiny_patch4_window7_224` (pretrained ImageNet via `timm`)
- **Decoder**: 4 upsampling stages with bilinear upsampling + Conv-BN-ReLU blocks
- **Loss**: CrossEntropyLoss
- **Optimiser**: AdamW (lr=1e-4, wd=1e-4)
- **Epochs**: 30 | **Best checkpoint**: Epoch 28

### Task B — YOLOv8s

- **Base model**: `yolov8s.pt` (ImageNet pretrained, 11.1M params)
- **Detection head**: Reconfigured from 80 → 1 class
- **Optimiser**: AdamW (lr=0.002, momentum=0.9)
- **Training**: AMP, mosaic augmentation (first 40 epochs), early stopping (patience=10)
- **Epochs**: 50 planned → 38 actual (stopped) | **Best checkpoint**: Epoch 28

---

## Results

### Task A

| Class | IoU |
|-------|-----|
| Bare | 0.785 |
| Forest | 0.630 |
| Cropland | 0.568 |
| Desert | 0.502 |
| Water | 0.409 |
| Built-up | 0.300 |
| Wetland | 0.000 (absent from test split) |
| **Mean IoU** | **0.456** |

### Task B

| Metric | Val | Test |
|--------|-----|------|
| mAP50 | 0.995 | 0.995 |
| mAP50-95 | 0.995 | 0.995 |
| Precision | 1.000 | 1.000 |
| Recall | 1.000 | 1.000 |
| F1 | 1.000 | 1.000 |

> **Note on confidence scores**: Raw `model.predict()` confidence values average ~0.021 across all test images. This is a known YOLOv8 calibration artefact for single-class high-background-ratio datasets. Lower the inference threshold to `conf=0.05` for operational deployment.

---

## Project Structure

```
GeoVision-Pakistan/
├── notebooks/
│   ├── Geovision_project_Datasets_cleanings.ipynb   # Data cleaning pipeline
│   ├── Prepare_dataset_taskA.ipynb                  # Task A dataset prep
│   ├── Prepare_dataset_taskB.ipynb                  # Task B dataset prep
│   ├── Task_A_LULC_training.ipynb                   # Swin-UNet training
│   └── Task_B_YOLOv8_Graveyard_Detection.ipynb      # YOLOv8s training
│
├── README.md
└── report/
    └── GeoVision_Pakistan_CVPR_Report.docx
```

**Google Drive folder structure** (GeoVision exports (1)/):
```
tiles/
├── images/          # 11,305 × .npy (6, 224, 224) float32
├── masks/           # 11,305 × .npy (224, 224) int labels
└── splits.json      # Train/val/test file paths (seed 42)

graveyards_yolo/
├── images/
│   ├── train/  (2,349 JPEGs)
│   ├── val/    (335 JPEGs)
│   └── test/   (672 JPEGs)
├── labels/
│   ├── train/  (2,349 .txt)
│   ├── val/    (335 .txt)
│   └── test/   (672 .txt)
└── dataset.yaml

checkpoints/
├── best_model.pth              # Swin-UNet (epoch 28, val mIoU=0.419)
└── yolo_runs/graveyard_detection/weights/best.pt   # YOLOv8s (epoch 28)
```

---

## Setup & Usage

### Requirements

```bash
pip install torch torchvision timm ultralytics rasterio scikit-image numpy matplotlib
```

Or open any notebook directly in **Google Colab** (recommended — all notebooks are Colab-ready with Drive mount).

### Quick Start

**Task A — LULC Segmentation:**
1. Open `notebooks/Prepare_dataset_taskA.ipynb` in Colab
2. Mount Drive and point `task_a_base` to your tiles folder
3. Run `notebooks/Task_A_LULC_training.ipynb` to train or load `best_model.pth` for inference

**Task B — Graveyard Detection:**
1. Open `notebooks/Prepare_dataset_taskB.ipynb` in Colab
2. Verify `dataset.yaml` paths point to your `graveyards_yolo/` folder
3. Run `notebooks/Task_B_YOLOv8_Graveyard_Detection.ipynb` or load `best.pt` for inference:

```python
from ultralytics import YOLO
model = YOLO("path/to/best.pt")
results = model.predict("your_image.jpg", conf=0.05)  # Use conf=0.05 for this model
```

---

## Technical Environment

| Component | Details |
|-----------|---------|
| Platform | Google Colab (free tier) |
| GPU | NVIDIA Tesla T4 (14 GB VRAM) |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| Ultralytics | 8.4.72 |
| timm | Latest (Swin-Tiny pretrained) |
| rasterio | GeoTIFF I/O |
| Data Source | Google Earth Engine (project: geovisionpakistan) |

---

## Team

| Name | Student ID | Role |
|------|-----------|------|
| **Saad Ali** (Leader) | MSDS25066 | GEE pipeline, Swin-UNet architecture, Task A training, visualisations, report |
| **Numan Hassan** | MSDS25067 | Task B OSM annotation pipeline, YOLO dataset construction, dataset.yaml fix |
| **Afzal Kareem** | MSDS25062 | YOLOv8s training & evaluation, diagnostic visualisations, confidence analysis |
| **Usama Umer** | MSDS25044 | Task A tiling pipeline, validity filters, NaN audit, Task B tiling, report QA |

---

## References

1. Demir et al., "DeepGlobe 2018: A Challenge to Parse the Earth through Satellite Images," CVPR Workshops, 2018.
2. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," ICLR, 2021.
3. Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows," ICCV, 2021.
4. Cao et al., "Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation," ECCV Workshops, 2022.
5. ESA WorldCover Team, "ESA WorldCover 10m 2021 v200," European Space Agency, 2021.
6. Xia et al., "DOTA: A Large-Scale Dataset for Object Detection in Aerial Images," CVPR, 2018.
7. Jocher et al., "Ultralytics YOLOv8," https://github.com/ultralytics/ultralytics, 2023.
8. Ji et al., "Fully Convolutional Networks for Remote Sensing Image Semantic Segmentation," TGRS, 2019.

---

*MS Data Science Program — GeoVision Pakistan Project*
