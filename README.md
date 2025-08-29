# Graph Of Graph (GiG-CTR-GCN) – Skeleton-Based Action Recognition (NTU RGB+D)

Implementation of a **Graph-of-Graph (GiG-CTR-GCN)** model for skeleton-based human action recognition on the **NTU RGB+D 60** dataset.
Pipeline covers: raw skeleton ingestion → denoising → normalization/packing → training/inference → visualization.

> Project uses [uv](https://docs.astral.sh/uv/) for environment management and targets CUDA 12.6 wheels for PyTorch.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset Download](#dataset-download)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Testing](#testing)
- [Model Configuration](#model-configuration)
- [Results](#results)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

CTR-GCN is a graph convolutional network designed for skeleton-based action recognition. It refines the topology of graph convolutions in a channel-wise manner, improving the representation learning for human actions from skeleton data.

**Key Features:**

- Channel-wise topology refinement
- Support for NTU RGB+D 60 and NTU RGB+D 120 datasets
- Cross-subject and cross-view evaluation protocols
- Joint, bone, joint motion, and bone motion modalities
- Model ensemble capabilities

## 📦 Requirements

### System Requirements

- Python 3.12+
- CUDA-capable GPU (CUDA 12.6 support)
- At least 8GB RAM
- 50GB+ free disk space for dataset

### Python Dependencies

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install the required packages:

```bash
uv sync
```

**Key dependencies:**

- PyTorch 2.8.0+ (with CUDA 12.6 support)
- torchvision 0.23.0+
- opencv-python
- scikit-learn
- tensorboard
- matplotlib
- seaborn
- PyYAML
- Pillow

## 📥 Dataset Download

### Dataset Information

- **Source**: Nanyang Technological University
- **Paper**: "NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis"
- **Official Website**: [NTU RGB+D Dataset](https://rose1.ntu.edu.sg/dataset/actionRecognition/)

### Versions

* **NTU RGB+D 60**: 56,880 skeleton sequences (\~5GB)
* **NTU RGB+D 120**: 114,480 skeleton sequences (\~10GB)

Extract `.skeleton` files into:

```
data/nturgbd_raw/nturgb+d_skeletons/nturgb+d_skeletons/*.skeleton
```

---

## 🛠️ Installation

1. Install prerequisites

   * Python 3.12+
   * CUDA 12.6 GPU (recommended)
   * [uv](https://docs.astral.sh/uv/)

2. Clone repo & sync deps

   ```bash
   git clone <repository-url>
   cd <repo-name>
   uv sync
   ```

3. Prepare raw data directory

   ```bash
   mkdir -p data/nturgbd_raw/nturgb+d_skeletons
   ```

4. Extract skeleton zip archives into that folder.

---

## 🔄 Data Preparation

Run in project root:

1. **Extract raw skeleton data**

   ```bash
   uv run python -m data.ntu.get_raw_skes_data
   ```

   → `data/ntu/raw_data/raw_skes_data.pkl`

2. **Denoise & clean**

   ```bash
   uv run python -m data.ntu.get_raw_denoised_data
   ```

   → `data/ntu/denoised_data/raw_denoised_joints.pkl`

3. **Transform & split** (normalize, pad 300 frames)

   ```bash
   uv run python -m data.ntu.seq_transformation
   ```

   → `NTU60_CS.npz`, `NTU60_CV.npz`

---

## ⚙️ Configuration

Configs are stored in [`config/`](config/).
Examples:

* `config/nturgbd-cross-subject/default.yaml` → Cross-Subject setup
* `config/nturgbd-cross-view/default.yaml` → Cross-View setup

Each YAML specifies:

* **Feeder** (data loader arguments)
* **Model** (architecture + graph)
* **Optimizer & training schedule**
* **Device, batch size, epochs**

---

## 🚀 Training & Evaluation

**Train (Cross-Subject):**

```bash
uv run python main.py --config config/nturgbd-cross-subject/default.yaml --phase train
```

**Evaluate (with saved weights):**

```bash
uv run python main.py --config config/nturgbd-cross-subject/default.yaml --phase test --weights <path-to-pt>
```

**TensorBoard:**

```bash
uv run python -m tensorboard --logdir work_dir/ntu60/xsub/gig_ctrgcn/runs
```

---

## 📦 Processed Data Format

Each NPZ (e.g. `NTU60_CS.npz`) contains:

```
x_train : (N_train, 300, 150)
y_train : (N_train, 60)
x_test  : (N_test, 300, 150)
y_test  : (N_test, 60)
```

* 300 = temporal length (padded)
* 150 = 2 persons × 25 joints × 3 coords (x,y,z)
* 60 = one-hot action classes

---

## 👁️ Visualization

Sample random processed sequences & render joint trajectories:

```bash
uv run python -m viz.viz_seq
```

#### 4. Import Errors

```
ModuleNotFoundError: No module named 'torchlight'
```

**Solution**: Ensure all dependencies are installed:

```bash
uv sync
```

### Performance Tips

- Use GPU for training: `--device 0`
- Use multiple GPUs: `--device 0 1 2 3`
- Monitor training with TensorBoard:

```bash
tensorboard --logdir ./work_dir/ntu60/xsub/ctrgcn_joint/runs
```
