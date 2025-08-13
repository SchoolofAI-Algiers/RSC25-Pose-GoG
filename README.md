# GiG-CTR-GCN for Skeleton-Based Action Recognition

This repository contains the implementation of CTR-GCN (Channel-wise Topology Refinement Graph Convolutional Networks) for skeleton-based human action recognition using NTU RGB+D datasets.

> **Note**: This project is optimized for CUDA 12.6 and uses [uv](https://docs.astral.sh/uv/) for dependency management.

## 📋 Table of Contents

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

### Download Links

Choose the appropriate dataset version:

#### NTU RGB+D 60 (60 action classes)

- **Skeleton Files**: [nturgbd_skeletons_s001_to_s017.zip](https://drive.google.com/file/d/1CUZnBtYwifVXS21yVg62T-vrPVayso5H/view)
- **Size**: ~5GB
- **Contains**: 56,880 skeleton sequences

#### NTU RGB+D 120 (120 action classes)

- **Skeleton Files**: [nturgbd_skeletons_s018_to_s032.zip](https://drive.google.com/file/d/1tEbuaEqMxAV7dNc4fqu1O4M7mC6CJ50w/view)
- **Size**: ~10GB
- **Contains**: 114,480 skeleton sequences

### Directory Structure

After downloading, organize your files as follows:

```
GIG-CTR-GCN/
├── data/
│   └── ntu/                     # Working directory for processing
│       ├── statistics/          # Metadata files (already included)
│       ├── raw_data/           # Created during preprocessing
│       ├── denoised_data/      # Created during preprocessing
│       ├── NTU60_CS.npz       # Final processed data (Cross-Subject)
│       └── NTU60_CV.npz       # Final processed data (Cross-View)
├── nturgbd_raw/                # ⚠️ CREATE THIS DIRECTORY
│   └── nturgb+d_skeletons/     # 📁 EXTRACT DOWNLOADED FILES HERE
│       ├── S001C001P001R001A001.skeleton
│       ├── S001C001P001R001A002.skeleton
│       └── ... (thousands of .skeleton files)
└── ... (other project files)
```

## 🛠️ Installation

1. **Prerequisites:**

   - Ensure you have [uv](https://docs.astral.sh/uv/) installed
   - CUDA 12.6 compatible GPU (recommended for performance)

2. **Clone the repository:**

```bash
git clone <repository-url>
cd GIG-CTR-GCN
```

3. **Install dependencies:**

```bash
uv sync
```

4. **Create the required directory structure:**

```bash
mkdir -p nturgbd_raw/nturgb+d_skeletons
```

5. **Extract downloaded dataset:**

```bash
# For NTU60
unzip nturgbd_skeletons_s001_to_s017.zip -d nturgbd_raw/nturgb+d_skeletons/

# For NTU120 (additional)
unzip nturgbd_skeletons_s018_to_s032.zip -d nturgbd_raw/nturgb+d_skeletons/
```

## 🔄 Data Preparation

Process the raw skeleton data through the following steps:

### Step 1: Extract Raw Skeleton Data

```bash
uv run python -m data.ntu.get_raw_skes_data
```

**Purpose**: Reads raw .skeleton files and extracts joint positions and metadata.
**Output**: `raw_data/raw_skes_data.pkl`

### Step 2: Denoise and Clean Data

```bash
uv run python -m data.ntu.get_raw_denoised_data
```

**Purpose**: Removes noise, handles missing frames, and selects main actors.
**Output**: `denoised_data/raw_denoised_joints.pkl`

### Step 3: Transform and Split Dataset

```bash
uv run python -m data.ntu.seq_transformation
```

**Purpose**: Normalizes data and creates train/test splits.
**Output**:

- `NTU60_CS.npz` (Cross-Subject evaluation)
- `NTU60_CV.npz` (Cross-View evaluation)

### Verification

After preprocessing, you should have:

```bash
ls data/ntu/
# Expected output:
# NTU60_CS.npz  NTU60_CV.npz  denoised_data/  raw_data/  statistics/
```

## 🚀 Training

### Basic Training

Train with default configuration:

```bash
# Cross-Subject evaluation
uv run python main.py --config config/nturgbd-cross-subject/default.yaml

# Cross-View evaluation
uv run python main.py --config config/nturgbd-cross-view/default.yaml
```

### Training Different Modalities

```bash
# Joint modality (default)
uv run python main.py --config config/nturgbd-cross-subject/default.yaml

# Bone modality
uv run python main.py --config config/nturgbd-cross-subject/default.yaml --train_feeder_args bone=True --test_feeder_args bone=True --model_saved_name ./work_dir/ntu60/xsub/ctrgcn_bone

# Joint motion modality
uv run python main.py --config config/nturgbd-cross-subject/default.yaml --train_feeder_args vel=True --test_feeder_args vel=True --model_saved_name ./work_dir/ntu60/xsub/ctrgcn_joint_motion

# Bone motion modality
uv run python main.py --config config/nturgbd-cross-subject/default.yaml --train_feeder_args vel=True bone=True --test_feeder_args vel=True bone=True --model_saved_name ./work_dir/ntu60/xsub/ctrgcn_bone_motion
```

### Custom Training Parameters

```bash
uv run python main.py \
    --config config/nturgbd-cross-subject/default.yaml \
    --batch-size 32 \
    --base-lr 0.05 \
    --num-epoch 80 \
    --device 0 1 \
    --work-dir ./work_dir/custom_experiment
```

## 🧪 Testing

### Test a Trained Model

```bash
uv run python main.py \
    --config config/nturgbd-cross-subject/default.yaml \
    --phase test \
    --weights ./work_dir/ntu60/xsub/ctrgcn_joint/runs-XX-XXXXX.pt \
    --test-batch-size 256
```

### Model Ensemble

Combine multiple modalities for better performance:

```bash
uv run python ensemble.py \
    --dataset ntu/xsub \
    --joint-dir ./work_dir/ntu60/xsub/ctrgcn_joint \
    --bone-dir ./work_dir/ntu60/xsub/ctrgcn_bone \
    --joint-motion-dir ./work_dir/ntu60/xsub/ctrgcn_joint_motion \
    --bone-motion-dir ./work_dir/ntu60/xsub/ctrgcn_bone_motion
```

## ⚙️ Model Configuration

### Key Configuration Files

- `config/nturgbd-cross-subject/default.yaml`: Cross-subject evaluation settings
- `config/nturgbd-cross-view/default.yaml`: Cross-view evaluation settings

### Important Parameters

```yaml
# Model architecture
model: model.ctrgcn.Model
model_args:
  num_class: 60 # 60 for NTU60, 120 for NTU120
  num_point: 25 # Number of joints
  num_person: 2 # Maximum number of people
  graph: graph.ntu_rgb_d.Graph

# Training settings
batch_size: 64
base_lr: 0.1
num_epoch: 65
weight_decay: 0.0004

# Data settings
train_feeder_args:
  data_path: data/ntu/NTU60_CS.npz
  window_size: 64
  bone: False # Set to True for bone modality
  vel: False # Set to True for motion modality
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Dataset Not Found Error

```
Error: Skeleton file not found
```

**Solution**: Ensure skeleton files are in `nturgbd_raw/nturgb+d_skeletons/`

#### 2. Memory Issues

```
RuntimeError: CUDA out of memory
```

**Solutions**:

- Reduce batch size: `--batch-size 32`
- Use smaller test batch: `--test-batch-size 128`
- Use CPU: `--device -1`

#### 3. Missing Processed Data

```
FileNotFoundError: data/ntu/NTU60_CS.npz not found
```

**Solution**: Run data preparation steps again:

```bash
uv run python -m data.ntu.get_raw_skes_data
uv run python -m data.ntu.get_raw_denoised_data
uv run python -m data.ntu.seq_transformation
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