# Graph Of Graph (GiG-CTR-GCN) – Skeleton-Based Action Recognition (NTU RGB+D)

Implementation of a **Graph-of-Graph (GiG-CTR-GCN)** model for skeleton-based human action recognition on the **NTU RGB+D 60** dataset.
Pipeline covers: raw skeleton ingestion → denoising → normalization/packing → training/inference → visualization.

> Project uses [uv](https://docs.astral.sh/uv/) for environment management and targets CUDA 12.6 wheels for PyTorch.

---

## 📑 Table of Contents

- [Graph Of Graph (GiG-CTR-GCN) – Skeleton-Based Action Recognition (NTU RGB+D)](#graph-of-graph-gig-ctr-gcn--skeleton-based-action-recognition-ntu-rgbd)
  - [📑 Table of Contents](#-table-of-contents)
  - [📁 Repository Structure](#-repository-structure)
  - [📥 Dataset](#-dataset)
    - [Versions](#versions)
  - [🛠️ Installation](#️-installation)
  - [🔄 Data Preparation](#-data-preparation)
  - [⚙️ Configuration](#️-configuration)
  - [🚀 Training \& Evaluation](#-training--evaluation)
  - [📦 Processed Data Format](#-processed-data-format)
  - [👁️ Visualization](#️-visualization)

---

## 📁 Repository Structure

```
data/
	ntu/                       # Preprocessing scripts & intermediate outputs
		get_raw_skes_data.py     # Read raw .skeleton → raw bodies pickles
		get_raw_denoised_data.py # Denoising, actor selection
		seq_transformation.py    # Normalize, pad (T=300), pack NPZ splits
	nturgbd_raw/               # Raw extracted skeleton files (nested folder structure)

processed_data/ntu60/         # Final NPZ datasets (created after pipeline)
	NTU60_CS.npz
	NTU60_CV.npz

data_feeders/
	bone_pairs.py              # Joint connectivity
	data_utils.py              # Graph/data utilities
	feeder_ntu.py              # PyTorch Dataset (clips, augmentations)

graph/
	graph_utils.py             # Graph construction helpers
	ntu_rgb_d.py               # Defines 25-joint NTU graph

model/
	gig.py                     # Model architecture implementation

viz/
	viz_seq.py                 # Random sequence animation visualization (matplotlib)

config/
	nturgbd-cross-subject/default.yaml
	nturgbd-cross-view/default.yaml

main.py                       # Training / evaluation entrypoint
project_Setup.py              # Centralized path definitions 
```

---

## 📥 Dataset

* **Source**: Nanyang Technological University
* **Paper**: *"NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis"*
* **Website**: [NTU RGB+D Dataset](https://rose1.ntu.edu.sg/dataset/actionRecognition/)

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

