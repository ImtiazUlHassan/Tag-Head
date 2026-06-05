# TAG-Head: Time-Aligned Graph Head for Plug-and-Play Fine-grained Action Recognition

**ICPR 2026** | Imtiaz Ul Hassan, Nik Bessis, Ardhendu Behera  
Department of Computer Science, Edge Hill University

> *Code will be released on GitHub.*  — as stated in the paper abstract.

---

## Overview

TAG-Head is a lightweight spatio-temporal graph head that plugs into any standard 3D CNN backbone (SlowFast, R(2+1)D, I3D, etc.) and improves fine-grained action recognition using **RGB only** — no pose, no text, no optical flow.

The head applies:
1. **Learnable 3D positional encodings (ST-PE)** to backbone tokens
2. **Transformer encoder** for global space-time context
3. **Spatio-temporal graph** with intra-frame fully-connected (IF-FC) and time-aligned temporal (TAT) edges
4. **APPNP propagation** for parameter-free feature refinement
5. **Global mean pooling** + linear classifier

![TAG-Head architecture](paper/Tag-Head/figures/fig2.png)

---

## Results

### Main comparison (Table 1)

| Model | Input | GYM99 Top-1 | GYM99 MCA | GYM288 Top-1 | GYM288 MCA | HAA500 Top-1 |
|---|---|---|---|---|---|---|
| TQN | R | 93.8 | 90.6 | 89.6 | 61.9 | — |
| PGVT | R+P | 96.7 | 91.6 | 91.0 | 63.6 | — |
| PEVL | R+T+P | **97.0** | 91.8 | 91.8 | 64.0 | 84.7 |
| **TAG-Head (Ours)** | **R** | 95.6 | **93.8** | **92.2** | **68.6** | **86.1** |

R → RGB only, T → Text, P → Pose

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ImtiazUlHassan/Tag-Head.git
cd TAG-Head
```

### 2. Create a conda environment

```bash
conda create -n taghead python=3.12
conda activate taghead
```

### 3. Install PyTorch

Install PyTorch matching your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).  
The paper experiments used **Python 3.12**, **PyTorch 2.x**, **CUDA 12.x**.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install PyTorch Geometric

```bash
pip install torch-geometric
pip install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

> Adjust the PyTorch and CUDA version in the URL to match your installation.

### 5. Install remaining dependencies

```bash
pip install nvidia-dali-cuda120   # or cuda118 / cuda121 depending on your setup
pip install pytorchvideo pandas numpy wandb
```

---

## Dataset Preparation

All three datasets use the same CSV format:

```
FileName,Class,TotalFrames,ClassEncoded
video_clip.mp4,ClassName,64,0
...
```

Split CSV files are provided in this repository under `data/`.

### FineGym99

Download from the [FineGym project page](https://sdolivia.github.io/FineGym/).  
We use the **v1.1** temporal annotation split: 26,319 train / 8,520 val across 99 classes.

Organise videos as:
```
gym99/
└── videos/
    ├── {class_id}/
    │   ├── clip1.mp4
    │   └── ...
    └── ...
```

### FineGym288

Download from the [FineGym project page](https://sdolivia.github.io/FineGym/).  
We use the **v1.1** temporal annotation split: 29,333 train / 9,645 val across 288 classes.

```
gym288/
└── videos/
    ├── {class_id}/
    └── ...
```

### HAA500

Download from the [HAA500 project page](https://www.cse.ust.hk/haa/).  
Split: 8,000 train / 500 val / 1,500 test across 500 classes.

```
haa500/
└── videos/
    ├── {ClassName}/
    │   ├── video.mp4
    │   └── ...
    └── ...
```

---

## Pretrained Weights

Download pretrained checkpoints and place them in `weights/`:

| File | Dataset | Top-1 | MCA |
|---|---|---|---|
| `haa500_best.pth` | HAA500 | 86.1% | — |
| `gym99_best.pth` | FineGym99 | 95.6% | 93.8% |
| `gym288_last.pth` | FineGym288 | 92.2% | 68.6% |

> **Download:** https://drive.google.com/drive/folders/18Hv0Y83Yo0GHg_pydtWV08P0Bbvcqhuv?usp=sharing

The R(2+1)D-34 backbone weights (pretrained on IG-65M) are downloaded
automatically via `torch.hub` on first run — internet access is required.

---

## Evaluation (Pretrained Weights)

```bash
# HAA500 — evaluate on test split
python evaluate.py \
    --dataset   haa500 \
    --csv       data/haa500test.csv \
    --video_dir /path/to/haa500/videos/ \
    --weights   weights/haa500_best.pth

# FineGym99 — evaluate on val split
python evaluate.py \
    --dataset   gym99 \
    --csv       data/gym99val_filtered.csv \
    --video_dir /path/to/gym99/videos/ \
    --weights   weights/gym99_best.pth

# FineGym288 — evaluate on val split
python evaluate.py \
    --dataset   gym288 \
    --csv       data/FineGym288_val.csv \
    --video_dir /path/to/gym288/videos/ \
    --weights   weights/gym288_last.pth
```

Expected output format:
```
Dataset : HAA500
Top-1   : 86.1%
MCA     : --
```

---

## Training from Scratch

```bash
# HAA500 (32 frames, 80 epochs)
python train.py \
    --dataset    haa500 \
    --csv_train  data/haa500train.csv \
    --csv_val    data/haa500val.csv \
    --video_dir  /path/to/haa500/videos/ \
    --output_dir runs/haa500

# FineGym99 (64 frames, 60 epochs)
python train.py \
    --dataset    gym99 \
    --csv_train  data/gym99train_filtered.csv \
    --csv_val    data/gym99val_filtered.csv \
    --video_dir  /path/to/gym99/videos/ \
    --output_dir runs/gym99

# FineGym288 (64 frames, 60 epochs)
python train.py \
    --dataset    gym288 \
    --csv_train  data/FineGym288_train.csv \
    --csv_val    data/FineGym288_val.csv \
    --video_dir  /path/to/gym288/videos/ \
    --output_dir runs/gym288
```

### Training defaults (per dataset)

| Dataset | Frames | Epochs | Batch | LR | FFN width |
|---|---|---|---|---|---|
| HAA500 | 32 | 80 | 8 | 1e-5 | 512 |
| FineGym99 | 64 | 60 | 8 | 1e-5 | 1024 |
| FineGym288 | 64 | 60 | 8 | 1e-5 | 2048 |

All defaults match the implementation details in the paper (Section 4).  
Override any value with the corresponding flag, e.g. `--epochs 100 --batch_size 16`.

### Optional WandB logging

```bash
python train.py --dataset haa500 ... --wandb --wandb_project my_project
```

### Hardware

All experiments were run on a single **NVIDIA RTX PRO 6000 Blackwell** GPU.
The backbone is loaded from `torch.hub` (moabitcoin/ig65m-pytorch).

---

## Model Architecture Details

| Component | Configuration |
|---|---|
| Backbone | R(2+1)D-34, pretrained on IG-65M |
| Token dim (d_model) | 512 (fixed by backbone output) |
| Transformer layers | 2 |
| Attention heads | 8 |
| APPNP steps K | 2 |
| APPNP teleport α | 0.1 |
| Graph nodes (32 frames) | 4 × 7 × 7 = 196 |
| Graph nodes (64 frames) | 8 × 7 × 7 = 392 |
| Input resolution | 112 × 112 |
| Optimiser | Adam, lr = 1e-5 |
| LR schedule | Cosine annealing |
| Loss | Cross-entropy |

---

## Repository Structure

```
TAG-Head/
├── model.py        # TAG-Head model (dataset-agnostic)
├── train.py        # Training script (argparse, all datasets)
├── evaluate.py     # Evaluation script (Top-1 + MCA)
├── data_utils.py   # DALI pipeline, sampling, transforms
├── requirements.txt
├── weights/        # Place downloaded .pth files here
└── data/           # Split CSV files for all three datasets
    ├── haa500train.csv
    ├── haa500val.csv
    ├── haa500test.csv
    ├── gym99train_filtered.csv
    ├── gym99val_filtered.csv
    ├── FineGym288_train.csv
    └── FineGym288_val.csv
```

---

## Citation

```bibtex
@inproceedings{hassan2026taghead,
  title     = {TAG-Head: Time-Aligned Graph Head for Plug-and-Play Fine-grained Action Recognition},
  author    = {Hassan, Imtiaz Ul and Bessis, Nik and Behera, Ardhendu},
  booktitle = {Proceedings of the International Conference on Pattern Recognition (ICPR)},
  year      = {2026}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
