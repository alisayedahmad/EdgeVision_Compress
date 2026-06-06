# EdgeVision-Compress

> End-to-End Model Compression Pipeline for Industrial Visual Anomaly Detection on Edge Hardware

[![CI](https://github.com/alisayedahmad/EdgeVIsion_Compress/actions/workflows/ci.yml/badge.svg)](https://github.com/alisayedahmad/EdgeVIsion_Compress/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What this project is

EdgeVision-Compress is a research engineering project that takes a ResNet50 anomaly detection model trained on [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) and compresses it step by step for deployment on edge hardware like the Raspberry Pi 4 and Jetson Nano.

The goal is not just to make the model smaller — it is to understand exactly what each compression technique costs in accuracy, and what it buys in speed and size. Every step is tracked, benchmarked, and compared against the previous one.

The pipeline covers four techniques applied in sequence:

1. **Unstructured pruning** — zeroes out the smallest weights globally (L1 magnitude)
2. **Structured channel pruning** — removes entire filters using the Taylor criterion, giving real FLOPs reduction
3. **Quantization-aware training** — simulates INT8 arithmetic during training so the model adapts before export
4. **Knowledge distillation** — trains a MobileNetV3-Small student to mimic the compressed ResNet50 teacher

## Results

Updated as each module is completed.

| Model | Image AUROC | Latency P50 (ms) | Size (MB) | FLOPs |
|---|---|---|---|---|
| ResNet50 baseline | 0.9902 | — | — | — |
| + Unstructured pruning | — | — | — | — |
| + Structured pruning | — | — | — | — |
| + INT8 QAT | — | — | — | — |
| MobileNetV3-Small (student) | — | — | — | — |

Evaluated on the MVTec AD `bottle` category, test split.

## Installation

```bash
git clone https://github.com/alisayedahmad/EdgeVIsion_Compress
cd EdgeVIsion_Compress

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / Mac
source .venv/bin/activate

# Install PyTorch with CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install the project and dev dependencies
pip install -e ".[dev]"
```

## Dataset

This project uses the [MVTec Anomaly Detection dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) (Bergmann et al., CVPR 2019). It contains 15 categories of industrial objects and textures, with pixel-level ground truth masks for all defect types.

Download and prepare:

```bash
python scripts/download_mvtec.py --output data/mvtec
dvc add data/mvtec
```

The dataset is tracked with DVC and is not stored in Git.

## Training the baseline

```bash
# Full training on the bottle category (default)
python scripts/train_baseline.py

# Different category
python scripts/train_baseline.py data.category=capsule

# Quick smoke test (2 epochs)
python scripts/train_baseline.py training.epochs=2 training.batch_size=16
```

All hyperparameters live in `configs/` — nothing is hardcoded in the source files.

## Tracking experiments

```bash
mlflow ui
```

Opens the MLflow dashboard at `http://localhost:5000`. Every training run logs its parameters, metrics per epoch, and the best checkpoint.

## Running tests

```bash
pytest tests/ -v
```

## Project structure

```
EdgeVision-Compress/
├── configs/
│   ├── model/                  # resnet50.yaml, mobilenet.yaml, ...
│   ├── compression/            # pruning.yaml, quantization.yaml, ...
│   ├── data/                   # mvtec.yaml
│   ├── train.yaml              # main training config
│   └── benchmark.yaml          # benchmark config
├── data/                       # DVC-tracked, not in Git
├── src/
│   ├── models/                 # ResNet50AnomalyDetector, student model
│   ├── compression/
│   │   ├── pruning/            # unstructured + structured pruning
│   │   ├── quantization/       # PTQ and QAT
│   │   └── distillation/       # knowledge distillation
│   ├── data/                   # MVTecDataset, CutPaste augmentation
│   ├── evaluation/             # image AUROC, pixel AUROC
│   ├── training/               # Trainer, EarlyStopping
│   └── utils/                  # logging, seeding
├── scripts/                    # train_baseline.py, download_mvtec.py, ...
├── notebooks/                  # exploration only, no production logic
├── tests/                      # pytest test suite — one file per module
├── demo/                       # Gradio demo (Module 9)
├── .github/workflows/ci.yml    # runs pytest on every push
├── dvc.yaml                    # DVC pipeline stages
└── pyproject.toml              # package definition and dependencies
```

## Design decisions

**Why CutPaste for training?** MVTec provides only normal images in the training split. CutPaste creates synthetic anomalies on-the-fly by cutting a patch from an image and pasting it elsewhere. The model learns to detect spatial inconsistencies, which transfers well to real industrial defects.

**Why AUROC and not accuracy?** The test set is imbalanced (more normal images than defective ones). Accuracy would be misleading. AUROC measures ranking quality over the full operating range, independent of any threshold choice.

**Why Hydra for config?** Every hyperparameter lives in a YAML file. Launching a variant requires no code change: `python train.py training.lr=1e-3`. Hydra also saves the exact config used for every run automatically.

**Why DVC for data?** The MVTec dataset is ~5 GB. Git is not designed for large files. DVC versions the data separately while keeping a lightweight pointer in Git, so the dataset version is always linked to the code version.

## Stack

| Tool | Role |
|---|---|
| PyTorch 2.x | Model training and compression |
| torchvision | ResNet50 pretrained weights, transforms |
| Hydra | Configuration management |
| MLflow | Experiment tracking |
| DVC | Data versioning |
| ONNX | Model export for edge deployment |
| pytest | Test suite |

## Reference

Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2019). MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection. *CVPR 2019*.

Li, C., Sohn, K., Yoon, J., & Pfister, T. (2021). CutPaste: Self-Supervised Learning for Anomaly Detection and Localization. *CVPR 2021*.

## License

MIT
