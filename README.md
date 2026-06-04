# EdgeVision-Compress

> End-to-End Model Compression Pipeline for Industrial Visual Anomaly Detection on Edge Hardware

[![CI](https://github.com/alisayedahmad/EdgeVIsion_Compress/actions/workflows/ci.yml/badge.svg)](https://github.com/alisayedahmad/EdgeVIsion_Compress/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

EdgeVision-Compress compresses a ResNet50 anomaly detection model trained on
[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) for
deployment on CPU-constrained edge hardware (Raspberry Pi 4, Jetson Nano).

## Compression Pipeline

| Stage | Technique | Goal |
|-------|-----------|------|
| Baseline | ResNet50 feature extractor | AUROC reference |
| Module 3 | Unstructured L1 pruning | 50% weight sparsity |
| Module 4 | Structured channel pruning (Taylor) | Real FLOPs reduction |
| Module 5 | Quantization-aware training (INT8) | CPU latency reduction |
| Module 6 | Knowledge distillation to MobileNetV3-Small | Edge deployment |

## Results

> To be populated as modules are completed.

| Model | AUROC | Latency P50 (ms) | Size (MB) | FLOPs |
|-------|-------|-----------------|-----------|-------|
| ResNet50 baseline | — | — | — | — |
| + Unstructured pruning | — | — | — | — |
| + Structured pruning | — | — | — | — |
| + INT8 QAT | — | — | — | — |
| MobileNetV3-Small (student) | — | — | — | — |

## Installation

```bash
git clone https://github.com/alisayedahmad/EdgeVIsion_Compress
cd EdgeVIsion_Compress

python -m venv .venv
# Windows:   .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"
```

## Project Structure
EdgeVision-Compress/
├── configs/              # Hydra YAML configuration files
├── data/                 # DVC-tracked dataset (MVTec AD)
├── src/
│   ├── models/           # Model architectures
│   ├── compression/      # Pruning, quantization, distillation
│   ├── benchmark/        # Latency and accuracy benchmarking
│   ├── data/             # Dataset loaders
│   └── utils/            # Logging, seeding, shared utilities
├── notebooks/            # Exploration only — no production logic
├── tests/                # pytest test suite
└── .github/workflows/    # CI pipeline

## Tech Stack

PyTorch 2.x · Hydra · MLflow · DVC · ONNX · pytest

## License

MIT
