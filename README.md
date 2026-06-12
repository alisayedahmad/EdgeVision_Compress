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

Evaluated on the MVTec AD `bottle` category, test split.

### Compression pipeline

| Model | Image AUROC | Latency P50 (ms) | Size (MB) | FLOPs |
|---|---|---|---|---|
| ResNet50 baseline | 0.9902 | 63.1 | 96.3 | 8.18G |
| + Unstructured pruning (85% sparse) | 1.0000 | — | ~96 | 8.18G |
| + Structured pruning (30% channels) | 0.9926 | 56.8 | 77.2 | 6.22G |
| + PTQ INT8 | 0.9369 | 31.4 | 19.7 | — |
| + QAT INT8 | 0.9951 | — | 19.7 | — |
| MobileNetV3-Small (distilled) | 0.9890 | 19.0 | 4.1 | 109.9M |

### Edge deployment (ONNX Runtime, single-thread CPU)

| Model | P50 (ms) | FPS | ONNX Size (MB) | Pi4 estimate |
|---|---|---|---|---|
| ResNet50 baseline | 133.2 | 3.5 | 96.1 | ~800 ms (1.3 FPS) |
| Structured pruned | 97.3 | 4.3 | 77.0 | ~584 ms (1.7 FPS) |
| MobileNetV3 student | 4.2 | 234.8 | 4.0 | ~25 ms (40 FPS) |

**Bottom line:** the distilled student is 23.9x smaller, 31.9x faster, and loses only 0.12% AUROC compared to the baseline. On a Raspberry Pi 4 it runs at an estimated 40 FPS — real-time industrial inspection.

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

```bash
python scripts/download_mvtec.py --output data/mvtec
dvc add data/mvtec
```

The dataset is tracked with DVC and is not stored in Git.

## Usage

```bash
# Train baseline
python scripts/train_baseline.py

# Unstructured pruning
python scripts/prune_unstructured.py "mlflow.tracking_uri=sqlite:///mlflow.db"

# Structured channel pruning
python scripts/prune_structured.py "mlflow.tracking_uri=sqlite:///mlflow.db"

# Quantization (PTQ + QAT)
python scripts/quantize.py "mlflow.tracking_uri=sqlite:///mlflow.db"

# Knowledge distillation
python scripts/distill.py "mlflow.tracking_uri=sqlite:///mlflow.db"

# Full pipeline benchmark
python scripts/benchmark_all.py "mlflow.tracking_uri=sqlite:///mlflow.db"

# Edge deployment simulation (ONNX Runtime)
python scripts/edge_deploy.py "mlflow.tracking_uri=sqlite:///mlflow.db"

# Different category
python scripts/train_baseline.py data.category=capsule
```

All hyperparameters live in `configs/` — nothing is hardcoded in the source files.

## Demo

Launch the interactive Gradio demo locally:

```bash
pip install gradio
python scripts/export_demo.py
python demo/app.py
```

Opens at `http://localhost:7860`. Upload an industrial image to get an anomaly score from the distilled student model.

## Tracking experiments

```bash
mlflow ui
```

Opens the MLflow dashboard at `http://localhost:5000`. Every run logs parameters, metrics per epoch, and the best checkpoint.

## Running tests

```bash
pytest tests/ -v
```

## Project structure

```
EdgeVision-Compress/
├── configs/
│   ├── model/                  # resnet50.yaml
│   ├── compression/            # pruning.yaml
│   ├── data/                   # mvtec.yaml
│   ├── train.yaml              # baseline training
│   ├── prune.yaml              # unstructured pruning
│   ├── prune_structured.yaml   # structured pruning
│   ├── quantize.yaml           # PTQ and QAT
│   ├── distill.yaml            # knowledge distillation
│   └── benchmark.yaml          # benchmark configuration
├── data/                       # DVC-tracked, not in Git
├── src/
│   ├── models/                 # ResNet50AnomalyDetector, MobileNetV3Student
│   ├── compression/
│   │   ├── pruning/            # unstructured (L1), structured (Taylor), FLOPs counter
│   │   ├── quantization/       # PTQ and QAT via FX graph mode
│   │   └── distillation/       # combined loss, distillation trainer
│   ├── benchmark/              # CPU latency, ONNX Runtime benchmark
│   ├── data/                   # MVTecDataset, CutPaste augmentation
│   ├── evaluation/             # image AUROC, pixel AUROC
│   ├── training/               # Trainer, EarlyStopping
│   └── utils/                  # logging, seeding
├── scripts/                    # one script per module
├── demo/                       # Gradio interactive demo
├── tests/                      # pytest — one file per module
├── .github/workflows/ci.yml    # CI on every push
├── dvc.yaml                    # DVC pipeline stages
└── pyproject.toml              # package definition
```

## Design decisions

**Why CutPaste for training?** MVTec provides only normal images in the training split. CutPaste creates synthetic anomalies on-the-fly by cutting a patch and pasting it elsewhere. The model learns to detect spatial inconsistencies, which transfers well to real defects.

**Why AUROC?** The test set is imbalanced. AUROC measures ranking quality over the full operating range, independent of threshold choice.

**Why structured pruning after unstructured?** Unstructured pruning proves the model has massive redundancy (85% zeros, no AUROC loss). Structured pruning exploits that by removing entire filters for real FLOPs and latency reduction.

**Why FX graph mode for quantization?** Our model uses custom `PrunedBottleneck` blocks with residual connections. The Eager Mode API requires manual `QuantStub`/`DeQuantStub` insertion and cannot handle `+=` in residuals. FX mode traces the full computation graph and inserts quantization nodes automatically.

**Why distillation last?** The compressed teacher is still too large for edge deployment. Distillation transfers its knowledge into MobileNetV3-Small (1M params, 4.1 MB), which runs at 4.2 ms on desktop CPU and ~25 ms estimated on Raspberry Pi 4.

**Why ONNX Runtime for edge benchmarks?** PyTorch latency measurements don't reflect real deployment. ONNX Runtime with single-thread execution and the CPUExecutionProvider simulates the constraints of edge hardware. The student achieves 234.8 FPS on desktop and an estimated 40 FPS on Raspberry Pi 4.

## Extending to other categories

The pipeline is demonstrated on the `bottle` category. Extending to all 15 MVTec categories requires rerunning the same pipeline with `data.category=<name>` — no code changes needed.

```bash
# Example: train and compress on the capsule category
python scripts/train_baseline.py data.category=capsule
python scripts/prune_unstructured.py data.category=capsule "mlflow.tracking_uri=sqlite:///mlflow.db"
python scripts/prune_structured.py data.category=capsule "mlflow.tracking_uri=sqlite:///mlflow.db"
python scripts/distill.py data.category=capsule "mlflow.tracking_uri=sqlite:///mlflow.db"
```

## Roadmap

- [ ] Multi-category training with automated pipeline script
- [ ] TensorRT INT8 export for Jetson Nano
- [ ] Real Raspberry Pi 4 latency validation
- [ ] HuggingFace Spaces deployment
- [ ] Pixel-level anomaly heatmap in the Gradio demo

## Stack

| Tool | Role |
|---|---|
| PyTorch 2.x | Training and compression |
| torchvision | Pretrained weights, transforms |
| Hydra | Configuration management |
| MLflow | Experiment tracking |
| DVC | Data versioning |
| ONNX / ONNX Runtime | Model export and edge inference |
| Gradio | Interactive demo |
| pytest | Test suite |

## References

Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2019). MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection. *CVPR 2019*.

Li, C., Sohn, K., Yoon, J., & Pfister, T. (2021). CutPaste: Self-Supervised Learning for Anomaly Detection and Localization. *CVPR 2021*.

Molchanov, P., Tyree, S., Karras, T., Aila, T., & Kautz, J. (2017). Pruning Convolutional Neural Networks for Resource Efficient Inference. *ICLR 2017*.

Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. *NIPS Workshop*.

## License

MIT