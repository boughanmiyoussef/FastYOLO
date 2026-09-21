# FastYolo: Real-Time Object Detection on Edge Devices with YOLOv4-Tiny

## 📌 Project Overview

**FastYolo** is a robust implementation and analytical study of **YOLOv4-Tiny**, engineered specifically for real-time object detection on resource-constrained edge hardware such as the Raspberry Pi 4B. By stripping down the heavy architecture of full YOLOv4, YOLOv4-Tiny trades a minimal margin of accuracy for massive speed gains, making it an optimal solution for low-power environments.

Born out of an academic project analyzing speed-accuracy tradeoffs and edge optimizations, this repository provides a fully runnable, CPU-only workflow. Powered by OpenCV's DNN module, it eliminates the need for complex Darknet builds or dedicated GPUs. It features a custom **edge profile simulator** that replicates hardware constraints (such as thread caps and custom input resolutions) on any development laptop, alongside complete benchmarking and COCO validation evaluation scripts.

---

## 🚀 Key Features

* **Lightweight Architecture:** Features a streamlined CSPDarknet53-Tiny backbone, fewer convolution layers, and an FPN-only neck to maximize performance on modest hardware.
* **Simulated Edge Profiles:** Benchmark performance matching target hardware like the Raspberry Pi 4B directly from your local machine using strict thread limitations and resolution caps.
* **CPU-Only Inference Engine:** Runs smoothly across Linux, macOS, and Windows via OpenCV's headless DNN module without heavy system graphics dependencies.
* **End-to-End Tooling:** Includes complete scripts for multi-source inference (image, video, webcam), rigorous FPS/latency analysis, and mAP precision/recall evaluation on COCO validation subsets.

---

## 🏗 Architecture & Design

The YOLOv4-Tiny architecture reduces parameters down to ~6M (compared to ~63.6M in full YOLOv4) by removing heavy layers like SPP and PANet, relying instead on an FPN-only neck and dual-scale detection heads.

![FastYolo Architecture Diagram](docs/architecture.png)

### Core Architectural Optimizations:
* **Backbone:** CSPDarknet53-Tiny utilizing Cross Stage Partial connections.
* **Activation:** LeakyReLU instead of Mish for lighter compute requirements.
* **Loss Function:** Complete Intersection over Union (CIoU) loss for precise bounding box regression.

---

## 🛠 System Requirements & Setup

### Prerequisites
* **Python:** 3.8+
* **Platforms:** Linux, macOS, Windows

### Installation Guide

1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   # On Linux/macOS:
   source .venv/bin/activate
   # On Windows:
   .venv\Scripts\activate

```

2. **Install core dependencies:**
```bash
pip install -r requirements.txt

```


3. **Download pretrained COCO weights:**
```bash
python models/download_weights.py

```


*(Optional: Pass `--full` to also fetch full YOLOv4 weights for comparative analysis).*

---

## 🏃 Usage Guide

### 1. Object Detection

Run inference on sample images, directories, or video streams, saving outputs to the `results/` directory:

```bash
PYTHONPATH=src python src/detect.py --source data/samples --profile pi4b

```

### 2. Edge-Profile Benchmarking

Analyze FPS and latency profiles across distinct simulated hardware envelopes:

```bash
PYTHONPATH=src python src/benchmark.py --models tiny --profiles unconstrained,pi4b,pi4b_320,edge_2core

```

### 3. Accuracy Evaluation (Optional)

Evaluate mAP, precision, and recall metrics against a subset of the COCO validation set:

```bash
pip install -r requirements-eval.txt
PYTHONPATH=src python src/eval_map.py --model tiny --num-images 200

```

---

## 📊 Performance & Benchmarks

### Cited Target Metrics (Raspberry Pi 4B, 4GB, No TPU)

| Metric | YOLOv4 | YOLOv4-Tiny | Performance Delta |
| --- | --- | --- | --- |
| **Inference Speed** | ~2 FPS | ~14 FPS | **~7× Faster** |
| **mAP @ 0.5** | ~55% | ~45% | Minor Accuracy Trade-off |
| **Parameters** | ~63.6M | ~6M | **~10× Lighter** |

### Local Simulated Benchmark (AMD Ryzen 7 7840HS / CPU-Only)

*Evaluated using `src/benchmark.py` on a single 768×576 image (30 timed iterations):*

| Profile | Threads | Input Size | Mean Latency | p95 Latency | FPS |
| --- | --- | --- | --- | --- | --- |
| **unconstrained** | 16 | 416 | 25.3 ms | 29.2 ms | 39.6 |
| **pi4b** | 4 | 416 | 22.4 ms | 23.5 ms | 44.6 |
| **pi4b_320** | 4 | 320 | 15.4 ms | 17.6 ms | 64.9 |
| **edge_2core** | 2 | 256 | 15.9 ms | 16.7 ms | 62.8 |

> **Key Insight:** Thread scaling hits a sharp saturation point early for lightweight models; reducing input resolution (e.g., from 416 to 320) yields a much more significant performance boost on constrained hardware.

---

## 📁 Repository Structure

```text
FastYolo/
├── Dockerfile                  # Container setup mimicking a Raspberry Pi 4B environment
├── requirements.txt            # Core runtime dependencies
├── requirements-eval.txt       # Evaluation dependencies (pycocotools)
├── configs/
│   └── edge_profiles.yaml      # Named device hardware parameter envelopes
├── models/
│   └── download_weights.py     # Script to fetch configuration, weights, and class labels
├── data/
│   └── samples/                # Directory for test images and media streams
├── docs/
│   └── architecture.png        # System architecture diagram
├── src/
│   ├── detect.py               # Image, video, and webcam inference pipeline
│   ├── benchmark.py            # Latency and FPS profiling utility
│   ├── eval_map.py             # COCO mAP accuracy calculation script
│   ├── edge_sim.py             # Hardware thread and resource throttling engine
│   └── utils.py                # Post-processing, letterboxing, and visualization helpers
└── results/                    # Generated execution artifacts and benchmark logs

```

---

## 📄 License

Distributed under the **MIT License**.

---

## 👤 Author

**Youssef Boughanmi**

*Machine Learning & Software Developer*

* **Email:** [yussefboughanmy@gmail.com](https://www.google.com/search?q=mailto%3Ayussefboughanmy%40gmail.com)
* **Phone:** +216 26 068 101
* **LinkedIn:** [linkedin.com/in/youssef-boughanmi-4990222a0](https://www.google.com/search?q=https://linkedin.com/in/youssef-boughanmi-4990222a0&utm_source=gemini)
* **GitHub:** [github.com/boughanmiyoussef](https://www.google.com/search?q=https://github.com/boughanmiyoussef&utm_source=gemini)
* **Portfolio:** [boughanmiyoussef.github.io](https://www.google.com/search?q=https://boughanmiyoussef.github.io&utm_source=gemini)
