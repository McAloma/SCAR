# SCAR
**SCAR: A Data Quality Description System**

SCAR is a framework designed to evaluate and visualize data quality across various multi-modal datasets. It supports multiple tasks including classification, alignment, and regression, and provides tools for clustering, embedding analysis, and reporting.

---

## 📁 Project Structure

The project is organized into modular directories that serve different roles in data processing, evaluation, and visualization:

### `experiments/`
Contains dataset-specific experiment scripts for classification and alignment tasks, such as:
- `cifar10_classify.py`, `imagenet1k_classify.py`: For image classification.
- `Flickr_alignment.py`, `AudioCaps_alignment.py`: For multi-modal alignment tasks.
- `regression/`: Subfolder with scripts for regression-based evaluations.
- `results/`: Scripts for aggregating and organizing experiment results.

### `draw/`
Provides visualization tools used in SCAR reports:
- `draw_exp_func.py`, `bubble.py`, `size_bar.py`: Used for drawing quality metrics, bar charts, and bubble plots.
- `draw_scatter.py`: For scatter plot visualization of embedding clusters.

### `data/`
Scripts for preparing and processing raw datasets into usable formats:
- `cifar2image.py`: Converts CIFAR labels into images.
- Dataset-specific subfolders like `Flickr30k/`, `MSR_VTT/`, `AudioCaps/`, `imagenet-1k/` include data loaders and processing utilities.

### `src/`
Core source code of SCAR, structured into:
- `loader/`: Dataset loader implementations for classification and alignment.
- `encoder/`: Multi-modal and single-modal encoders, including CLIP variants, audio-text, vision-text, etc.
- `model/`: Simple model components (e.g., linear classifier).
- `library/`: Includes 3rd-party or wrapped libraries like VideoCLIP and Pengi for evaluation or encoding.
- `scar/`: The core logic of the SCAR score computation and recalibration scripts.

---

## 🔧 Features

- Supports multi-modal data: image, text, audio, and video.
- Compatible with datasets such as CIFAR-10/100, ImageNet, COCO, Flickr30k, AudioCaps, Wikipedia, and more.
- Modular encoder system: plug in your own encoders or use built-in CLIP, VideoCLIP, Pengi, etc.
- Generates interpretable visualizations and cluster statistics.
- Supports SCAR metrics: scale, coverage, authenticity, richness.

---

## 🚀 Getting Started

1. Clone this repository.
2. Prepare your data under the `data/` directory.
3. Select and run an experiment script under `experiments/` or `test_hanlin/`.
4. Visualize results using scripts under `draw/`.

---

## 📦 Dependencies

Make sure to install required dependencies (e.g., PyTorch, NumPy, Matplotlib). Some components may depend on external models such as CLIP or VideoCLIP.

---

## 📁 Notes

- All results and intermediate files are saved to structured folders (e.g., `results/`, `test_hanlin_results/`).
- You can define and evaluate your own datasets by adding new loaders and experiment scripts.

---

## 📄 License

This project is for academic use. License terms to be added here.

---