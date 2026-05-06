# EEG Emotion Recognition Reproduction Guide

## 1. Project Overview

This project provides a reproducible pipeline for EEG-based emotion recognition using the **REFED-2025** dataset.

The pipeline includes:
1. Downloading EEG data
2. Preprocessing raw EEG signals
3. Training deep learning models
4. Comparing multiple baselines
5. Visualizing experiment results

The main model used in this project is **EEGNet**, a lightweight CNN designed specifically for EEG signal classification.

The entire workflow is organized as a sequence of scripts:

```
Download data
      ↓
Preprocess EEG
      ↓
Train models
      ↓
Compare baselines
      ↓
Visualize results
```

---

## 2. Project Structure

Recommended directory structure:

```
project/
│
├── 00_download_data.py
├── 01_preprocess.py
├── 02_train_and_compare.py
├── 03_final_comparison.py
├── 03_final_comparison_multi.py
├── 04_visualize.py
├── requirements.txt
│
├── data/
│   ├── refed/
│   └── processed/
│
├── models/
├── results/
└── figures/
```

Create required folders before running:

```bash
mkdir -p data/processed
mkdir -p models
mkdir -p results
mkdir -p figures
```

---

## 3. Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies include:

- numpy
- scipy
- torch
- scikit-learn
- matplotlib
- pandas
- tqdm
- huggingface_hub

---

## 4. Full Reproduction Pipeline

Run the scripts in the following order:

```
00_download_data.py
        ↓
01_preprocess.py
        ↓
02_train_and_compare.py       (ablation experiments)
        ↓
03_final_comparison.py        (final comparison)
        ↓
03_final_comparison_multi.py  (multi-baseline comparison)
        ↓
04_visualize.py               (visualization)
```

---

## 5. Script Documentation

### 5.1 `00_download_data.py`

**Purpose**

Download the REFED-2025 EEG dataset from HuggingFace. The dataset includes:
- EEG recordings
- Emotion annotations
- Metadata files

**What the script does**

1. Creates dataset folders:
   - `data/refed/data`
   - `data/refed/annotations`
2. Downloads metadata files:
   - `Metadata.csv`
   - `SAM_score.csv`
   - `PANAS_score.csv`
   - `Video_info.csv`
   - `EEG_channels.csv`
3. Downloads EEG recordings for 32 subjects (each subject contains EEG recordings for 15 videos)

**Output**

```
data/refed/
    Metadata.csv
    data/
        1/
        2/
        ...
        32/
    annotations/
        1_label.mat
        2_label.mat
```

**How to run**

```bash
python 00_download_data.py
```

> This step may take several minutes depending on internet speed.

---

### 5.2 `01_preprocess.py`

**Purpose**

Convert raw REFED EEG data into a model-ready dataset. The script transforms the REFED dataset into a SEED-like format used by the training scripts.

**Main tasks**

1. **Load EEG recordings** — Each subject contains `EEG_videos.mat`, with each video containing an EEG signal of shape `(channels, timepoints)`
2. **Load emotion annotations** — Valence and arousal labels as continuous values in range `0–255`
3. **Convert to discrete emotion labels** — Continuous valence values are mapped to 3 classes:
   - `0` → Negative
   - `1` → Neutral
   - `2` → Positive
4. **Signal processing:**
   - Bandpass filtering: `0.5–50 Hz`
   - Downsampling: `1000 Hz → 200 Hz`
   - Segmentation: split into 1-second windows
5. **Normalization** — Z-score normalization per EEG channel

**Output**

Saved to `data/processed/refed_preprocessed.npz`, containing:
- `X_train`
- `y_train`
- `X_test`
- `y_test`
- `subject_ids`

**How to run**

```bash
python 01_preprocess.py
```

---

### 5.3 `02_train_and_compare.py`

**Purpose**

Perform ablation experiments to evaluate the effects of network architecture and data augmentation.

**Compared Models**
- EEGNet
- SimpleCNN

**EEGNet Architecture**

EEGNet is designed for EEG signals using depthwise and separable convolutions:

| Layer | Type |
|---|---|
| 1 | Conv2D (temporal) |
| 2 | BatchNorm |
| 3 | Depthwise convolution |
| 4 | Average pooling |
| 5 | Dropout |
| 6 | Separable convolution |
| 7 | Fully connected classifier |

**Data Augmentation**

Applies time masking — randomly masks segments of the EEG time series to improve model robustness.

**Training Configuration**

| Parameter | Value |
|---|---|
| Batch size | 64 |
| Epochs | 40 |
| Learning rate | 1e-3 |
| Training samples | 3000 |

**Output**

Results saved to `results/ablation_results.txt` with accuracy and F1 score metrics.

**How to run**

```bash
python 02_train_and_compare.py
```

---

### 5.4 `03_final_comparison.py`

**Purpose**

Perform the final comparison experiment between **EEGNet** and **SimpleCNN** to demonstrate EEGNet's performance advantage.

**Training Setup**

| Parameter | Value |
|---|---|
| Training samples | 3000 |
| Epochs | 40 |
| Batch size | 64 |
| Learning rate | 1e-3 |

**Output**

- Trained models saved to:
  - `models/eegnet_final.pth`
  - `models/simplecnn_baseline.pth`
- Results saved to: `results/final_comparison.txt`

**How to run**

```bash
python 03_final_comparison.py
```

---

### 5.5 `03_final_comparison_multi.py`

**Purpose**

Compare EEGNet with multiple baseline methods, extending the comparison beyond CNN models.

**Compared Methods**

| Model | Type |
|---|---|
| EEGNet | Deep learning |
| SimpleCNN | Deep learning |
| SVM | Classical ML |
| RandomForest | Classical ML |

**Training Process**

1. Load preprocessed EEG dataset
2. Train each model separately
3. Evaluate on the test set
4. Compare performance (Accuracy & F1 Score)

**Output**

Results saved to `results/multi_model_comparison_multi.txt`

**How to run**

```bash
python 03_final_comparison_multi.py
```

---

### 5.6 `04_visualize.py`

**Purpose**

Generate visualizations for the experiment results.

**Generated Visualizations**

1. **Accuracy comparison plots** — Performance differences between models
2. **Confusion matrix** — Classification performance across emotion classes
3. **t-SNE feature visualization** — Visualizes learned EEG feature embeddings to evaluate whether emotion classes form separable clusters

**Output**

Figures saved to `figures/`:
- `accuracy_comparison.png`
- `confusion_matrix.png`
- `tsne_visualization.png`

**How to run**

```bash
python 04_visualize.py
```

---

### 5.7 `requirements.txt`

Defines all required Python dependencies:

```
numpy
scipy
torch
scikit-learn
matplotlib
seaborn
pandas
tqdm
huggingface_hub
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 6. Complete Example Workflow

```bash
# Install dependencies
pip install -r requirements.txt

# Create required folders
mkdir -p data/processed models results figures

# Run pipeline in order
python 00_download_data.py
python 01_preprocess.py
python 02_train_and_compare.py
python 03_final_comparison_multi.py
python 04_visualize.py
```

---

## 7. Summary

This project implements a complete EEG emotion recognition pipeline based on the **REFED-2025** dataset.

**Main contributions:**
- Preprocessing pipeline for EEG signals
- EEGNet implementation
- Ablation experiments
- Multi-model comparison
- Visualization tools

The project is designed to be **fully reproducible** using the provided scripts.
