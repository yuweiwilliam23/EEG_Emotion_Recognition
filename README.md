EEG Emotion Recognition Reproduction Guide

1. Project Overview
This project provides a reproducible pipeline for EEG-based emotion recognition using the REFED-2025 dataset.
The pipeline includes:
  1.Downloading EEG data
  2.Preprocessing raw EEG signals
  3.Training deep learning models
  4.Comparing multiple baselines
  5.Visualizing experiment results
The main model used in this project is EEGNet, a lightweight CNN designed specifically for EEG signal classification.
The entire workflow is organized as a sequence of scripts:
Download data
      ↓
Preprocess EEG
      ↓
Train models
      ↓
Compare baselines
      ↓
Visualize results

2. Project Structure
Recommended directory structure:
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
Create required folders before running:
mkdir -p data/processed
mkdir -p models
mkdir -p results
mkdir -p figures

3. Environment Setup
Install dependencies:
pip install -r requirements.txt
Dependencies include:
numpy
scipy
torch
scikit-learn
matplotlib
pandas
tqdm
huggingface_hub

4. Full Reproduction Pipeline
Run the scripts in the following order:
00_download_data.py
        ↓
01_preprocess.py
        ↓
02_train_and_compare.py   (ablation experiments)
        ↓
03_final_comparison.py    (final comparison)
        ↓
03_final_comparison_multi.py (multi-baseline comparison)
        ↓
04_visualize.py           (visualization)

5. Script Documentation
Below is a detailed explanation of every file in the project.

5.1 00_download_data.py
Purpose
Download the REFED-2025 EEG dataset from HuggingFace.
The dataset includes:
EEG recordings
emotion annotations
metadata files

What the script does
1.Creates dataset folders
data/refed/data
data/refed/annotations
2.Downloads metadata files:
Metadata.csv
SAM_score.csv
PANAS_score.csv
Video_info.csv
EEG_channels.csv
3.Downloads EEG recordings for 32 subjects
Each subject contains EEG recordings for 15 videos.

Output
Downloaded files will be stored in:
data/refed/
Example:
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

How to run
python 00_download_data.py
This step may take several minutes depending on internet speed.

5.2 01_preprocess.py
Purpose
Convert raw REFED EEG data into a model-ready dataset.
The script transforms the REFED dataset into a SEED-like format used by the training scripts.

Main tasks
The preprocessing pipeline includes:
1. Load EEG recordings
Each subject contains:
EEG_videos.mat
Each video contains:
EEG signal shape
(channels, timepoints)

2. Load emotion annotations
Emotion labels include:
valence
arousal
These are continuous values in range:
0 – 255

3. Convert to discrete emotion labels
The script converts continuous valence values to 3 emotion classes:
0  negative
1  neutral
2  positive

4. Signal processing
EEG signals undergo:
Bandpass filtering
0.5 – 50 Hz
Downsampling
1000 Hz → 200 Hz
Segmentation
EEG signals are split into 1-second windows.

5. Normalization
Each EEG channel is standardized using Z-score normalization.

Output
The processed dataset is saved as:
data/processed/refed_preprocessed.npz
Contents include:
X_train
y_train
X_test
y_test
subject_ids

How to run
python 01_preprocess.py

5.3 02_train_and_compare.py
Purpose
Perform ablation experiments to evaluate the effects of:
network architecture
data augmentation

Compared Models
1.EEGNet
2.SimpleCNN

EEGNet Architecture
EEGNet is designed for EEG signals using:
depthwise convolution
separable convolution
temporal feature extraction
Main layers:
Conv2D (temporal)
BatchNorm
Depthwise convolution
Average pooling
Dropout
Separable convolution
Fully connected classifier

Data Augmentation
The script applies time masking:
Randomly masks segments of the EEG time series to improve model robustness.

Training Configuration
batch size = 64
epochs = 40
learning rate = 1e-3
training samples = 3000

Output
Results are saved to:
results/ablation_results.txt
Metrics include:
Accuracy
F1 score

How to run
python 02_train_and_compare.py

5.4 03_final_comparison.py
Purpose
Perform the final comparison experiment between:
EEGNet
vs
SimpleCNN

Goal
Demonstrate the performance advantage of EEGNet.

Training Setup
training samples = 3000
epochs = 40
batch size = 64
learning rate = 1e-3

Evaluation Metrics
The script computes:
Accuracy
F1 Score

Model Output
Trained models are saved to:
models/eegnet_final.pth
models/simplecnn_baseline.pth

Result Files
Experiment results saved to:
results/final_comparison.txt

How to run
python 03_final_comparison.py

5.5 03_final_comparison_multi.py
Purpose
Compare EEGNet with multiple baseline methods.
This script extends the comparison beyond CNN models.

Compared Methods
EEGNet
SimpleCNN
SVM
RandomForest

Why this script is important
It demonstrates that EEGNet outperforms both deep learning and classical machine learning baselines.

Training Process
1.Load preprocessed EEG dataset
2.Train each model separately
3.Evaluate on the test set
4.Compare performance

Metrics
Accuracy
F1 Score

Output
Results saved to:
results/multi_model_comparison_multi.txt

How to run
python 03_final_comparison_multi.py

5.6 04_visualize.py
Purpose
Generate visualizations for the experiment results.
This script produces figures used for analysis and reporting.

Generated Visualizations
1. Accuracy comparison plots
Shows performance differences between models.

2. Confusion matrix
Displays classification performance across emotion classes.

3. t-SNE feature visualization
t-SNE is used to visualize the learned EEG feature embeddings.
It helps evaluate whether emotion classes form separable clusters.

Output
Figures are saved to:
figures/
Examples:
accuracy_comparison.png
confusion_matrix.png
tsne_visualization.png

How to run
python 04_visualize.py

5.7 requirements.txt
Purpose
Defines all required Python dependencies for the project.

Libraries
numpy
scipy
torch
scikit-learn
matplotlib
seaborn
pandas
tqdm
huggingface_hub

Install
pip install -r requirements.txt

6. Complete Example Workflow
Example reproduction process:
pip install -r requirements.txt

mkdir -p data/processed models results figures

python 00_download_data.py

python 01_preprocess.py

python 02_train_and_compare.py

python 03_final_comparison_multi.py

python 04_visualize.py

7. Summary
This project implements a complete EEG emotion recognition pipeline based on the REFED-2025 dataset.
Main contributions:
preprocessing pipeline for EEG signals
EEGNet implementation
ablation experiments
multi-model comparison
visualization tools
The project is designed to be fully reproducible using the provided scripts.

