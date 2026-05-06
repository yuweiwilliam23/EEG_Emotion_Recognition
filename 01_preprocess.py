#!/usr/bin/env python3
"""
Preprocess REFED-2025 dataset for emotion recognition
Convert from REFED format to SEED-compatible format
"""

import os
import numpy as np
from scipy.io import loadmat
from scipy import signal
from tqdm import tqdm

def load_refed_subject(subject_id):
    """
    Load EEG data and labels for one subject
    Returns: trials (list of arrays), labels (list of valence/arousal)
    """
    # Load EEG videos data
    eeg_file = f'data/refed/data/{subject_id}/EEG_videos.mat'
    label_file = f'data/refed/annotations/{subject_id}_label.mat'

    eeg_data = loadmat(eeg_file)
    label_data = loadmat(label_file)

    trials = []
    labels = []

    # REFED has 15 videos
    for video_idx in range(1, 16):
        video_key = f'video_{video_idx}'

        if video_key in eeg_data:
            # EEG shape: (channels, timepoints)
            eeg_trial = eeg_data[video_key]

            # Get corresponding label (valence, arousal)
            # Shape: (timepoints, 2) - [valence, arousal] in range 0-255
            label_trial = label_data[video_key]

            # Average valence and arousal over time to get single label
            avg_valence = np.mean(label_trial[:, 0])  # Column 0 is valence
            avg_arousal = np.mean(label_trial[:, 1])  # Column 1 is arousal

            # Convert to 3-class emotion (similar to SEED)
            # Scale is 0-255, with 128 as neutral
            # High valence (>160) -> positive (2)
            # Low valence (<96) -> negative (0)
            # Medium valence (96-160) -> neutral (1)
            if avg_valence > 160:  # High valence
                emotion_label = 2  # Positive
            elif avg_valence < 96:  # Low valence
                emotion_label = 0  # Negative
            else:
                emotion_label = 1  # Neutral

            trials.append(eeg_trial)
            labels.append(emotion_label)

    return trials, labels

def preprocess_eeg(eeg_data, target_fs=200, original_fs=1000):
    """
    Preprocess EEG: filter, downsample, segment
    """
    n_channels, n_timepoints = eeg_data.shape

    # 1. Bandpass filter (0.5-50 Hz)
    nyq = original_fs / 2
    low = 0.5 / nyq
    high = 50 / nyq
    b, a = signal.butter(4, [low, high], btype='band')
    eeg_filtered = signal.filtfilt(b, a, eeg_data, axis=1)

    # 2. Downsample from 1000 Hz to 200 Hz
    downsample_factor = original_fs // target_fs
    eeg_downsampled = eeg_filtered[:, ::downsample_factor]

    # 3. Segment into 1-second windows (200 samples)
    window_size = target_fs  # 1 second
    n_windows = eeg_downsampled.shape[1] // window_size

    segments = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        segment = eeg_downsampled[:, start:end]

        # Z-score normalization per channel
        segment = (segment - segment.mean(axis=1, keepdims=True)) / (segment.std(axis=1, keepdims=True) + 1e-8)

        segments.append(segment)

    return segments

def main():
    print("=" * 60)
    print("REFED-2025 Data Preprocessing")
    print("=" * 60)

    all_data = []
    all_labels = []
    all_subjects = []

    print("\nProcessing 32 subjects...")

    for subject_id in tqdm(range(1, 33), desc="Subjects"):
        try:
            # Load subject data
            trials, labels = load_refed_subject(subject_id)

            # Process each trial
            for trial_eeg, trial_label in zip(trials, labels):
                # Preprocess and segment
                segments = preprocess_eeg(trial_eeg)

                # Add all segments with same label
                for segment in segments:
                    all_data.append(segment)
                    all_labels.append(trial_label)
                    all_subjects.append(subject_id)

        except Exception as e:
            print(f"\nWarning: Failed to process subject {subject_id}: {e}")
            continue

    # Convert to numpy arrays
    all_data = np.array(all_data)  # (n_samples, 64, 200)
    all_labels = np.array(all_labels)
    all_subjects = np.array(all_subjects)

    print(f"\n\nData shape: {all_data.shape}")
    print(f"Labels shape: {all_labels.shape}")
    print(f"Unique labels: {np.unique(all_labels)}")
    print(f"Label distribution:")
    for label in np.unique(all_labels):
        count = np.sum(all_labels == label)
        print(f"  Class {label}: {count} samples ({count/len(all_labels)*100:.1f}%)")

    # Split: Leave-One-Subject-Out (use subject 1 as test)
    test_mask = all_subjects == 1
    train_mask = ~test_mask

    X_train = all_data[train_mask]
    y_train = all_labels[train_mask]
    X_test = all_data[test_mask]
    y_test = all_labels[test_mask]

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Save preprocessed data
    os.makedirs('data/processed', exist_ok=True)
    output_file = 'data/processed/refed_preprocessed.npz'

    np.savez(
        output_file,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        subject_ids=all_subjects
    )

    print(f"\n✓ Preprocessed data saved to: {output_file}")
    print("\nNext steps:")
    print("  1. Run: python 02_train_ssl.py")
    print("  2. Run: python 03_finetune.py")

if __name__ == "__main__":
    main()
