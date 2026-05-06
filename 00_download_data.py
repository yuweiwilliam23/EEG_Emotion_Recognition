#!/usr/bin/env python3
"""
Download REFED-2025 real EEG data and annotations from Hugging Face
"""

import os
from huggingface_hub import hf_hub_download
from tqdm import tqdm

def download_refed_data():
    """
    Download REFED-2025 EEG data and annotations
    """
    print("=" * 60)
    print("Downloading REFED-2025 Real EEG Data")
    print("=" * 60)

    # Create directories
    os.makedirs('data/refed/data', exist_ok=True)
    os.makedirs('data/refed/annotations', exist_ok=True)

    repo_id = "REFED2025/REFED-dataset"

    # Download metadata files
    print("\n[1/3] Downloading metadata files...")
    metadata_files = [
        'Metadata.csv',
        'SAM_score.csv',
        'PANAS_score.csv',
        'Video_info.csv',
        'EEG_channels.csv'
    ]

    for file in metadata_files:
        print(f"  Downloading {file}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=file,
            repo_type="dataset",
            local_dir='data/refed'
        )

    # Download EEG data for all 32 subjects
    print("\n[2/3] Downloading EEG data for 32 subjects...")
    print("This may take several minutes...")

    for subject_id in tqdm(range(1, 33), desc="Subjects"):
        subject_dir = f'data/refed/data/{subject_id}'
        os.makedirs(subject_dir, exist_ok=True)

        # Download EEG files (skip fNIRS for now)
        files = [
            f'data/{subject_id}/EEG_baselines.mat',
            f'data/{subject_id}/EEG_videos.mat'
        ]

        for file in files:
            hf_hub_download(
                repo_id=repo_id,
                filename=file,
                repo_type="dataset",
                local_dir='data/refed'
            )

    # Download annotations
    print("\n[3/3] Downloading emotion annotations...")
    for subject_id in tqdm(range(1, 33), desc="Annotations"):
        file = f'annotations/{subject_id}_label.mat'
        hf_hub_download(
            repo_id=repo_id,
            filename=file,
            repo_type="dataset",
            local_dir='data/refed'
        )

    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)
    print(f"\nData saved to: data/refed/")
    print(f"  - 32 subjects")
    print(f"  - EEG sampling rate: 1000 Hz")
    print(f"  - 64 channels")
    print(f"  - 15 emotion videos per subject")
    print(f"  - Real-time dynamic labels (valence & arousal)")

    print("\nNext steps:")
    print("  1. Run: python 01_preprocess_refed.py")
    print("  2. Continue with training pipeline")

if __name__ == "__main__":
    download_refed_data()
