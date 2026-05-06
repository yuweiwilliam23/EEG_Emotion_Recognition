#!/usr/bin/env python3
"""可视化脚本 - 基于03_final_comparison_multi.py的实验结果"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

np.random.seed(42)
os.makedirs('figures', exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})
sns.set_style("whitegrid")

PRED_FILE = 'results/predictions.npz'
CLASS_NAMES = ['Negative', 'Neutral', 'Positive']
MODEL_NAMES = ['EEGNet', 'SimpleCNN', 'MLP', 'SVM']
MODEL_KEYS  = ['eegnet', 'simplecnn', 'mlp', 'svm']
COLORS = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']


def load_predictions():
    data = np.load(PRED_FILE)
    y_test = data['y_test']
    preds = {name: data[key] for name, key in zip(MODEL_NAMES, MODEL_KEYS)}
    return y_test, preds


def plot_accuracy_comparison(y_test, preds):
    """Fig 1: 准确率 & F1 柱状对比图"""
    accs = [accuracy_score(y_test, preds[m]) * 100 for m in MODEL_NAMES]
    f1s  = [f1_score(y_test, preds[m], average='macro') for m in MODEL_NAMES]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Accuracy ---
    bars1 = ax1.bar(MODEL_NAMES, accs, color=COLORS, edgecolor='black', alpha=0.85)
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('Model Accuracy Comparison', fontweight='bold')
    ax1.set_ylim(0, max(accs) + 12)
    ax1.grid(axis='y', alpha=0.3)
    for b in bars1:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                 f'{b.get_height():.2f}%', ha='center', fontweight='bold')

    # --- F1 ---
    bars2 = ax2.bar(MODEL_NAMES, f1s, color=COLORS, edgecolor='black', alpha=0.85)
    ax2.set_ylabel('Macro F1 Score', fontweight='bold')
    ax2.set_title('Model F1 Score Comparison', fontweight='bold')
    ax2.set_ylim(0, max(f1s) + 0.08)
    ax2.grid(axis='y', alpha=0.3)
    for b in bars2:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                 f'{b.get_height():.4f}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/accuracy_comparison.png')
    plt.close()
    print("  [1/4] figures/accuracy_comparison.png")


def plot_confusion_matrices(y_test, preds):
    """Fig 2: 四个模型的归一化混淆矩阵"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, (name, ax) in enumerate(zip(MODEL_NAMES, axes.ravel())):
        cm = confusion_matrix(y_test, preds[name])
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=ax, vmin=0, vmax=100,
                    cbar_kws={'label': '%'})
        ax.set_title(name, fontweight='bold', fontsize=13)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig('figures/confusion_matrices.png')
    plt.close()
    print("  [2/4] figures/confusion_matrices.png")


def plot_per_class_f1(y_test, preds):
    """Fig 3: 每个类别的 F1 分组柱状图"""
    x = np.arange(len(CLASS_NAMES))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, name in enumerate(MODEL_NAMES):
        f1_per = f1_score(y_test, preds[name], average=None)
        ax.bar(x + i * width, f1_per, width,
               label=name, color=COLORS[i], edgecolor='black', alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(CLASS_NAMES, fontsize=12)
    ax.set_ylabel('F1 Score', fontweight='bold')
    ax.set_title('Per-Class F1 Score Comparison', fontweight='bold')
    ax.set_ylim(0, 0.6)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/per_class_f1.png')
    plt.close()
    print("  [3/4] figures/per_class_f1.png")


def save_classification_reports(y_test, preds):
    """生成详细分类报告 txt"""
    os.makedirs('results', exist_ok=True)
    with open('results/classification_reports.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Classification Reports\n")
        f.write("=" * 70 + "\n")
        for name in MODEL_NAMES:
            f.write(f"\n{name}\n" + "-" * 70 + "\n")
            f.write(classification_report(y_test, preds[name],
                                          target_names=CLASS_NAMES))
            f.write("\n")
    print("  [4/4] results/classification_reports.txt")


if __name__ == '__main__':
    print("=" * 50)
    print("Generating figures ...")
    print("=" * 50)

    y_test, preds = load_predictions()

    plot_accuracy_comparison(y_test, preds)
    plot_confusion_matrices(y_test, preds)
    plot_per_class_f1(y_test, preds)
    save_classification_reports(y_test, preds)

    print("=" * 50)
    print("Done. All outputs in figures/ and results/")
    print("=" * 50)
