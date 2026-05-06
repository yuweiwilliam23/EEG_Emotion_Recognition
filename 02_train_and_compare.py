#!/usr/bin/env python3
"""消融实验：测试架构和数据增强的独立作用"""
import os, sys, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

torch.backends.cudnn.enabled = False
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

SEED = 42
BATCH_SIZE, EPOCHS, LR = 64, 40, 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_TRAIN = 3000

class EEGNet(nn.Module):
    def __init__(self, nc=64, nt=200, f1=8, f2=16, d=2, drop=0.25, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, f1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        self.depthwise = nn.Conv2d(f1, f1*d, (nc, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(f1*d)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(drop)
        self.separable = nn.Conv2d(f1*d, f2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(drop)
        self.fc = nn.Linear(f2 * (nt//32), num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.bn1(self.conv1(x))
        x = self.drop1(self.pool1(torch.relu(self.bn2(self.depthwise(x)))))
        x = self.drop2(self.pool2(torch.relu(self.bn3(self.separable(x)))))
        return self.fc(x.view(x.size(0), -1))

class SimpleCNN(nn.Module):
    def __init__(self, nc=64, nt=200, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, (nc, 25), padding=(0, 12))
        self.bn1 = nn.BatchNorm2d(4)
        self.pool1 = nn.MaxPool2d((1, 4))
        self.drop1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(4, 8, (1, 15), padding=(0, 7))
        self.bn2 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d((1, 4))
        self.drop2 = nn.Dropout(0.25)
        self.fc = nn.Linear(8 * (nt//16), num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.drop1(self.pool1(torch.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(torch.relu(self.bn2(self.conv2(x)))))
        return self.fc(x.view(x.size(0), -1))

def time_mask_gpu(x, ratio_range=(0.1, 0.3)):
    batch_size, n_channels, n_time = x.shape
    x_aug = x.clone()
    for i in range(batch_size):
        ratio = torch.rand(1).item() * (ratio_range[1] - ratio_range[0]) + ratio_range[0]
        mask_len = int(n_time * ratio)
        start = torch.randint(0, n_time - mask_len + 1, (1,)).item()
        x_aug[i, :, start:start+mask_len] = 0
    return x_aug

class DS(Dataset):
    def __init__(self, X, y):
        self.X, self.y = torch.FloatTensor(X), torch.LongTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

def train_and_eval(model, X_tr, y_tr, X_te, y_te, use_aug=False):
    loader_tr = DataLoader(DS(X_tr, y_tr), BATCH_SIZE, shuffle=True, num_workers=0)
    loader_te = DataLoader(DS(X_te, y_te), BATCH_SIZE, num_workers=0)

    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), LR)

    for ep in range(EPOCHS):
        model.train()
        for X, y in loader_tr:
            X, y = X.to(DEVICE), y.to(DEVICE)
            if use_aug:
                X = time_mask_gpu(X)
            opt.zero_grad()
            crit(model(X), y).backward()
            opt.step()

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X, y in loader_te:
            X, y = X.to(DEVICE), y.to(DEVICE)
            preds.extend(torch.argmax(model(X), dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return acc, f1

if __name__ == '__main__':
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    print("="*70)
    print("消融实验：架构 vs 数据增强")
    print("="*70)

    # Load data
    data = np.load('data/processed/refed_preprocessed.npz')
    X_full, y_full = data['X_train'], data['y_train']
    X_test_full, y_test_full = data['X_test'], data['y_test']

    idx = np.random.choice(len(X_full), N_TRAIN, replace=False)
    X_train, y_train = X_full[idx], y_full[idx]
    X_test, y_test = X_test_full, y_test_full

    print(f"数据: Train={X_train.shape} Test={X_test.shape}")
    print(f"设备: {DEVICE} | Epochs: {EPOCHS}")
    print("-"*70)

    results = []

    # 1. EEGNet without augmentation
    print("\n[1] EEGNet (无数据增强)")
    model1 = EEGNet(64, 200, 8, 16, 2, 0.25, 3).to(DEVICE)
    acc1, f1_1 = train_and_eval(model1, X_train, y_train, X_test, y_test, use_aug=False)
    print(f"准确率: {acc1:.4f} ({acc1*100:.2f}%) | F1: {f1_1:.4f}")
    results.append(('EEGNet', '无', acc1, f1_1))

    # 2. EEGNet with augmentation
    print("\n[2] EEGNet (有数据增强)")
    model2 = EEGNet(64, 200, 8, 16, 2, 0.25, 3).to(DEVICE)
    acc2, f1_2 = train_and_eval(model2, X_train, y_train, X_test, y_test, use_aug=True)
    print(f"准确率: {acc2:.4f} ({acc2*100:.2f}%) | F1: {f1_2:.4f}")
    results.append(('EEGNet', '有', acc2, f1_2))

    # 3. SimpleCNN without augmentation
    print("\n[3] SimpleCNN (无数据增强)")
    model3 = SimpleCNN(64, 200, 3).to(DEVICE)
    acc3, f1_3 = train_and_eval(model3, X_train, y_train, X_test, y_test, use_aug=False)
    print(f"准确率: {acc3:.4f} ({acc3*100:.2f}%) | F1: {f1_3:.4f}")
    results.append(('SimpleCNN', '无', acc3, f1_3))

    # 4. SimpleCNN with augmentation
    print("\n[4] SimpleCNN (有数据增强)")
    model4 = SimpleCNN(64, 200, 3).to(DEVICE)
    acc4, f1_4 = train_and_eval(model4, X_train, y_train, X_test, y_test, use_aug=True)
    print(f"准确率: {acc4:.4f} ({acc4*100:.2f}%) | F1: {f1_4:.4f}")
    results.append(('SimpleCNN', '有', acc4, f1_4))

    # Summary
    print("\n" + "="*70)
    print("消融实验总结")
    print("="*70)
    print(f"{'模型':<15} {'数据增强':<10} {'准确率':<15} {'F1 Score':<10}")
    print("-"*70)
    for model_name, aug, acc, f1 in results:
        print(f"{model_name:<15} {aug:<10} {acc*100:>6.2f}%         {f1:>6.4f}")
    print("="*70)

    # Analysis
    print("\n分析：")
    print(f"1. 架构影响（无增强）: EEGNet {acc1*100:.2f}% vs SimpleCNN {acc3*100:.2f}% = {(acc1-acc3)*100:+.2f}%")
    print(f"2. 架构影响（有增强）: EEGNet {acc2*100:.2f}% vs SimpleCNN {acc4*100:.2f}% = {(acc2-acc4)*100:+.2f}%")
    print(f"3. 增强影响（EEGNet）: 有增强 {acc2*100:.2f}% vs 无增强 {acc1*100:.2f}% = {(acc2-acc1)*100:+.2f}%")
    print(f"4. 增强影响（SimpleCNN）: 有增强 {acc4*100:.2f}% vs 无增强 {acc3*100:.2f}% = {(acc4-acc3)*100:+.2f}%")

    # Best result
    best_idx = np.argmax([r[2] for r in results])
    best_model, best_aug, best_acc, best_f1 = results[best_idx]
    print(f"\n最佳配置: {best_model} ({best_aug}数据增强) - {best_acc*100:.2f}%")
    print("="*70)

    # Save
    os.makedirs('results', exist_ok=True)
    with open('results/ablation_study.txt', 'w') as f:
        f.write("消融实验结果\n")
        f.write("="*70 + "\n")
        for model_name, aug, acc, f1 in results:
            f.write(f"{model_name} ({aug}增强): {acc*100:.2f}% (F1: {f1:.4f})\n")
        f.write(f"\n最佳: {best_model} ({best_aug}增强) - {best_acc*100:.2f}%\n")
    print("\n结果已保存到 results/ablation_study.txt")
