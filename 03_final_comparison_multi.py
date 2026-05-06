#!/usr/bin/env python3
"""最终对比实验：EEGNet vs 多个Baseline"""
import os, sys, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

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

class SimpleMLP(nn.Module):
    def __init__(self, nc=64, nt=200, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(nc * nt, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.drop1(torch.relu(self.bn1(self.fc1(x))))
        x = self.drop2(torch.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)

class DS(Dataset):
    def __init__(self, X, y):
        self.X, self.y = torch.FloatTensor(X), torch.LongTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

def train_and_eval_nn(model, X_tr, y_tr, X_te, y_te):
    loader_tr = DataLoader(DS(X_tr, y_tr), BATCH_SIZE, shuffle=True, num_workers=0)
    loader_te = DataLoader(DS(X_te, y_te), BATCH_SIZE, num_workers=0)

    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), LR)

    for ep in range(EPOCHS):
        model.train()
        for X, y in loader_tr:
            X, y = X.to(DEVICE), y.to(DEVICE)
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
    return acc, f1, preds, labels

if __name__ == '__main__':
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    print("="*70)
    print("EEGNet vs Multiple Baselines")
    print("="*70)

    # Load data
    data = np.load('data/processed/refed_preprocessed.npz')
    X_full, y_full = data['X_train'], data['y_train']
    X_test_full, y_test_full = data['X_test'], data['y_test']

    idx = np.random.choice(len(X_full), N_TRAIN, replace=False)
    X_train, y_train = X_full[idx], y_full[idx]
    X_test, y_test = X_test_full, y_test_full

    print(f"Data: Train={X_train.shape} Test={X_test.shape}")
    print(f"Device: {DEVICE} | Epochs: {EPOCHS}")
    print("-"*70)

    results = {}

    # 1. EEGNet
    print("\n[1] EEGNet (Ours)")
    model1 = EEGNet(64, 200, 8, 16, 2, 0.25, 3).to(DEVICE)
    acc1, f1_1, preds1, _ = train_and_eval_nn(model1, X_train, y_train, X_test, y_test)
    print(f"Accuracy: {acc1:.4f} ({acc1*100:.2f}%) | F1: {f1_1:.4f}")
    torch.save(model1.state_dict(), 'models/eegnet_final.pth')
    results['EEGNet'] = {'acc': acc1, 'f1': f1_1, 'preds': preds1}

    # 2. SimpleCNN
    print("\n[2] SimpleCNN")
    model2 = SimpleCNN(64, 200, 3).to(DEVICE)
    acc2, f1_2, preds2, _ = train_and_eval_nn(model2, X_train, y_train, X_test, y_test)
    print(f"Accuracy: {acc2:.4f} ({acc2*100:.2f}%) | F1: {f1_2:.4f}")
    torch.save(model2.state_dict(), 'models/simplecnn_baseline.pth')
    results['SimpleCNN'] = {'acc': acc2, 'f1': f1_2, 'preds': preds2}

    # 3. MLP
    print("\n[3] MLP")
    model3 = SimpleMLP(64, 200, 3).to(DEVICE)
    acc3, f1_3, preds3, _ = train_and_eval_nn(model3, X_train, y_train, X_test, y_test)
    print(f"Accuracy: {acc3:.4f} ({acc3*100:.2f}%) | F1: {f1_3:.4f}")
    results['MLP'] = {'acc': acc3, 'f1': f1_3, 'preds': preds3}

    # 4. SVM (using SGD for speed)
    print("\n[4] SVM")
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    from sklearn.linear_model import SGDClassifier
    svm = SGDClassifier(loss='hinge', alpha=0.0001, max_iter=1000, random_state=SEED, n_jobs=-1)
    print("Training SVM (SGD)...")
    svm.fit(X_train_flat, y_train)
    preds4 = svm.predict(X_test_flat)
    acc4 = accuracy_score(y_test, preds4)
    f1_4 = f1_score(y_test, preds4, average='macro')
    print(f"Accuracy: {acc4:.4f} ({acc4*100:.2f}%) | F1: {f1_4:.4f}")
    results['SVM'] = {'acc': acc4, 'f1': f1_4, 'preds': preds4}

    # Summary
    print("\n" + "="*70)
    print("Results Summary")
    print("="*70)
    print(f"{'Method':<15} {'Accuracy':<15} {'F1 Score':<15}")
    print("-"*70)
    for name, res in results.items():
        print(f"{name:<15} {res['acc']*100:>6.2f}%         {res['f1']:>6.4f}")
    print("="*70)

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/final_comparison_multi.txt', 'w') as f:
        f.write("EEGNet vs Multiple Baselines\n")
        f.write("="*70 + "\n")
        f.write(f"Training samples: {N_TRAIN}\n")
        f.write(f"Test samples: {len(X_test)}\n\n")
        for name, res in results.items():
            f.write(f"{name}: {res['acc']*100:.2f}% (F1: {res['f1']:.4f})\n")

    # Save predictions for visualization
    np.savez('results/predictions.npz',
             y_test=y_test,
             eegnet=preds1,
             simplecnn=preds2,
             mlp=preds3,
             svm=preds4)

    print("\nResults saved to results/")
