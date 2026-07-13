"""
Question 4: Neural Networks [25 points]
========================================
This module handles:
  a. Two NN architectures:
       - Simple feedforward (1-2 hidden layers)
       - Deeper network (3+ hidden layers + dropout)
  b. Compare architectures: training curves, metrics, confusion matrices

Uses PyTorch with Adam optimizer.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.preprocessing import label_binarize
import seaborn as sns

# ── Device configuration ───────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Architecture 1: Simple Feedforward (2 hidden layers) ──────────────────────
class SimpleFFN(nn.Module):
    """
    Simple feedforward neural network: Input → 128 → 64 → Output
    Activation: ReLU, no dropout.
    """
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ── Architecture 2: Deeper Network (4 hidden layers + Dropout) ────────────────
class DeepFFN(nn.Module):
    """
    Deeper feedforward network: Input → 256 → 128 → 64 → 32 → Output
    Activation: ReLU, BatchNorm, Dropout for regularization.
    """
    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def _make_dataloaders(X_train, y_train, X_test, y_test, batch_size: int = 16):
    """Convert numpy arrays to PyTorch DataLoaders."""
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_te = torch.tensor(X_test,  dtype=torch.float32)
    y_te = torch.tensor(y_test,  dtype=torch.long)

    train_ds = TensorDataset(X_tr, y_tr)
    test_ds  = TensorDataset(X_te, y_te)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, X_te, y_te


def _train_model(model, train_loader, num_epochs: int, lr: float, arch_name: str):
    """Train model with Adam optimizer and CrossEntropyLoss. Return loss history."""
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            out  = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        scheduler.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    [{arch_name}] Epoch {epoch+1:>3}/{num_epochs}  Loss: {epoch_loss:.4f}")

    return train_losses


def _evaluate_nn(model, X_te_tensor, y_te_tensor, class_names, label: str, output_dir: str) -> dict:
    """Evaluate a trained model on the test set."""
    model.eval()
    with torch.no_grad():
        logits = model(X_te_tensor.to(DEVICE))
        preds  = torch.argmax(logits, dim=1).cpu().numpy()
    y_true = y_te_tensor.numpy()

    acc   = accuracy_score(y_true, preds)
    f1    = f1_score(y_true, preds, average='weighted')
    prec  = precision_score(y_true, preds, average='weighted', zero_division=0)
    rec   = recall_score(y_true, preds, average='weighted', zero_division=0)
    cm    = confusion_matrix(y_true, preds)

    print(f"\n  {'─'*55}")
    print(f"  {label}")
    print(f"  {'─'*55}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1 (wtd)  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, preds, target_names=class_names))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {label}", fontsize=11, fontweight='bold')
    plt.tight_layout()
    safe_label = label.replace(' ', '_').replace('(', '').replace(')', '')
    fig_path = f"{output_dir}/q4_cm_{safe_label}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved: {fig_path}")

    return {'label': label, 'accuracy': acc, 'f1': f1,
            'precision': prec, 'recall': rec, 'cm': cm}


def run_neural_networks(X_train_pca, X_test_pca, y_train, y_test,
                        class_names, output_dir: str = '.') -> dict:
    """
    Execute Neural Network sub-parts 4a-4b.

    Returns dict with results for Simple and Deep networks.
    """
    results = {}
    input_dim   = X_train_pca.shape[1]
    num_classes = len(class_names)

    print("\n" + "="*70)
    print(f"PART 4: Neural Networks  (device: {DEVICE})")
    print("="*70)
    print(f"  Input dimensions (PCA components): {input_dim}")
    print(f"  Number of classes                : {num_classes}")

    # Hyperparameters
    NUM_EPOCHS  = 100
    BATCH_SIZE  = 16
    LR_SIMPLE   = 1e-3
    LR_DEEP     = 5e-4
    DROPOUT     = 0.3

    train_loader, test_loader, X_te_t, y_te_t = _make_dataloaders(
        X_train_pca, y_train, X_test_pca, y_test, batch_size=BATCH_SIZE
    )

    # ── 4a: Architecture 1 — Simple FFN ───────────────────────────────────────
    print(f"\n--- 4a-i: Simple Feedforward Network (2 hidden layers) ---")
    print(f"  Architecture: {input_dim} → 128 → 64 → {num_classes}")
    print(f"  Activation: ReLU | Optimizer: Adam (lr={LR_SIMPLE}) | Epochs: {NUM_EPOCHS}")

    simple_model = SimpleFFN(input_dim, num_classes)
    simple_losses = _train_model(simple_model, train_loader, NUM_EPOCHS, LR_SIMPLE, 'SimpleFFN')
    r_simple = _evaluate_nn(simple_model, X_te_t, y_te_t, class_names, 'Simple FFN', output_dir)
    results['simple'] = r_simple
    results['simple_model'] = simple_model

    # ── 4a: Architecture 2 — Deep FFN ─────────────────────────────────────────
    print(f"\n--- 4a-ii: Deep Feedforward Network (4 hidden layers + Dropout) ---")
    print(f"  Architecture: {input_dim} → 256 → 128 → 64 → 32 → {num_classes}")
    print(f"  Regularization: BatchNorm + Dropout({DROPOUT}) | Optimizer: Adam (lr={LR_DEEP}) | Epochs: {NUM_EPOCHS}")

    deep_model = DeepFFN(input_dim, num_classes, dropout_rate=DROPOUT)
    deep_losses = _train_model(deep_model, train_loader, NUM_EPOCHS, LR_DEEP, 'DeepFFN')
    r_deep = _evaluate_nn(deep_model, X_te_t, y_te_t, class_names, 'Deep FFN', output_dir)
    results['deep'] = r_deep
    results['deep_model'] = deep_model

    # ── 4b: Training loss curves comparison ────────────────────────────────────
    print("\n--- 4b: Architecture Comparison ---")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Neural Network Training Comparison", fontsize=13, fontweight='bold')

    # Loss curves
    axes[0].plot(simple_losses, color='steelblue', label='Simple FFN', linewidth=2)
    axes[0].plot(deep_losses,   color='darkorange', label='Deep FFN', linewidth=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training Loss (CrossEntropy)")
    axes[0].set_title("Training Loss Curves")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Metric comparison bar chart
    metric_names = ['Accuracy', 'F1 (wtd)', 'Precision', 'Recall']
    simple_vals  = [r_simple['accuracy'], r_simple['f1'], r_simple['precision'], r_simple['recall']]
    deep_vals    = [r_deep['accuracy'],   r_deep['f1'],   r_deep['precision'],   r_deep['recall']]
    x = np.arange(len(metric_names)); w = 0.35
    axes[1].bar(x - w/2, simple_vals, w, label='Simple FFN', color='steelblue', alpha=0.85)
    axes[1].bar(x + w/2, deep_vals,   w, label='Deep FFN',   color='darkorange', alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels(metric_names)
    axes[1].set_ylim([0, 1.1]); axes[1].set_ylabel("Score")
    axes[1].set_title("Test Set Metrics")
    axes[1].legend()
    for i, (s, d) in enumerate(zip(simple_vals, deep_vals)):
        axes[1].text(i - w/2, s + 0.02, f'{s:.3f}', ha='center', fontsize=8)
        axes[1].text(i + w/2, d + 0.02, f'{d:.3f}', ha='center', fontsize=8)

    plt.tight_layout()
    fig_path = f"{output_dir}/q4_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Comparison plot saved: {fig_path}")

    # Print architecture summary table
    print("\n  ┌─ Architecture Summary ─────────────────────────────────────────┐")
    print(f"  │ {'Model':<20} {'Layers':<12} {'Params':<12} {'Accuracy':<10} {'F1':<8} │")
    print(f"  │ {'─'*62} │")

    simple_params = sum(p.numel() for p in simple_model.parameters())
    deep_params   = sum(p.numel() for p in deep_model.parameters())

    print(f"  │ {'Simple FFN':<20} {'2 hidden':<12} {simple_params:<12,} {r_simple['accuracy']:<10.4f} {r_simple['f1']:<8.4f} │")
    print(f"  │ {'Deep FFN':<20} {'4 hidden':<12} {deep_params:<12,} {r_deep['accuracy']:<10.4f} {r_deep['f1']:<8.4f} │")
    print(f"  └{'─'*64}┘")

    print("\n✅ Part 4 Neural Networks complete.\n")
    results['simple_losses'] = simple_losses
    results['deep_losses']   = deep_losses
    return results


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, '.')
    from question1_eda import run_eda
    from question2_pca import run_pca

    DATA_PATH  = 'wearable_data.csv'
    OUTPUT_DIR = 'outputs'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_raw  = pd.read_csv(DATA_PATH)
    eda_out = run_eda(df_raw, output_dir=OUTPUT_DIR)
    df      = eda_out['df']

    pca_out = run_pca(df, output_dir=OUTPUT_DIR)
    nn_out  = run_neural_networks(
        pca_out['X_train_pca'], pca_out['X_test_pca'],
        pca_out['y_train'],     pca_out['y_test'],
        pca_out['classes'],     OUTPUT_DIR
    )
