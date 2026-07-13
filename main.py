"""
main.py — Wearable Physiological Signal Classification
=======================================================
Entry point that runs all parts of the final project in sequence:

  Part 1: Exploratory Data Analysis (EDA)                  [30 pts]
  Part 2: Principal Component Analysis (PCA)                [30 pts]
  Part 3: Logistic Regression                               [25 pts]
  Part 4: Neural Networks                                   [25 pts]
  Part 5: Cross-Validation with Different Folds             [20 pts]
  Part 6: SHAP Analysis                                     [20 pts]
  Part 7: Bonus Question                                    [ 5 pts]
  Total:                                                   [155 pts]

Dataset: wearable_data.csv
  - 5 sensor modalities × 8 statistical features = 40 numerical features
  - 3 target classes: relaxed, parasympathetic, sympathetic
  - Additional categorical metadata: gender, posture_type,
    yoga_experience, difficulty_level, subject

Outputs:
  All figures saved to ./outputs/
  All console output written to ./outputs/run_log.txt (via tee)

Usage:
  conda run -n base python main.py
  OR
  python main.py

Requirements:
  pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, torch, shap
"""

import os
import sys
import time
import warnings

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# ── Output directory ────────────────────────────────────────────────────────
OUTPUT_DIR = 'outputs'
DATA_PATH  = 'wearable_data.csv'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Import question modules ──────────────────────────────────────────────────
from question1_eda              import run_eda
from question2_pca              import run_pca
from question3_logistic_regression import run_logistic_regression
from question4_neural_networks  import run_neural_networks
from question5_cross_validation import run_cross_validation
from question6_shap             import run_shap_analysis
from question7_bonus            import run_bonus


def print_header(title: str, part_num: int, points: int):
    """Print a formatted section header."""
    bar = "═" * 70
    print(f"\n{bar}")
    print(f"  PART {part_num}: {title}  [{points} points]")
    print(f"{bar}")


def main():
    t0 = time.time()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║   Wearable Physiological Signal Classification — Final Project       ║")
    print("║   Building Machine Learning Models for ANS State Classification      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"\n  Dataset : {DATA_PATH}")
    print(f"  Outputs : ./{OUTPUT_DIR}/\n")

    # ── Load raw data ────────────────────────────────────────────────────────
    print("  Loading dataset...")
    df_raw = pd.read_csv(DATA_PATH)
    print(f"  Raw data shape: {df_raw.shape}")

    # ─────────────────────────────────────────────────────────────────────────
    # PART 1 — EDA
    # ─────────────────────────────────────────────────────────────────────────
    print_header("Exploratory Data Analysis (EDA)", 1, 30)
    eda_results = run_eda(df_raw, output_dir=OUTPUT_DIR)
    df_clean    = eda_results['df']
    t1 = time.time()
    print(f"\n  ✓ Part 1 done in {t1-t0:.1f}s")

    # ─────────────────────────────────────────────────────────────────────────
    # PART 2 — PCA
    # ─────────────────────────────────────────────────────────────────────────
    print_header("Principal Component Analysis (PCA)", 2, 30)
    pca_results = run_pca(df_clean, output_dir=OUTPUT_DIR)
    t2 = time.time()
    print(f"\n  ✓ Part 2 done in {t2-t1:.1f}s")

    # Unpack PCA outputs (shared across parts 3-6)
    X_train_pca = pca_results['X_train_pca']
    X_test_pca  = pca_results['X_test_pca']
    y_train     = pca_results['y_train']
    y_test      = pca_results['y_test']
    class_names = pca_results['classes']

    print(f"\n  PCA-reduced shapes: X_train={X_train_pca.shape}, X_test={X_test_pca.shape}")
    print(f"  Classes: {list(class_names)}")
    print(f"  Class distribution in test: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # ─────────────────────────────────────────────────────────────────────────
    # PART 3 — Logistic Regression
    # ─────────────────────────────────────────────────────────────────────────
    print_header("Logistic Regression", 3, 25)
    lr_results = run_logistic_regression(
        X_train_pca, X_test_pca, y_train, y_test, class_names, OUTPUT_DIR
    )
    t3 = time.time()
    print(f"\n  ✓ Part 3 done in {t3-t2:.1f}s")
    print(f"\n  Comparison Table:")
    print(lr_results['comparison'].to_string(index=False))

    # Best LR model selection (used for SHAP)
    best_lr_model = lr_results['3c']['model']   # L2 Ridge, typically most stable

    # ─────────────────────────────────────────────────────────────────────────
    # PART 4 — Neural Networks
    # ─────────────────────────────────────────────────────────────────────────
    print_header("Neural Networks (PyTorch)", 4, 25)
    nn_results = run_neural_networks(
        X_train_pca, X_test_pca, y_train, y_test, class_names, OUTPUT_DIR
    )
    t4 = time.time()
    print(f"\n  ✓ Part 4 done in {t4-t3:.1f}s")

    # ─────────────────────────────────────────────────────────────────────────
    # PART 5 — Cross-Validation
    # ─────────────────────────────────────────────────────────────────────────
    print_header("Cross-Validation with Different Folds", 5, 20)
    cv_results = run_cross_validation(
        X_train_pca, X_test_pca, y_train, y_test, class_names, OUTPUT_DIR
    )
    t5 = time.time()
    print(f"\n  ✓ Part 5 done in {t5-t4:.1f}s")

    # ─────────────────────────────────────────────────────────────────────────
    # PART 6 — SHAP Analysis
    # ─────────────────────────────────────────────────────────────────────────
    print_header("SHAP Analysis", 6, 20)
    shap_results = run_shap_analysis(
        best_lr_model, X_train_pca, X_test_pca, y_test, class_names, OUTPUT_DIR
    )
    t6 = time.time()
    print(f"\n  ✓ Part 6 done in {t6-t5:.1f}s")

    # ─────────────────────────────────────────────────────────────────────────
    # PART 7 — Bonus
    # ─────────────────────────────────────────────────────────────────────────
    print_header("Bonus Question", 7, 5)
    bonus_text = run_bonus()
    t7 = time.time()
    print(f"\n  ✓ Part 7 done in {t7-t6:.1f}s")

    # ─────────────────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    total_time = t7 - t0
    print("\n" + "═"*70)
    print("  FINAL SUMMARY — All Parts Complete")
    print("═"*70)

    print("\n  Model Performance Summary:")
    print(f"  {'Model':<35} {'Accuracy':>10} {'F1 (wtd)':>10}")
    print(f"  {'-'*57}")
    print(f"  {'LR — No Regularization':<35} {lr_results['3a']['accuracy']:>10.4f} {lr_results['3a']['f1']:>10.4f}")
    print(f"  {'LR — L1 (Lasso)':<35} {lr_results['3b']['accuracy']:>10.4f} {lr_results['3b']['f1']:>10.4f}")
    print(f"  {'LR — L2 (Ridge)':<35} {lr_results['3c']['accuracy']:>10.4f} {lr_results['3c']['f1']:>10.4f}")
    print(f"  {'Neural Net — Simple FFN':<35} {nn_results['simple']['accuracy']:>10.4f} {nn_results['simple']['f1']:>10.4f}")
    print(f"  {'Neural Net — Deep FFN':<35} {nn_results['deep']['accuracy']:>10.4f} {nn_results['deep']['f1']:>10.4f}")

    # Best model
    all_accs = {
        'LR No Reg':    lr_results['3a']['accuracy'],
        'LR L1':        lr_results['3b']['accuracy'],
        'LR L2':        lr_results['3c']['accuracy'],
        'Simple FFN':   nn_results['simple']['accuracy'],
        'Deep FFN':     nn_results['deep']['accuracy'],
    }
    best_name = max(all_accs, key=all_accs.get)
    print(f"\n  ★ Best model: {best_name} (accuracy={all_accs[best_name]:.4f})")

    print(f"\n  Output figures saved in: ./{OUTPUT_DIR}/")
    print(f"  Total runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print("\n" + "═"*70 + "\n")

    return {
        'eda':   eda_results,
        'pca':   pca_results,
        'lr':    lr_results,
        'nn':    nn_results,
        'cv':    cv_results,
        'shap':  shap_results,
        'bonus': bonus_text,
    }


if __name__ == '__main__':
    main()
