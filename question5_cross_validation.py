"""
Question 5: Cross-Validation Folds [20 points]
================================================
This module handles:
  a. K-Fold cross-validation on Logistic Regression with k=5, 10
  b. Stratified vs. non-stratified cross-validation analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_score,
                                     cross_validate)
from sklearn.metrics import f1_score, make_scorer
import seaborn as sns


def run_cross_validation(X_train_pca, X_test_pca, y_train, y_test,
                         class_names, output_dir: str = '.') -> dict:
    """
    Execute Cross-Validation sub-parts 5a-5b.

    Parameters
    ----------
    X_train_pca, X_test_pca : PCA-reduced feature arrays
    y_train, y_test         : encoded target arrays
    class_names             : list of class label strings
    output_dir              : directory to save figures

    Returns
    -------
    dict with CV results for all configurations
    """
    results = {}

    # Combine train+test for CV over full dataset
    X_full = np.vstack([X_train_pca, X_test_pca])
    y_full = np.concatenate([y_train, y_test])

    # Base model (L2, consistent with best LR)
    lr_model = LogisticRegression(
        penalty='l2', solver='lbfgs', C=1.0,
        max_iter=5000, random_state=42
    )

    f1_scorer = make_scorer(f1_score, average='weighted', zero_division=0)

    print("\n" + "="*70)
    print("PART 5: Cross-Validation Analysis (Logistic Regression, L2)")
    print("="*70)
    print(f"  Dataset size: {len(y_full)} samples | Features: {X_full.shape[1]} PCA components")

    # ── 5a: K-Fold with k=5 and k=10 ──────────────────────────────────────────
    print("\n--- 5a: K-Fold Cross-Validation (k=5 and k=10) ---")

    cv_results_5a = {}
    for k in [5, 10]:
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        cv_out = cross_validate(
            lr_model, X_full, y_full, cv=kf,
            scoring={'accuracy': 'accuracy', 'f1': f1_scorer},
            return_train_score=True
        )
        mean_acc  = cv_out['test_accuracy'].mean()
        std_acc   = cv_out['test_accuracy'].std()
        mean_f1   = cv_out['test_f1'].mean()
        std_f1    = cv_out['test_f1'].std()

        print(f"\n  KFold(k={k}) — Non-Stratified:")
        print(f"    Accuracy : {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"    F1 (wtd) : {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"    Per-fold accuracies: {[round(x,4) for x in cv_out['test_accuracy']]}")

        cv_results_5a[k] = {
            'accuracy_mean': mean_acc, 'accuracy_std': std_acc,
            'f1_mean': mean_f1,        'f1_std': std_f1,
            'fold_accs': cv_out['test_accuracy'],
            'fold_f1s':  cv_out['test_f1'],
        }

    results['5a'] = cv_results_5a

    # ── 5b: Stratified vs Non-Stratified ─────────────────────────────────────
    print("\n--- 5b: Stratified vs Non-Stratified Cross-Validation ---")

    configurations = [
        ('KFold k=5',            KFold(n_splits=5,  shuffle=True, random_state=42),          'non-strat'),
        ('KFold k=10',           KFold(n_splits=10, shuffle=True, random_state=42),           'non-strat'),
        ('StratifiedKFold k=5',  StratifiedKFold(n_splits=5,  shuffle=True, random_state=42), 'strat'),
        ('StratifiedKFold k=10', StratifiedKFold(n_splits=10, shuffle=True, random_state=42), 'strat'),
    ]

    all_results = []
    for name, cv_obj, strat_type in configurations:
        cv_out = cross_validate(
            lr_model, X_full, y_full, cv=cv_obj,
            scoring={'accuracy': 'accuracy', 'f1': f1_scorer},
            return_train_score=False
        )
        row = {
            'Config': name,
            'Type': strat_type,
            'Acc Mean': round(cv_out['test_accuracy'].mean(), 4),
            'Acc Std':  round(cv_out['test_accuracy'].std(),  4),
            'F1 Mean':  round(cv_out['test_f1'].mean(), 4),
            'F1 Std':   round(cv_out['test_f1'].std(),  4),
            'fold_accs': cv_out['test_accuracy'],
        }
        all_results.append(row)
        print(f"\n  {name} ({strat_type}):")
        print(f"    Accuracy : {row['Acc Mean']:.4f} ± {row['Acc Std']:.4f}")
        print(f"    F1 (wtd) : {row['F1 Mean']:.4f} ± {row['F1 Std']:.4f}")

    results['5b'] = all_results

    comparison_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'fold_accs'}
        for r in all_results
    ])
    print(f"\n  Summary Table:\n{comparison_df.to_string(index=False)}")
    results['comparison_df'] = comparison_df

    # ── Visualizations ────────────────────────────────────────────────────────

    # 1. Box plot of per-fold accuracy across all configs
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Cross-Validation: Stratified vs Non-Stratified", fontsize=13, fontweight='bold')

    config_names = [r['Config'] for r in all_results]
    fold_acc_data = [r['fold_accs'] for r in all_results]
    colors_box = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

    bp = axes[0].boxplot(fold_acc_data, label=config_names, patch_artist=True, notch=False)
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_title("Per-Fold Accuracy Distribution")
    axes[0].set_ylabel("Accuracy")
    axes[0].tick_params(axis='x', rotation=20)
    axes[0].grid(alpha=0.3, axis='y')
    axes[0].set_ylim([0, 1.05])

    # 2. Mean ± Std bar chart for Accuracy and F1
    x = np.arange(len(all_results)); w = 0.35
    acc_means = [r['Acc Mean'] for r in all_results]
    acc_stds  = [r['Acc Std']  for r in all_results]
    f1_means  = [r['F1 Mean']  for r in all_results]
    f1_stds   = [r['F1 Std']   for r in all_results]

    axes[1].bar(x - w/2, acc_means, w, yerr=acc_stds, capsize=4,
                label='Accuracy', color='steelblue', alpha=0.85)
    axes[1].bar(x + w/2, f1_means,  w, yerr=f1_stds,  capsize=4,
                label='F1 (weighted)', color='darkorange', alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([r['Config'].replace(' ', '\n') for r in all_results], fontsize=8)
    axes[1].set_ylim([0, 1.15])
    axes[1].set_ylabel("Score")
    axes[1].set_title("Mean ± Std: Accuracy & F1 Across Configurations")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path = f"{output_dir}/q5_crossval_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Cross-validation comparison plot saved: {fig_path}")

    # ── 5b: Observations (printed) ────────────────────────────────────────────
    print("\n" + "="*70)
    print("PART 5b: Observations & Analysis")
    print("="*70)
    observations = """
  Impact of Number of Folds (k):
  ─────────────────────────────────────────────────────────────────────
  • Increasing k from 5 to 10 generally DECREASES variance (std) of
    performance estimates because each fold uses more data for training
    and the average is over more evaluations.
  • However, k=10 is computationally more expensive (10 model fits vs 5).
  • With a small dataset (n≈241), k=10 can cause very small test folds
    where rare class samples may be absent, destabilizing estimates.
  • k=5 offers a good bias-variance tradeoff for this dataset size.

  Impact of Stratification:
  ─────────────────────────────────────────────────────────────────────
  • Without stratification, random fold assignment can produce folds
    where minority classes (relaxed: 39 samples) are severely under-
    represented or entirely absent in a test fold.
  • Stratified KFold preserves class proportions in every fold, giving
    more representative and STABLE estimates of generalisation performance.
  • The standard deviation of fold accuracies is typically LOWER with
    stratification because each fold sees balanced class distributions.
  • Stratification is STRONGLY recommended for imbalanced datasets like
    this one (parasympathetic:125, sympathetic:77, relaxed:39).
    """
    print(observations)
    results['observations'] = observations.strip()

    print("\n✅ Part 5 Cross-Validation complete.\n")
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
    cv_out  = run_cross_validation(
        pca_out['X_train_pca'], pca_out['X_test_pca'],
        pca_out['y_train'],     pca_out['y_test'],
        pca_out['classes'],     OUTPUT_DIR
    )
