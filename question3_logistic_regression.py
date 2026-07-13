"""
Question 3: Logistic Regression [25 points]
===========================================
This module handles:
  a. Logistic Regression without regularization
  b. LR with L1 (Lasso) regularization
  c. LR with L2 (Ridge) regularization
  d. Discussion of regularization impact (printed analysis)
  e. Assumptions about data split and subject-leakage consideration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
import seaborn as sns


def _evaluate_model(model, X_train, X_test, y_train, y_test, label: str,
                    class_names, output_dir: str) -> dict:
    """Train model, evaluate, and plot confusion matrix."""
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n  {'─'*55}")
    print(f"  {label}")
    print(f"  {'─'*55}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 (wtd) : {f1:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {label}", fontsize=11, fontweight='bold')
    plt.tight_layout()
    safe_label = label.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '')
    fig_path = f"{output_dir}/q3_cm_{safe_label}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved: {fig_path}")

    return {'label': label, 'accuracy': acc, 'f1': f1,
            'report': report, 'confusion_matrix': cm, 'model': model}


def run_logistic_regression(X_train_pca, X_test_pca, y_train, y_test,
                            class_names, output_dir: str = '.') -> dict:
    """
    Execute Logistic Regression sub-parts 3a-3e.

    Returns dict with results for 3a, 3b, 3c and comparison DataFrame.
    """
    results = {}

    print("\n" + "="*70)
    print("PART 3: Logistic Regression (using PCA-reduced features)")
    print("="*70)

    # ── 3a: No regularization (penalty=None) ─────────────────────────────────
    print("\n--- 3a: Logistic Regression — No Regularization ---")
    model_no_reg = LogisticRegression(
        penalty=None,
        solver='lbfgs',
        multi_class='auto',
        max_iter=5000,
        random_state=42
    )
    r_no_reg = _evaluate_model(model_no_reg, X_train_pca, X_test_pca,
                               y_train, y_test, 'LR No Regularization', class_names, output_dir)
    results['3a'] = r_no_reg

    # ── 3b: L1 regularization (Lasso) ────────────────────────────────────────
    print("\n--- 3b: Logistic Regression — L1 (Lasso) Regularization ---")
    model_l1 = LogisticRegression(
        penalty='l1',
        solver='saga',      # saga supports L1 multiclass
        C=1.0,
        multi_class='auto',
        max_iter=5000,
        random_state=42
    )
    r_l1 = _evaluate_model(model_l1, X_train_pca, X_test_pca,
                           y_train, y_test, 'LR L1 (Lasso)', class_names, output_dir)
    results['3b'] = r_l1

    # ── 3c: L2 regularization (Ridge) ────────────────────────────────────────
    print("\n--- 3c: Logistic Regression — L2 (Ridge) Regularization ---")
    model_l2 = LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        C=1.0,
        multi_class='auto',
        max_iter=5000,
        random_state=42
    )
    r_l2 = _evaluate_model(model_l2, X_train_pca, X_test_pca,
                           y_train, y_test, 'LR L2 (Ridge)', class_names, output_dir)
    results['3c'] = r_l2

    # ── Comparison bar chart ──────────────────────────────────────────────────
    labels  = ['No Reg (3a)', 'L1 Lasso (3b)', 'L2 Ridge (3c)']
    accs    = [r_no_reg['accuracy'], r_l1['accuracy'], r_l2['accuracy']]
    f1s     = [r_no_reg['f1'],       r_l1['f1'],       r_l2['f1']]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, accs, width, label='Accuracy', color='steelblue', alpha=0.85)
    bars2 = ax.bar(x + width/2, f1s,  width, label='F1 (weighted)', color='darkorange', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim([0, 1.05])
    ax.set_ylabel('Score')
    ax.set_title('Logistic Regression: Accuracy & F1 by Regularization', fontsize=12, fontweight='bold')
    ax.legend()
    for bar in bars1: ax.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x()+bar.get_width()/2, bar.get_height()), ha='center', va='bottom', fontsize=9)
    for bar in bars2: ax.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x()+bar.get_width()/2, bar.get_height()), ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    fig_path = f"{output_dir}/q3_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Comparison plot saved: {fig_path}")

    comparison_df = pd.DataFrame({
        'Model':    ['No Regularization', 'L1 (Lasso)', 'L2 (Ridge)'],
        'Accuracy': [round(a, 4) for a in accs],
        'F1 (wtd)': [round(f, 4) for f in f1s],
    })
    results['comparison'] = comparison_df

    # ── 3d: Discussion ────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("PART 3d: Discussion — Impact of Regularization")
    print("="*70)
    discussion_3d = """
  Regularization Impact Analysis:
  ─────────────────────────────────────────────────────────────────────
  • No Regularization (3a): The model is free to fit the training data
    without any penalty on coefficient size. This can lead to overfitting,
    especially when features are correlated (though PCA reduces this).
    Performance on the test set serves as the baseline.

  • L1 (Lasso) Regularization (3b): Adds |β| penalty, encouraging sparse
    solutions — many coefficients are driven exactly to zero, effectively
    performing feature selection among PCA components. This is beneficial
    when only a few PCs are truly discriminative. However, it may discard
    slightly useful components, potentially reducing accuracy marginally.

  • L2 (Ridge) Regularization (3c): Adds β² penalty, shrinking all
    coefficients toward zero uniformly without zeroing any. This handles
    multicollinearity well and typically yields more stable, slightly better
    generalisation than L1 on dense problems. For PCA-reduced data (already
    decorrelated), L2 provides a smooth regularization effect.

  Summary: On a PCA-reduced, decorrelated feature space, L2 (Ridge) tends
  to perform similarly to or slightly better than L1 and No-regularization
  because PCA already eliminates most redundancy. L1 may aggressively
  prune components, while the unregularized model risks overfitting if the
  number of retained components is large relative to sample size.
    """
    print(discussion_3d)
    results['discussion_3d'] = discussion_3d.strip()

    # ── 3e: Assumptions & Subject Leakage ─────────────────────────────────────
    print("\n" + "="*70)
    print("PART 3e: Data Split Assumptions & Subject Leakage")
    print("="*70)
    discussion_3e = """
  Assumptions Made:
  ─────────────────────────────────────────────────────────────────────
  1. Random stratified split (80/20): Stratification ensures the class
     proportions are maintained in both train and test sets, which is
     important given the class imbalance (parasympathetic:125, sympathetic:77,
     relaxed:39).

  2. i.i.d. assumption: The standard split assumes all samples are
     independent and identically distributed. However, each subject
     contributes MULTIPLE rows (one per 60-second window), so this
     assumption is VIOLATED — the data has a repeated-measures structure.

  Subject Leakage Problem:
  ─────────────────────────────────────────────────────────────────────
  If the same subject appears in BOTH train and test sets, the model can
  effectively memorise individual physiological baselines rather than
  learning generalizable class patterns. This inflates test performance —
  the model "recognises" a person rather than classifying their state.

  How to Address:
  ─────────────────────────────────────────────────────────────────────
  • Use GroupShuffleSplit or LeaveOneGroupOut (LOGO) with subject as the
    group key to ensure each subject appears ONLY in train OR test.
  • This simulates real-world deployment where the model encounters
    entirely new subjects it has never seen during training.
  • For cross-validation, use GroupKFold with groups=subject identifiers.
    """
    print(discussion_3e)
    results['discussion_3e'] = discussion_3e.strip()

    print("\n✅ Part 3 Logistic Regression complete.\n")
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
    lr_out  = run_logistic_regression(
        pca_out['X_train_pca'], pca_out['X_test_pca'],
        pca_out['y_train'],     pca_out['y_test'],
        pca_out['classes'],     OUTPUT_DIR
    )
