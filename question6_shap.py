"""
Question 6: SHAP Analysis [20 points]
=======================================
This module handles:
  a. SHAP global feature importance bar plot + beeswarm plot
  b. SHAP waterfall plot for one test sample
  c. SHAP waterfall plot for a second test sample

Note: shap.LinearExplainer on a multi-class LogisticRegression returns
      shap_values with shape (n_samples, n_features, n_classes).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

import shap


def run_shap_analysis(best_model, X_train_pca, X_test_pca, y_test,
                      class_names, output_dir: str = '.') -> dict:
    """
    Execute SHAP analysis sub-parts 6a-6c.

    Parameters
    ----------
    best_model   : trained sklearn model (LogisticRegression)
    X_train_pca  : training features (used to build SHAP background)
    X_test_pca   : test features to explain
    y_test       : true labels for test set
    class_names  : list of class label strings
    output_dir   : directory to save figures

    Returns
    -------
    dict with SHAP values and figure paths
    """
    results = {}

    print("\n" + "="*70)
    print("PART 6: SHAP Analysis")
    print("="*70)

    n_pcs    = X_train_pca.shape[1]
    pc_names = [f"PC{i+1}" for i in range(n_pcs)]

    # ── Create SHAP LinearExplainer ───────────────────────────────────────────
    print("  Building SHAP LinearExplainer on LogisticRegression model...")
    masker    = shap.maskers.Independent(X_train_pca, max_samples=len(X_train_pca))
    explainer = shap.LinearExplainer(best_model, masker)

    print("  Computing SHAP values on test set...")
    shap_values = explainer.shap_values(X_test_pca)
    # shap_values shape: (n_samples, n_features, n_classes)  for multi-class
    print(f"  SHAP values shape: {np.array(shap_values).shape}")

    results['shap_values'] = shap_values
    results['explainer']   = explainer

    # Normalize to 3-D array regardless of sklearn/shap version
    sv_arr = np.array(shap_values)          # (n_samples, n_features, n_classes) or (n_samples, n_features)
    if sv_arr.ndim == 2:
        # Binary or already flat — expand to (n_samples, n_features, 1)
        sv_arr = sv_arr[:, :, np.newaxis]

    n_samples, n_features, n_classes = sv_arr.shape

    # ── 6a: Global feature importance ────────────────────────────────────────
    print("\n--- 6a: Global SHAP Feature Importance ---")

    # Mean |SHAP| across samples and classes → (n_features,)
    mean_abs_shap_vals = np.abs(sv_arr).mean(axis=(0, 2))   # (n_features,)
    mean_abs_shap = pd.Series(mean_abs_shap_vals, index=pc_names).sort_values(ascending=False)

    # (i) Global bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(mean_abs_shap)))
    ax.barh(mean_abs_shap.index[::-1], mean_abs_shap.values[::-1], color=colors[::-1])
    ax.set_xlabel("Mean |SHAP Value| (average impact on prediction)")
    ax.set_title("SHAP Global Feature Importance — PCA Components\n(Best Model: Logistic Regression L2)",
                 fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    fig_path = f"{output_dir}/q6a_shap_global_bar.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Global bar plot saved: {fig_path}")

    # (ii) Beeswarm plot — use class-0 SHAP values (n_samples, n_features)
    sv_class0 = sv_arr[:, :, 0]   # (n_samples, n_features)
    ev_class0 = explainer.expected_value[0] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value

    shap_exp_beeswarm = shap.Explanation(
        values=sv_class0,
        base_values=np.full(n_samples, ev_class0),
        data=X_test_pca,
        feature_names=pc_names
    )

    fig = plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_exp_beeswarm, max_display=n_pcs, show=False)
    plt.title(f"SHAP Beeswarm — Class: {class_names[0]} (vs rest)",
              fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig_path = f"{output_dir}/q6a_shap_beeswarm.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Beeswarm plot saved: {fig_path}")

    # Analysis
    top3 = mean_abs_shap.head(3)
    print(f"\n  Top 3 most influential PCA components (global):")
    for pc, val in top3.items():
        print(f"    {pc}: mean |SHAP| = {val:.4f}")

    print("""
  Discussion (6a):
  ─────────────────────────────────────────────────────────────────────
  The global SHAP bar plot ranks PCA components by their average absolute
  contribution across all test samples and classes. Components at the top
  are those the Logistic Regression model relies on most to discriminate
  physiological states. Comparing these with the factor loadings from
  Part 2c reveals which original sensor modalities (RR interval, accelero-
  meter, temperature) drive classification. The beeswarm plot additionally
  shows the DIRECTION: red points push toward the class, blue push away.
  Consistent direction across many samples indicates a strong monotonic
  relationship between that PC and the predicted class.
    """)

    # ── 6b: Waterfall — Sample 1 ──────────────────────────────────────────────
    print("\n--- 6b: SHAP Waterfall Plot — Test Sample 1 ---")

    sample_idx_1 = 0
    true_class_1 = class_names[y_test[sample_idx_1]]
    proba_1      = best_model.predict_proba(X_test_pca[[sample_idx_1]])[0]
    pred_class_1 = class_names[np.argmax(proba_1)]
    pred_cls_1   = int(np.argmax(proba_1))

    sv_1   = sv_arr[sample_idx_1, :, pred_cls_1]   # (n_features,)
    base_1 = (explainer.expected_value[pred_cls_1]
              if hasattr(explainer.expected_value, '__len__')
              else explainer.expected_value)

    shap_exp_1 = shap.Explanation(
        values=sv_1,
        base_values=float(base_1),
        data=X_test_pca[sample_idx_1],
        feature_names=pc_names
    )

    fig = plt.figure(figsize=(10, 7))
    shap.plots.waterfall(shap_exp_1, max_display=n_pcs, show=False)
    plt.title(f"SHAP Waterfall — Sample 1 | True: {true_class_1} | Pred: {pred_class_1}",
              fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig_path = f"{output_dir}/q6b_shap_waterfall_sample1.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Waterfall (sample 1) saved: {fig_path}")
    print(f"  True class: {true_class_1} | Predicted: {pred_class_1}")

    top_push_1 = pd.Series(sv_1, index=pc_names).sort_values(key=abs, ascending=False).head(3)
    print(f"\n  Top contributing features for sample 1:")
    for pc, val in top_push_1.items():
        direction = "TOWARD" if val > 0 else "AWAY FROM"
        print(f"    {pc}: SHAP={val:.4f} → {direction} '{pred_class_1}'")

    print("""
  Discussion (6b):
  ─────────────────────────────────────────────────────────────────────
  The waterfall plot traces the prediction from the base value (expected
  model output) to the final prediction for this individual sample. Red
  bars increase the log-odds toward the predicted class; blue bars
  decrease them. The dominant components reveal which physiological
  dimensions (captured by the PCs with highest loadings on specific
  modalities) pushed this particular physiological reading toward its
  classified state — providing individual-level interpretability.
    """)

    # ── 6c: Waterfall — Sample 2 ──────────────────────────────────────────────
    print("\n--- 6c: SHAP Waterfall Plot — Test Sample 2 ---")

    # Find a sample with a different true class than sample 1
    sample_idx_2 = None
    for idx in range(1, len(y_test)):
        if y_test[idx] != y_test[sample_idx_1]:
            sample_idx_2 = idx
            break
    if sample_idx_2 is None:
        sample_idx_2 = 1   # fallback

    true_class_2 = class_names[y_test[sample_idx_2]]
    proba_2      = best_model.predict_proba(X_test_pca[[sample_idx_2]])[0]
    pred_class_2 = class_names[np.argmax(proba_2)]
    pred_cls_2   = int(np.argmax(proba_2))

    sv_2   = sv_arr[sample_idx_2, :, pred_cls_2]
    base_2 = (explainer.expected_value[pred_cls_2]
              if hasattr(explainer.expected_value, '__len__')
              else explainer.expected_value)

    shap_exp_2 = shap.Explanation(
        values=sv_2,
        base_values=float(base_2),
        data=X_test_pca[sample_idx_2],
        feature_names=pc_names
    )

    fig = plt.figure(figsize=(10, 7))
    shap.plots.waterfall(shap_exp_2, max_display=n_pcs, show=False)
    plt.title(f"SHAP Waterfall — Sample 2 | True: {true_class_2} | Pred: {pred_class_2}",
              fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig_path = f"{output_dir}/q6c_shap_waterfall_sample2.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Waterfall (sample 2) saved: {fig_path}")
    print(f"  True class: {true_class_2} | Predicted: {pred_class_2}")

    top_push_2 = pd.Series(sv_2, index=pc_names).sort_values(key=abs, ascending=False).head(3)
    print(f"\n  Top contributing features for sample 2:")
    for pc, val in top_push_2.items():
        direction = "TOWARD" if val > 0 else "AWAY FROM"
        print(f"    {pc}: SHAP={val:.4f} → {direction} '{pred_class_2}'")

    print("""
  Discussion (6c):
  ─────────────────────────────────────────────────────────────────────
  Comparing this waterfall to sample 1's shows how the same model applies
  different feature weights for different individuals. The shift in
  dominant PCA components between the two samples reflects inter-subject
  variability in physiological responses — a key challenge in wearable
  health monitoring. This confirms the need for personalised models or
  subject-specific normalisation strategies in future work.
    """)

    print("\n✅ Part 6 SHAP Analysis complete.\n")
    results['mean_abs_shap'] = mean_abs_shap
    results['sample_1_idx'] = sample_idx_1
    results['sample_2_idx'] = sample_idx_2
    return results


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, '.')
    from question1_eda import run_eda
    from question2_pca import run_pca
    from question3_logistic_regression import run_logistic_regression

    DATA_PATH  = 'wearable_data.csv'
    OUTPUT_DIR = 'outputs'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_raw  = pd.read_csv(DATA_PATH)
    eda_out = run_eda(df_raw, output_dir=OUTPUT_DIR)
    pca_out = run_pca(eda_out['df'], output_dir=OUTPUT_DIR)
    lr_out  = run_logistic_regression(
        pca_out['X_train_pca'], pca_out['X_test_pca'],
        pca_out['y_train'],     pca_out['y_test'],
        pca_out['classes'],     OUTPUT_DIR
    )
    shap_out = run_shap_analysis(
        lr_out['3c']['model'],
        pca_out['X_train_pca'], pca_out['X_test_pca'],
        pca_out['y_test'],
        pca_out['classes'],     OUTPUT_DIR
    )
