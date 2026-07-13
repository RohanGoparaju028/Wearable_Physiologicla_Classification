"""
Question 2: Principal Component Analysis (PCA) [30 points]
===========================================================
This module handles:
  a. Standardize features, apply full PCA, plot explained variance
  b. Choose components for >=90% variance, scatter plot PC1 vs PC2
  c. Factor loadings for first 3 PCs, top-5 contributing features per PC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def run_pca(df: pd.DataFrame, output_dir: str = '.', test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Execute PCA sub-parts 2a-2c.

    Parameters
    ----------
    df           : cleaned/encoded DataFrame from question1_eda
    output_dir   : directory where figure PNGs are saved
    test_size    : fraction for test split
    random_state : seed for reproducibility

    Returns
    -------
    dict with X_train_pca, X_test_pca, y_train, y_test, pca, scaler, n_components
    """
    results = {}

    # ── Identify feature columns & target ─────────────────────────────────────
    # Drop all non-numeric / meta columns; keep numeric sensor features
    meta_cols = ['timestamp', 'subject_state', 'state', 'gender', 'posture_type',
                 'yoga_experience', 'difficulty_level', 'subject',
                 'state_enc', 'gender_enc', 'posture_type_enc',
                 'yoga_experience_enc', 'difficulty_level_enc', 'subject_enc']

    feature_cols = [c for c in df.columns
                    if c not in meta_cols
                    and df[c].dtype in [np.float64, np.int64, float, int]]

    print(f"Number of features used for PCA: {len(feature_cols)}")

    X = df[feature_cols].values
    y_raw = df['state'].values if 'state' in df.columns else df['state_enc'].values

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    results['label_encoder'] = le
    results['classes'] = le.classes_

    # ── Train/Test split (stratified) ─────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\nPre-PCA shapes  → X_train: {X_train.shape}, X_test: {X_test.shape}")

    # ── 2a: Standardize & full PCA ────────────────────────────────────────────
    print("\n" + "="*70)
    print("PART 2a: Standardize + Full PCA + Explained Variance Plots")
    print("="*70)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print(f"Scaled feature matrix shape (train): {X_train_scaled.shape}")

    pca_full = PCA(random_state=random_state)
    pca_full.fit(X_train_scaled)

    ev_ratio   = pca_full.explained_variance_ratio_
    cumul_ev   = np.cumsum(ev_ratio)
    n_all      = len(ev_ratio)
    components = np.arange(1, n_all + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PCA Explained Variance", fontsize=14, fontweight='bold')

    # Bar chart of per-component variance
    axes[0].bar(components, ev_ratio * 100, color='steelblue', edgecolor='white', alpha=0.85)
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance Ratio (%)")
    axes[0].set_title("Per-Component Explained Variance (Scree Plot)")
    axes[0].set_xticks(components[::max(1, n_all//15)])

    # Cumulative variance with 90% reference line
    axes[1].plot(components, cumul_ev * 100, marker='o', markersize=4,
                 color='darkorange', linewidth=2)
    axes[1].axhline(90, color='red', linestyle='--', linewidth=1.5, label='90% threshold')
    n_90 = np.searchsorted(cumul_ev, 0.90) + 1
    axes[1].axvline(n_90, color='green', linestyle=':', linewidth=1.5,
                    label=f'{n_90} components → 90%')
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Explained Variance (%)")
    axes[1].set_title("Cumulative Explained Variance Curve")
    axes[1].legend()
    axes[1].set_ylim([0, 101])

    plt.tight_layout()
    fig_path = f"{output_dir}/q2a_pca_variance.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")
    print(f"  Components needed for ≥90% variance: {n_90}")

    # ── 2b: Apply reduced PCA ─────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"PART 2b: Apply PCA with {n_90} components (≥90% variance retained)")
    print("="*70)

    pca = PCA(n_components=n_90, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    print(f"Shape BEFORE PCA: X_train={X_train_scaled.shape}, X_test={X_test_scaled.shape}")
    print(f"Shape AFTER  PCA: X_train={X_train_pca.shape},   X_test={X_test_pca.shape}")
    print(f"Total variance retained: {pca.explained_variance_ratio_.sum()*100:.2f}%")

    # Scatter plot: PC1 vs PC2 coloured by class
    class_names = le.classes_
    colors_map = {0: '#2ecc71', 1: '#3498db', 2: '#e74c3c'}
    color_labels = {cls: colors_map[i] for i, cls in enumerate(class_names)}

    fig, ax = plt.subplots(figsize=(9, 7))
    for cls_idx, cls_name in enumerate(class_names):
        mask = y_train == cls_idx
        ax.scatter(X_train_pca[mask, 0], X_train_pca[mask, 1],
                   c=colors_map[cls_idx], label=cls_name, alpha=0.65, s=50, edgecolors='k', linewidths=0.3)
    ax.set_xlabel(f"PC 1 ({ev_ratio[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC 2 ({ev_ratio[1]*100:.1f}% var)")
    ax.set_title("PCA Scatter Plot — PC1 vs PC2 (coloured by state)", fontsize=12, fontweight='bold')
    ax.legend(title='State', framealpha=0.8)
    plt.tight_layout()
    fig_path = f"{output_dir}/q2b_pca_scatter.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # ── 2c: Factor loadings for first 3 PCs ───────────────────────────────────
    print("\n" + "="*70)
    print("PART 2c: Factor Loadings — Top 5 Features per PC (first 3 PCs)")
    print("="*70)

    # loadings = eigenvectors × sqrt(explained_variance)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    loading_df = pd.DataFrame(
        loadings[:, :3],
        index=feature_cols,
        columns=[f'PC{i+1}' for i in range(3)]
    )

    for pc in ['PC1', 'PC2', 'PC3']:
        top5 = loading_df[pc].abs().nlargest(5)
        print(f"\n  Top 5 features for {pc}:")
        rows = []
        for feat in top5.index:
            rows.append({'Feature': feat, f'Loading ({pc})': round(loading_df.loc[feat, pc], 4)})
        top5_df = pd.DataFrame(rows).sort_values(f'Loading ({pc})', key=abs, ascending=False)
        print(top5_df.to_string(index=False))

    # Heatmap of loadings (all features × first 3 PCs)
    fig, ax = plt.subplots(figsize=(8, max(10, len(feature_cols) // 3)))
    short_feats = [f.replace('accelerometer_magnitude_', 'acc_')
                    .replace('bvp_light_absorption_nW_', 'bvp_')
                    .replace('eda_microsiemens_', 'eda_')
                    .replace('temperature_celcius_', 'temp_')
                    .replace('rr_interval_milliseconds_', 'rr_')
                   for f in feature_cols]
    sns.heatmap(loading_df.iloc[:, :3], annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, xticklabels=['PC1','PC2','PC3'],
                yticklabels=short_feats, ax=ax, linewidths=0.3, vmin=-1, vmax=1)
    ax.set_title("Factor Loadings: Original Features → PC1, PC2, PC3", fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig_path = f"{output_dir}/q2c_factor_loadings.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {fig_path}")

    print("\n✅ Part 2 PCA complete.\n")

    results.update({
        'X_train_pca': X_train_pca,
        'X_test_pca': X_test_pca,
        'y_train': y_train,
        'y_test': y_test,
        'pca': pca,
        'scaler': scaler,
        'n_components': n_90,
        'feature_cols': feature_cols,
        'loading_df': loading_df,
    })
    return results


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, '.')
    from question1_eda import run_eda, MODALITY_COLS

    DATA_PATH  = 'wearable_data.csv'
    OUTPUT_DIR = 'outputs'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_raw  = pd.read_csv(DATA_PATH)
    eda_out = run_eda(df_raw, output_dir=OUTPUT_DIR)
    df      = eda_out['df']

    pca_out = run_pca(df, output_dir=OUTPUT_DIR)
