"""
Question 1: Exploratory Data Analysis (EDA) [30 points]
=========================================================
This module handles:
  a. Loading data, handling missing values, encoding categoricals
  b. Class distribution visualization
  c. Per-modality correlation heatmaps
  d. Descriptive statistics per class per modality
  e. Box plots grouped by target class
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for saving figures
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Column groups by sensor modality ─────────────────────────────────────────
MODALITY_COLS = {
    'accelerometer': [
        'accelerometer_magnitude_mean', 'accelerometer_magnitude_std_dev',
        'accelerometer_magnitude_mean_derivative', 'accelerometer_magnitude_rms',
        'accelerometer_magnitude_peak_to_peak', 'accelerometer_magnitude_peak_to_rms',
        'accelerometer_magnitude_num_peaks', 'accelerometer_magnitude_sample_entropy',
    ],
    'BVP': [
        'bvp_light_absorption_nW_mean', 'bvp_light_absorption_nW_std_dev',
        'bvp_light_absorption_nW_mean_derivative', 'bvp_light_absorption_nW_rms',
        'bvp_light_absorption_nW_peak_to_peak', 'bvp_light_absorption_nW_peak_to_rms',
        'bvp_light_absorption_nW_num_peaks', 'bvp_light_absorption_nW_sample_entropy',
    ],
    'EDA': [
        'eda_microsiemens_mean', 'eda_microsiemens_std_dev',
        'eda_microsiemens_mean_derivative', 'eda_microsiemens_rms',
        'eda_microsiemens_peak_to_peak', 'eda_microsiemens_peak_to_rms',
        'eda_microsiemens_num_peaks', 'eda_microsiemens_sample_entropy',
    ],
    'temperature': [
        'temperature_celcius_mean', 'temperature_celcius_std_dev',
        'temperature_celcius_mean_derivative', 'temperature_celcius_rms',
        'temperature_celcius_peak_to_peak', 'temperature_celcius_peak_to_rms',
        'temperature_celcius_num_peaks', 'temperature_celcius_sample_entropy',
    ],
    'RR_interval': [
        'rr_interval_milliseconds_mean', 'rr_interval_milliseconds_std_dev',
        'rr_interval_milliseconds_mean_derivative', 'rr_interval_milliseconds_rms',
        'rr_interval_milliseconds_peak_to_peak', 'rr_interval_milliseconds_peak_to_rms',
        'rr_interval_milliseconds_num_peaks', 'rr_interval_milliseconds_sample_entropy',
    ],
}

CATEGORICAL_COLS = ['gender', 'posture_type', 'yoga_experience', 'difficulty_level', 'state', 'subject_state', 'subject']


def run_eda(df_raw: pd.DataFrame, output_dir: str = '.') -> dict:
    """
    Execute all EDA sub-parts (1a-1e).

    Parameters
    ----------
    df_raw     : raw DataFrame loaded from CSV
    output_dir : directory where figure PNGs are saved

    Returns
    -------
    dict with keys: df, encoders, results_text
    """
    results = {}

    # ── 1a: Load, explore dimensions, handle missing values ───────────────────
    print("\n" + "="*70)
    print("PART 1a: Dataset Overview & Encoding")
    print("="*70)

    df = df_raw.copy()

    # Drop non-informative / leaky columns
    drop_cols = ['timestamp', 'subject_state']   # subject_state encodes subject+posture
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    print(f"Dataset shape: {df.shape}")
    print(f"\nMissing values per column (top 10):")
    missing = df.isnull().sum()
    print(missing[missing > 0].sort_values(ascending=False).head(10))
    total_missing = df.isnull().sum().sum()
    pct_missing = total_missing / (df.shape[0] * df.shape[1]) * 100
    print(f"\nTotal missing cells: {total_missing} ({pct_missing:.2f}%)")

    # Strategy: fill numerical NaN with column median (robust to outliers)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    print(f"\nMissing values after imputation: {df.isnull().sum().sum()}")

    # Categorical columns present in df
    cat_cols_present = [c for c in CATEGORICAL_COLS if c in df.columns]
    print(f"\nCategorical columns: {cat_cols_present}")

    # Encode categorical columns with LabelEncoder
    from sklearn.preprocessing import LabelEncoder
    encoders = {}
    for col in cat_cols_present:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    results['df'] = df
    results['encoders'] = encoders

    # ── 1b: Class distribution ────────────────────────────────────────────────
    print("\n" + "="*70)
    print("PART 1b: Class Distribution of Target Variable 'state'")
    print("="*70)

    class_counts = df['state'].value_counts()
    print(class_counts.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Class Distribution of Target Variable: state", fontsize=14, fontweight='bold')

    # Bar plot
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    axes[0].bar(class_counts.index, class_counts.values, color=colors, edgecolor='black', alpha=0.85)
    axes[0].set_title('Count per Class')
    axes[0].set_xlabel('State')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 1, str(v), ha='center', fontweight='bold')

    # Pie chart
    axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=140)
    axes[1].set_title('Proportion per Class')

    plt.tight_layout()
    fig_path = f"{output_dir}/q1b_class_distribution.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")
    results['class_counts'] = class_counts

    # ── 1c: Per-modality correlation heatmaps ─────────────────────────────────
    print("\n" + "="*70)
    print("PART 1c: Per-Modality Correlation Heatmaps (5 heatmaps)")
    print("="*70)

    corr_observations = {}
    for mod, cols in MODALITY_COLS.items():
        existing_cols = [c for c in cols if c in df.columns]
        if not existing_cols:
            continue
        corr = df[existing_cols].corr()

        # Short column labels for readability
        short_labels = [c.split('_')[-2] + '_' + c.split('_')[-1] if '_' in c else c for c in existing_cols]

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    xticklabels=short_labels, yticklabels=short_labels,
                    ax=ax, linewidths=0.5, vmin=-1, vmax=1)
        ax.set_title(f"Correlation Matrix — {mod} modality", fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig_path = f"{output_dir}/q1c_corr_{mod}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  {mod}: saved to {fig_path}")

        # Identify highly correlated pairs (|r| > 0.85, excluding diagonal)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        high_corr = [(existing_cols[i], existing_cols[j], round(corr.iloc[i, j], 3))
                     for i in range(len(existing_cols))
                     for j in range(i+1, len(existing_cols))
                     if abs(corr.iloc[i, j]) > 0.85]
        corr_observations[mod] = high_corr
        if high_corr:
            print(f"    High-correlation pairs (|r|>0.85): {high_corr}")

    results['corr_observations'] = corr_observations

    # ── 1d: Descriptive stats per modality per class ──────────────────────────
    print("\n" + "="*70)
    print("PART 1d: Descriptive Statistics Per Sensor Modality × Class")
    print("="*70)

    variation_scores = {}
    for mod, cols in MODALITY_COLS.items():
        existing_cols = [c for c in cols if c in df.columns]
        if not existing_cols:
            continue
        stats = df.groupby('state')[existing_cols].agg(['mean', 'std'])
        print(f"\n--- {mod} ---")
        print(stats.to_string())

        # Measure between-class variation: mean of (std of class-means across features)
        class_means = df.groupby('state')[existing_cols].mean()
        variation = class_means.std(axis=0).mean()
        variation_scores[mod] = round(variation, 4)

    print(f"\nBetween-class variation score per modality:")
    for mod, score in sorted(variation_scores.items(), key=lambda x: -x[1]):
        print(f"  {mod}: {score}")
    best_modality = max(variation_scores, key=variation_scores.get)
    print(f"\n→ Modality with greatest class variation: {best_modality}")
    results['variation_scores'] = variation_scores

    # ── 1e: Box plots grouped by class ────────────────────────────────────────
    print("\n" + "="*70)
    print("PART 1e: Box Plots (one feature per modality, grouped by state)")
    print("="*70)

    representative_features = {
        'accelerometer': 'accelerometer_magnitude_mean',
        'BVP': 'bvp_light_absorption_nW_mean',
        'EDA': 'eda_microsiemens_mean',
        'temperature': 'temperature_celcius_mean',
        'RR_interval': 'rr_interval_milliseconds_mean',
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    palette = {'relaxed': '#2ecc71', 'parasympathetic': '#3498db', 'sympathetic': '#e74c3c'}

    for idx, (mod, feat) in enumerate(representative_features.items()):
        if feat not in df.columns:
            continue
        ax = axes[idx]
        sns.boxplot(data=df, x='state', y=feat, palette=palette, ax=ax,
                    order=['relaxed', 'parasympathetic', 'sympathetic'])
        ax.set_title(f"{mod}\n({feat.split('_')[-1].replace('mean','mean')})",
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('State')
        ax.tick_params(axis='x', rotation=15)

    # Turn off unused subplot
    for idx in range(len(representative_features), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Box Plots: One Feature per Sensor Modality by Target Class",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig_path = f"{output_dir}/q1e_boxplots.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    print("\n✅ Part 1 EDA complete.\n")
    return results


if __name__ == '__main__':
    DATA_PATH = 'wearable_data.csv'
    OUTPUT_DIR = 'outputs'
    import os; os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_raw = pd.read_csv(DATA_PATH)
    results = run_eda(df_raw, output_dir=OUTPUT_DIR)
