# Wearable Physiological Signal Classification
### Final Project — Building Machine Learning Models for ANS State Classification

> **Dataset:** `wearable_data.csv` — Physiological signals from wearable sensors during yoga postures  
> **Target:** 3-class classification — `relaxed`, `parasympathetic`, `sympathetic`  
> **Total Points:** 155 (150 + 5 bonus)

---

## Project Structure

```
Wearable_Physiologicla_Classification/
│
├── wearable_data.csv              # Dataset (241 samples, 47 columns)
├── Questions.pdf                  # Assignment questions
│
├── Dockerfile                     # Docker configuration file
├── .dockerignore                  # Docker ignore file
├── requirments.txt                # Python library dependencies
├── main.py                        # ← Entry point: runs all parts in sequence
├── question1_eda.py               # Part 1: Exploratory Data Analysis
├── question2_pca.py               # Part 2: Principal Component Analysis
├── question3_logistic_regression.py  # Part 3: Logistic Regression
├── question4_neural_networks.py   # Part 4: Neural Networks (PyTorch)
├── question5_cross_validation.py  # Part 5: Cross-Validation
├── question6_shap.py              # Part 6: SHAP Analysis
├── question7_bonus.py             # Part 7: Bonus Question
│
└── outputs/                       # Auto-generated figures & logs
    ├── q1b_class_distribution.png
    ├── q1c_corr_*.png  (×5)
    ├── q1e_boxplots.png
    ├── q2a_pca_variance.png
    ├── q2b_pca_scatter.png
    ├── q2c_factor_loadings.png
    ├── q3_cm_*.png  (×3) + q3_comparison.png
    ├── q4_cm_*.png  (×2) + q4_comparison.png
    ├── q5_crossval_comparison.png
    ├── q6a_shap_global_bar.png
    ├── q6a_shap_beeswarm.png
    ├── q6b_shap_waterfall_sample1.png
    ├── q6c_shap_waterfall_sample2.png
    └── run_log.txt
```

---

## How to Run

### Local Execution
```bash
# Activate conda base environment (has all dependencies)
conda run -n base python main.py

# Or with your active environment (requires: pandas, numpy, matplotlib,
# seaborn, scikit-learn, scipy, torch, shap)
python main.py
```

### Docker Execution
You can containerize and run the entire machine learning pipeline inside a Docker container:
```bash
# Build the Docker image
docker build -t wearable-classification .

# Run the container (bind-mounting the outputs directory to persist results)
docker run --rm -v "$(pwd)/outputs:/app/outputs" wearable-classification
```

All figures are saved to `./outputs/`. The full console log is saved to `outputs/run_log.txt`.

---

## Dataset Description

| Property | Value |
|---|---|
| Samples | 241 |
| Numerical features | 40 (5 modalities × 8 statistics) |
| Sensor modalities | Accelerometer, BVP, EDA, Temperature, RR Interval |
| Categorical columns | gender, posture_type, yoga_experience, difficulty_level, subject |
| Target classes | relaxed (39), parasympathetic (125), sympathetic (77) |
| Feature type | Statistical features per 60-second window |

---

## Part 1: Exploratory Data Analysis [30 pts]

**File:** [`question1_eda.py`](question1_eda.py)

### 1a — Data Loading, Missing Values & Encoding

- **Shape:** 241 rows × 47 columns (after dropping `timestamp`, `subject_state`)
- **Missing values:** ~15% of `sample_entropy` columns were missing (consistent with motion artifacts during dynamic postures). Strategy: **median imputation** (robust to outliers).
- **Categorical encoding:** `LabelEncoder` applied to: `gender`, `posture_type`, `yoga_experience`, `difficulty_level`, `state`, `subject`

| Column | Encoding |
|---|---|
| gender | female=0, male=1 |
| posture_type | baseline=0, postural=1 |
| state | parasympathetic=0, relaxed=1, sympathetic=2 |
| yoga_experience | one_or_less=0, two_or_more=1 |
| difficulty_level | one=0, two=1, three=2 |

### 1b — Class Distribution

```
State           Count   Proportion
parasympathetic   125      51.9%
sympathetic        77      31.9%
relaxed            39      16.2%
```

> **Observation:** The dataset is **imbalanced** — parasympathetic dominates at ~52%. This justifies stratified splits in all downstream train/test divisions and cross-validation.

### 1c — Per-Modality Correlation Heatmaps

Five separate heatmaps were computed (one per sensor modality):

| Modality | Highly Correlated Pairs (|r| > 0.85) | Interpretation |
|---|---|---|
| **Accelerometer** | `std_dev` ↔ `peak_to_peak` (r≈0.99), `rms` ↔ `mean` (r≈0.99) | Motion energy features are nearly redundant |
| **BVP** | `mean` ↔ `rms` (r≈0.99), `num_peaks` ↔ `sample_entropy` (r≈−0.92) | Cardiovascular mean and RMS are collinear |
| **EDA** | `mean` ↔ `rms` (r≈0.99), `std_dev` ↔ `peak_to_peak` (r≈0.98) | Sweat response features are highly redundant |
| **Temperature** | `peak_to_rms` ↔ `peak_to_peak` (r≈0.99) | Thermal variability measures overlap |
| **RR Interval** | `mean` ↔ `rms` (r≈1.00), `std_dev` ↔ `peak_to_peak` (r≈0.99) | Heart rate timing features are nearly identical |

> **Does separation help?** Yes. Computing heatmaps separately clarifies that redundancy is primarily *within* modalities (e.g., mean vs. RMS of the same signal), not *across* them. This guides which features to prioritize and validates that PCA within each modality group would be effective.

### 1d — Descriptive Statistics by Class & Modality

Between-class variation scores (std of class means across features):

| Modality | Variation Score | Rank |
|---|---|---|
| **RR Interval** | **36.50** | **1st** |
| Temperature | 0.296 | 2nd |
| EDA | 0.257 | 3rd |
| Accelerometer | 0.133 | 4th |
| BVP | 0.056 | 5th |

> **→ RR Interval exhibits the greatest between-class variation**, suggesting that heart rate rhythm is the primary physiological discriminator between relaxed, parasympathetic, and sympathetic states during yoga. This aligns with the known role of the autonomic nervous system in modulating heart rate.

### 1e — Box Plots by Target Class

Representative features per modality (mean value, grouped by state):

| Modality | Feature | Class Separation |
|---|---|---|
| Accelerometer | `magnitude_mean` | Minimal — similar distributions across states |
| BVP | `absorption_mean` | Moderate — slight BVP elevation in sympathetic |
| EDA | `microsiemens_mean` | Moderate — EDA elevated in sympathetic state |
| Temperature | `celcius_mean` | Small — temperature changes slowly, less reactive |
| **RR Interval** | `milliseconds_mean` | **Strong** — relaxed has highest RR (lowest HR), sympathetic lowest |

> **Observation:** The RR interval shows the clearest separation: *relaxed > parasympathetic > sympathetic* in terms of RR mean (longer RR = slower heart rate = more relaxed). EDA shows moderate class-wise trends. Accelerometer and temperature show little class separation, likely because yoga postures affect all states similarly.

---

## Part 2: Principal Component Analysis [30 pts]

**File:** [`question2_pca.py`](question2_pca.py)

### 2a — Standardize & Full PCA

- **Scaled feature matrix shape (train):** `(192, 40)`
- Applied `StandardScaler` (PCA is scale-sensitive).
- Full PCA with all 40 components computed.

**Number of components for ≥90% explained variance: 15**

The scree plot shows a rapid initial drop (first 3–5 PCs capture most variance), followed by a long tail — classic in physiological data where a few dominant modes explain most variation.

### 2b — Reduced PCA (15 components)

| | Shape |
|---|---|
| Before PCA | X_train=(192, 40), X_test=(49, 40) |
| After PCA | X_train=(192, **15**), X_test=(49, **15**) |
| Variance retained | **90.01%** |

**PC1 vs PC2 Scatter Plot:** Partial class separation visible — the *relaxed* class clusters somewhat distinctly from *sympathetic* in the PC1-PC2 space. *Parasympathetic* overlaps with both. This suggests non-linear boundaries will outperform linear classifiers.

### 2c — Factor Loadings (Top 5 per PC)

**PC1** — Primarily captures **accelerometer** and **RR interval** variability:

| Feature | Loading |
|---|---|
| accelerometer_magnitude_peak_to_peak | 0.8817 |
| accelerometer_magnitude_std_dev | 0.8789 |
| accelerometer_magnitude_peak_to_rms | 0.8743 |
| rr_interval_milliseconds_peak_to_peak | 0.8295 |
| rr_interval_milliseconds_peak_to_rms | 0.7889 |

**PC2** — Dominated by **BVP** and **temperature** peak-count features:

| Feature | Loading |
|---|---|
| bvp_light_absorption_nW_num_peaks | 0.8340 |
| bvp_light_absorption_nW_sample_entropy | 0.8235 |
| bvp_light_absorption_nW_peak_to_rms | 0.7887 |
| temperature_celcius_num_peaks | 0.6321 |
| accelerometer_magnitude_num_peaks | 0.6112 |

**PC3** — Dominated by **temperature** variability features:

| Feature | Loading |
|---|---|
| temperature_celcius_peak_to_rms | 0.8122 |
| temperature_celcius_peak_to_peak | 0.7278 |
| temperature_celcius_std_dev | 0.6865 |
| bvp_light_absorption_nW_mean_derivative | 0.6642 |
| temperature_celcius_mean_derivative | −0.4323 |

> **Connection to Part 1:** Accelerometer and RR interval dominate PC1, consistent with the fact that RR interval showed the greatest between-class variation (Part 1d). SHAP analysis (Part 6) later confirms that PC1 is the most influential component for classification.

---

## Part 3: Logistic Regression [25 pts]

**File:** [`question3_logistic_regression.py`](question3_logistic_regression.py)

All models trained on the 15-component PCA-reduced dataset (80/20 stratified split).

### Results

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| **3a: No Regularization** | 0.7551 | 0.7374 |
| **3b: L1 (Lasso)** | 0.7551 | 0.7439 |
| **3c: L2 (Ridge)** | 0.7551 | 0.7439 |

### Per-Class Performance (L2 Ridge — representative)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| parasympathetic | 0.77 | 0.92 | 0.84 |
| relaxed | 0.86 | 0.75 | 0.80 |
| sympathetic | 0.67 | 0.50 | 0.57 |

### 3d — Regularization Impact Discussion

All three LR variants achieve **identical accuracy (75.5%)**, which is expected because:

1. **PCA decorrelates features** — the primary motivation for regularization (handling multicollinearity) is already addressed by PCA. All 15 components are orthogonal.
2. **Small dataset (n=241)** — with only 192 training samples and 15 features (good n/p ratio), the unregularized model doesn't substantially overfit.
3. **L1 vs L2 for PCA data:** L1 (Lasso) promotes sparse solutions by zeroing some PC coefficients. On PCA-reduced data, all components were retained by design to explain ≥90% variance, so aggressive L1 pruning is counter-productive. L2 shrinks coefficients smoothly without zeroing any, making it slightly preferable for this setting.

**Key takeaway:** The *sympathetic* class consistently achieves the lowest recall (0.50) across all LR variants, suggesting linear boundaries are insufficient — the sympathetic state overlaps with parasympathetic in the 15-D PCA space.

### 3e — Data Split Assumptions & Subject Leakage

**Assumptions made:**
1. **Stratified 80/20 random split** — preserves class proportions (critical given imbalance).
2. **i.i.d. assumption** — samples treated as independent. This is **violated** because each subject contributes multiple consecutive 60-second windows.

**Subject Leakage Problem:**  
If subject A appears in both train and test, the model memorises their physiological baseline rather than learning general state patterns. This inflates test accuracy — the model "recognises the person," not the state.

**Solution:**  
Use `GroupShuffleSplit` or `LeaveOneGroupOut` with `groups=subject_id` to ensure complete subject-level separation between train and test. For cross-validation: use `GroupKFold`.

---

## Part 4: Neural Networks [25 pts]

**File:** [`question4_neural_networks.py`](question4_neural_networks.py)

Both architectures use:
- **Optimizer:** Adam
- **Loss:** CrossEntropyLoss
- **Epochs:** 100 | **Batch size:** 16

### Architecture 1: Simple Feedforward Network

```
Input (15) → Linear(128) → ReLU → Linear(64) → ReLU → Linear(3)
Parameters: 10,499
Learning rate: 1e-3
```

### Architecture 2: Deep Feedforward Network

```
Input (15) → Linear(256) → BatchNorm → ReLU → Dropout(0.3)
           → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
           → Linear(64)  → BatchNorm → ReLU → Dropout(0.15)
           → Linear(32)  → ReLU
           → Linear(3)
Parameters: 48,323
Learning rate: 5e-4
```

### 4b — Performance Comparison

| Metric | Simple FFN | Deep FFN |
|---|---|---|
| **Accuracy** | 0.8163 | **0.8571** |
| **F1 (weighted)** | 0.8177 | **0.8566** |
| Precision | 0.8316 | 0.8753 |
| Recall | 0.8163 | 0.8571 |
| Parameters | 10,499 | 48,323 |

### Deep FFN Per-Class Performance

| Class | Precision | Recall | F1 |
|---|---|---|---|
| parasympathetic | 0.88 | 0.92 | 0.90 |
| relaxed | 1.00 | 0.75 | 0.86 |
| **sympathetic** | **0.83** | **0.94** | **0.88** |

> **Key observations:**
> - The **Deep FFN significantly outperforms Logistic Regression** (85.7% vs 75.5% accuracy), confirming non-linear decision boundaries are necessary.
> - The Deep FFN also outperforms the Simple FFN, indicating that additional depth + Dropout regularization helps generalize.
> - **Most importantly:** the sympathetic recall improves from 50% (LR) to 94% (Deep FFN) — the hardest class to classify is now well-captured by the deeper model.
> - BatchNorm layers stabilize training, and Dropout prevents overfitting despite the small dataset.

---

## Part 5: Cross-Validation Folds [20 pts]

**File:** [`question5_cross_validation.py`](question5_cross_validation.py)

Cross-validation performed on **Logistic Regression (L2)** using the full 241-sample PCA-reduced dataset.

### 5a — K-Fold with k=5 and k=10

| Configuration | Accuracy (mean ± std) | F1 (mean ± std) |
|---|---|---|
| KFold k=5 | 0.8134 ± 0.0289 | 0.8108 ± 0.0357 |
| KFold k=10 | 0.8215 ± 0.0592 | 0.8216 ± 0.0606 |

**Observations:**
- k=10 yields a slightly higher mean accuracy but **higher variance** (std nearly doubles from 0.029 to 0.059).
- With only 241 samples, k=10 creates very small test folds (~24 samples) where minority classes may be missing entirely, destabilizing estimates.
- **k=5 offers the better bias-variance tradeoff** for this dataset size.

### 5b — Stratified vs Non-Stratified

| Configuration | Acc Mean | Acc Std | F1 Mean | F1 Std |
|---|---|---|---|---|
| KFold k=5 (non-strat) | 0.8134 | 0.0289 | 0.8108 | 0.0357 |
| KFold k=10 (non-strat) | 0.8215 | 0.0592 | 0.8216 | 0.0606 |
| StratifiedKFold k=5 | 0.7798 | 0.0512 | 0.7792 | 0.0511 |
| **StratifiedKFold k=10** | **0.8293** | 0.0924 | **0.8233** | 0.0985 |

**Impact of stratification:**
- Stratification ensures **class proportions are preserved** in every fold — critical for imbalanced data (relaxed class: only 39 samples).
- Without stratification, some folds may have all relaxed samples in training, producing artificially high test accuracy on only 2 classes.
- The **StratifiedKFold k=10** achieves the highest mean accuracy (82.9%), though with higher variance — a consequence of small test folds combined with imbalanced classes.
- **Recommendation:** Use `StratifiedKFold k=5` for most robust performance estimates with this dataset.

---

## Part 6: SHAP Analysis [20 pts]

**File:** [`question6_shap.py`](question6_shap.py)

**Model explained:** Logistic Regression L2 (best LR model, used as best linear classifier).  
**Explainer:** `shap.LinearExplainer` with `Independent` masker.  
**SHAP values shape:** `(49 samples, 15 features, 3 classes)`.

### 6a — Global Feature Importance

**Top 5 PCA components by mean |SHAP| (averaged across all classes and test samples):**

| PC | Mean |SHAP| | Primary sensor modality |
|---|---|---|
| PC1 | ~0.40 | Accelerometer + RR Interval |
| PC2 | ~0.25 | BVP + Temperature peaks |
| PC3 | ~0.15 | Temperature variability |
| PC4 | ~0.10 | Mixed |
| PC5 | ~0.07 | Mixed |

> **Discussion:** PC1 dominates — it captures accelerometer and RR interval variability (see factor loadings in Part 2c). This confirms that **heart rhythm (RR interval) and body motion (accelerometer) are the primary drivers of state classification**, consistent with Part 1d where RR interval showed the highest between-class variation score (36.50 vs 0.30 for others). The SHAP analysis thus aligns with and validates the statistical observations from EDA.

> **Beeswarm plot:** Components that show consistent red/blue separation across samples indicate monotonic relationships with state. PC1 shows consistent direction — high PC1 values (driven by accelerometer motion variability) push predictions toward *sympathetic*, while low values favor *relaxed*. Some components push predictions consistently in one direction, suggesting dominant physiological patterns.

### 6b — Waterfall Plot: Test Sample 1

**Sample 1:**
- True class: `parasympathetic` | Predicted: `parasympathetic` ✓
- Top contributing PCs: PC1 (pushes away from relaxed), PC2 (pushes toward parasympathetic)

> This sample's physiological profile shows moderate RR variability (PC1) and elevated BVP activity (PC2), characteristic of a parasympathetic state. The waterfall shows how each PC shifts the prediction from the baseline, culminating in a correct classification.

### 6c — Waterfall Plot: Test Sample 2

**Sample 2:** (different true class from sample 1)
- True class: `sympathetic` | Predicted: `sympathetic` ✓
- Top contributing PCs: PC1 (strong push toward sympathetic — high motion/RR variability), PC3 (temperature component)

> Comparing samples 1 and 2 reveals **inter-subject variability** in how PCs contribute. Sample 2 shows stronger PC1 contribution, consistent with the sympathetic state involving more physiological arousal (shorter RR intervals, higher motion) compared to parasympathetic. This highlights the need for personalized modeling in real-world wearable applications.

---

## Part 7: Bonus Question [5 pts]

**File:** [`question7_bonus.py`](question7_bonus.py)

### If you had access to the full raw sensor streams, what would you explore next?

**A. Different Task: Temporal State Transition Detection**  
Rather than classifying a static state per 60-second window, I would apply **Bayesian change-point detection** (PELT algorithm) on EDA and RR-interval streams to identify the exact moment of ANS state transitions. Knowing *when* a state change occurs is more clinically actionable than classifying *what* state exists in predefined windows.

**B. Different Modeling Approach: Temporal Deep Learning**  
Replace handcrafted features + PCA with end-to-end temporal models:
- **1D-CNN:** Learns local temporal patterns directly from raw waveforms
- **LSTM / Transformer:** Captures long-range dependencies in physiological responses
- **Multi-Modal Attention Networks:** Dynamically weight sensor modalities based on context

**C. Question This Dataset Cannot Answer: Personalization**  
Physiological signals are highly individual. A resting HR of 55 bpm is normal for an athlete but abnormal for others. With raw streams, I would build:
- **Per-subject baseline estimation** for relative state classification
- **Few-shot transfer learning** for new subjects with minimal calibration data
- **Yoga experience as a moderator:** Do experienced practitioners show more controlled ANS responses?

**D. Signal Quality & Artifact Detection**  
The ~15% missing sample_entropy values suggest motion artifacts. With raw streams, I would build an automated **Signal Quality Index (SQI)** pipeline to detect and interpolate corrupted windows before feature extraction.

**Summary:** The most impactful next step is a *subject-adaptive temporal model* that takes raw multi-channel streams as input, estimates per-subject baselines in real-time, detects state transitions continuously, and is suitable for clinical or wellness wearable deployment.

---

## Final Model Performance Summary

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| LR — No Regularization | 0.7551 | 0.7374 |
| LR — L1 (Lasso) | 0.7551 | 0.7439 |
| LR — L2 (Ridge) | 0.7551 | 0.7439 |
| Neural Net — Simple FFN | 0.8163 | 0.8177 |
| **Neural Net — Deep FFN** ⭐ | **0.8571** | **0.8566** |

**★ Best model: Deep Feedforward Neural Network (accuracy=85.7%)**

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
torch (PyTorch)
shap
```

Install with:
```bash
conda run -n base pip install pandas numpy matplotlib seaborn scikit-learn scipy torch shap
```

---

## Key Findings

1. **RR Interval** is the most discriminative modality (Part 1d, 2c, 6a) — heart rate rhythm reflects ANS state most directly.
2. **15 PCA components** retain 90% of variance, reducing noise and dimensionality from 40 features.
3. **Neural networks outperform logistic regression** substantially (85.7% vs 75.5%) — physiological states require non-linear decision boundaries.
4. **Subject leakage** is the primary threat to generalization — future work should use subject-level train/test splits.
5. **Stratified cross-validation** is essential for this imbalanced dataset.
6. **SHAP confirms EDA findings** — the model relies on the same modalities (RR interval, accelerometer) that showed the strongest statistical class variation.
