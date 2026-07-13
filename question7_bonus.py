"""
Question 7: Bonus Question [5 points]
=======================================
This module provides a thoughtful written answer to the bonus question:

  "If you had access to the full raw sensor streams from this study,
   what would you explore next — a different task, a different modeling
   approach, or a question this dataset could not answer?"
"""


def run_bonus() -> str:
    """
    Print and return the bonus question response.
    """
    bonus_text = """
================================================================================
PART 7 (BONUS): What Would I Explore Next with Raw Sensor Streams?
================================================================================

The current dataset consists of pre-computed statistical features extracted from
60-second windows of five wearable physiological signals. While this is a well-
designed feature engineering pipeline, working with the FULL RAW sensor streams
would unlock several exciting directions:

────────────────────────────────────────────────────────────────────────────────
A. DIFFERENT TASK: Temporal State Transition Detection
────────────────────────────────────────────────────────────────────────────────
Rather than classifying a static "state" over a 60-second window, with raw
streams I would explore CHANGE-POINT DETECTION — identifying the exact moment
when the autonomic nervous system transitions between states (e.g., from relaxed
to sympathetic activation during a challenging yoga pose). This is more clinically
actionable: knowing WHEN a state change occurs is often more valuable than
classifying what state is happening in a predefined window.

Method: Bayesian change-point detection (PELT algorithm) applied to the EDA and
RR-interval streams, which are particularly sensitive to ANS transitions.

────────────────────────────────────────────────────────────────────────────────
B. DIFFERENT MODELING APPROACH: Temporal Deep Learning
────────────────────────────────────────────────────────────────────────────────
With raw multi-channel time-series data, I would replace the handcrafted feature
extraction + PCA pipeline with:

  1. 1D Convolutional Neural Networks (1D-CNN): Learns local temporal patterns
     directly from raw signal windows, capturing features that hand-engineering
     might miss (e.g., specific waveform shapes in BVP/EDA).

  2. LSTM / Transformer-based models: Capture long-range temporal dependencies
     across signals — critical for physiological state classification because
     autonomic responses unfold over seconds to minutes.

  3. Multi-Modal Attention Networks: Use cross-modal attention to dynamically
     weight different sensor modalities (e.g., down-weighting accelerometer
     during stillness, up-weighting EDA during stress).

This removes the assumption that pre-defined statistical features (mean, std,
peak-to-peak) are sufficient representations — the model learns representations
that are optimal for the classification task.

────────────────────────────────────────────────────────────────────────────────
C. QUESTION THIS DATASET CANNOT ANSWER: Personalization & Physiological Baselines
────────────────────────────────────────────────────────────────────────────────
The current extracted-feature dataset treats all subjects as homogeneous — the
same model serves all subjects. But physiological signals are HIGHLY individual:
a resting heart rate of 55 bpm may be "normal" for an athlete but indicate
bradycardia in a sedentary person. The same EDA value means different things
for different individuals.

With raw streams and subject metadata, I would explore:

  • Personalized baseline estimation: Using the first N minutes of a session
    to estimate a subject-specific baseline, then classifying states relative
    to that individual baseline (z-scoring per-subject).

  • Transfer learning for new subjects: Pre-train a model on existing subjects,
    then fine-tune with just a few minutes of a new subject's data (few-shot
    personalization) — a critical requirement for real-world wearable deployment.

  • Yoga experience as a moderator: Do experienced practitioners show more
    controlled sympathetic responses (lower EDA peaks, more stable RR) during
    difficult postures? Raw streams would enable testing this hypothesis through
    time-series analysis of within-session physiological dynamics.

────────────────────────────────────────────────────────────────────────────────
D. SIGNAL QUALITY & ARTIFACT DETECTION
────────────────────────────────────────────────────────────────────────────────
The high proportion of missing sample_entropy values in the extracted features
hints at signal quality issues in the raw streams (motion artifacts, sensor
disconnections during movement-heavy postures). With raw streams, I would build
an automated signal quality index (SQI) pipeline to flag and interpolate
corrupted windows before feature extraction — making the classification pipeline
more robust for real-world deployment.

────────────────────────────────────────────────────────────────────────────────
Summary
────────────────────────────────────────────────────────────────────────────────
The most impactful next step would be a subject-adaptive temporal model that:
  1. Takes raw multi-channel streams as input
  2. Estimates per-subject baselines in real-time
  3. Detects state TRANSITIONS rather than classifying static windows
  4. Provides continuous physiological state monitoring suitable for
     wearable deployment in clinical or wellness settings

This moves from a research classification task to a practical, deployable
physiological monitoring system.
================================================================================
"""
    print(bonus_text)
    return bonus_text.strip()


if __name__ == '__main__':
    run_bonus()
