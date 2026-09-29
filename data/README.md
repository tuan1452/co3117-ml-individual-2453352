# Dataset Protocol: UCI Human Activity Recognition (HAR)
## 1. Dataset Source & Prediction Target
- **Dataset**: UCI Human Activity Recognition Using Smartphones (Version 1.0)
- **Freeze Date**: 2026-09-26 (Checkpoint R0)
- **Source URL**: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- **Use Case**: Multiclass physical activity classification from smartphone inertial sensor signals.
- **Target Classes (6 activities)**:
  1. WALKING
  2. WALKING_UPSTAIRS
  3. WALKING_DOWNSTAIRS
  4. SITTING
  5. STANDING
  6. LAYING

## 2. Representations & Preprocessing
- **Feature Representation**: 561-feature vector with normalized time and frequency domain variables (bounded within `[-1, 1]`).
- **Sequential / Structured View (for HMM / CRF Models)**:
  - Derived from raw inertial signals (3-axial linear acceleration and 3-axial angular velocity sampled at 50 Hz).
  - Fixed-width sliding windows of **128 readings (2.56 sec)** with **50% overlap (64 readings / 1.28 sec)**.
  - Chronological continuous sequences preserved strictly per subject to maintain Markovian transition dynamics.
- **Leakage Prevention Rule**: All scalers, encoders, feature selectors, and dimensionality reduction methods (PCA/LDA) must be fit on training data only, then applied to transform validation/test sets.

## 3. Split Policy & Sealed Test Set
- **Subject-Aware Splitting**:
  - 30 volunteers partitioned into disjoint groups to prevent identity leakage:
  - **Training Population (21 subjects)**: Subjects `[1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]` (7,352 samples).
  - **Held-out Test Population (9 subjects)**: Subjects `[2, 4, 9, 10, 12, 13, 18, 20, 24]` (2,947 samples).
- **Validation Protocol**:
  - Hyperparameter tuning and model diagnostics use **5-fold GroupKFold (`GroupKFold(n_splits=5)`)** strictly partitioned across the 21 training subjects(groups = subject_id).
- **Sealed Test Set**: The test set remains strictly sealed and is only evaluated for final benchmark comparison.

## 4. Evaluation Metrics
- **Primary Metric**: Macro-F1 score.
- **Secondary Metrics**: Multiclass Accuracy, Precision, Recall, and Confusion Matrix.

## 5. Experimental Reproducibility
- **Fixed Random Seed**: `2453352` across all experiments and model initializations.
## 6. Baseline Reference (Frozen at R0)
- Model: DummyClassifier(strategy="most_frequent")
- Evaluation Split: Held-out Test Set (2,947 samples)
- Baseline Macro-F1: 0.0514
- Baseline Accuracy: 0.1822
- Test set was evaluated ONCE for baseline only. It remains SEALED for all subsequent models until final benchmark. No hyperparameter tuning or model selection based on test set.
- Purpose: Empirical lower bound (floor performance) for all subsequent models.