# Dataset Protocol: UCI Human Activity Recognition (HAR)
## 1. Dataset Source & Prediction Target
- **Dataset**: UCI Human Activity Recognition Using Smartphones (Version 1.0)
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
- **Sequential / Structured View (for HMM/CRF)**: Derive chronological sequences per subject using time continuity from raw sensor signals.
- **Leakage Prevention Rule**: All scalers, encoders, feature selectors, and dimensionality reduction methods (PCA/LDA) must be fit on training data only, then applied to transform validation/test sets.

## 3. Split Policy & Sealed Test Set
- **Splitting Strategy**: Group/Subject-aware split across 30 volunteers (70% train: 21 subjects, 30% test: 9 subjects) to prevent subject-level identity leakage.
- **Validation**: Cross-validation or dedicated validation split is performed inside the training population only.
- **Sealed Test Set**: The test set remains strictly sealed and is only evaluated for final benchmark comparison.

## 4. Evaluation Metrics
- **Primary Metric**: Macro-F1 score.
- **Secondary Metrics**: Multiclass Accuracy, Precision, Recall, and Confusion Matrix.

## 5. Experimental Reproducibility
- **Fixed Random Seed**: `2453352` across all experiments and model initializations.