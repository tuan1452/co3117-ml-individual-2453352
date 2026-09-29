# PRE-RELEASE CATCHUP (W01–W04)

**Scope**: W01–W04 Catch-up Synthesis | **Date**: September 26, 2026

---

## A. Concept Capsule
This document synthesizes foundational machine learning concepts and Decision Trees from Weeks 1 to 4.

1. **Machine Learning Foundations & Trade-offs**:
   - **Taxonomy**: Supervised Learning (Classification, Regression) and Unsupervised Learning (Clustering, Dimensionality Reduction via PCA).
   - **8-Step Pipeline**: Problem Definition $\rightarrow$ Data Collection $\rightarrow$ Preprocessing $\rightarrow$ Feature Engineering $\rightarrow$ Model Selection/Training $\rightarrow$ Evaluation $\rightarrow$ Deployment $\rightarrow$ Monitoring.
   - **Data Principles**: Independent and identically distributed assumption. Preprocessing parameters must fit exclusively on training data to prevent Data Leakage.
   - **Overfitting vs. Underfitting**: Underfitting occurs when a simple model fails to capture underlying patterns (high bias). Overfitting happens when a complex model memorizes training noise, failing to generalize to unseen data (high variance).

2. **Decision Tree Algorithms (ID3, C4.5, CART)**:
   - **ID3**: Uses Entropy $H(S) = -\sum p_i \log_2 p_i$ and Information Gain $IG(S, A) = H(S) - \sum \frac{|S_v|}{|S|} H(S_v)$ for multiway categorical splits. Prone to overfitting and biased toward features with many values.
   - **C4.5**: Extends ID3 using Gain Ratio $GR(S, A) = \frac{IG(S, A)}{SplitInfo(S, A)}$ to normalize multi-valued feature bias. Handles continuous attributes via thresholding and applies pessimistic post-pruning.
   - **CART**: Constructs binary trees for classification (Gini Impurity $G(S) = 1 - \sum p_i^2$) and regression (Variance/SSE reduction). Uses cost-complexity pruning $R_\alpha(T) = R(T) + \alpha|T|$ and surrogate splits for missing values.

---

## B. One Derivation: Bias-Variance Decomposition
Consider regression $y = f(x) + \epsilon$ where $E[\epsilon] = 0$ and $Var(\epsilon) = \sigma^2$. The expected mean squared error at query point $x$ for estimate $\hat{f}(x)$ is:
$$E[(y - \hat{f}(x))^2] = E[(f(x) + \epsilon - \hat{f}(x))^2]$$

Because noise $\epsilon$ is independent of the model estimate, the cross-term vanishes:
$$= E[(f(x) - \hat{f}(x))^2] + \sigma^2$$

Adding and subtracting expected prediction $\mu(x) = E[\hat{f}(x)]$ yields:
$$= E[\{(f(x) - E[\hat{f}(x)]) + (E[\hat{f}(x)] - \hat{f}(x))\}^2] + \sigma^2$$

Expanding the square results in the canonical trade-off:
$$E[(y - \hat{f}(x))^2] = \text{Bias}(\hat{f}(x))^2 + \text{Var}(\hat{f}(x)) + \sigma^2$$

$\text{Bias}^2$: $(E[\hat{f}(x)] - f(x))^2$ measures error from overly simplistic assumptions (Underfitting).
Variance: $E[(\hat{f}(x) - E[\hat{f}(x)])^2]$ measures sensitivity to training data fluctuations (Overfitting).
Irreducible Noise: $\sigma^2$ is the inherent variance in data generation.

---

## C. Code-to-Theory Trace
Mapping theory to `src/from_scratch/decision_tree.py`:
1. **Gini Impurity (CART)**:
   - Math: $G(S) = 1 - \sum p_i^2$
   - Code: counts = np.unique(y, return_counts=True); p = counts / len(y)
2. **Best Binary Split Evaluation**:
    - Math formula: 
      $$
      \arg\max_{j,t} \Delta G(S, j, t) = G(S) - \left( \frac{|S_L|}{|S|} G(S_L) + \frac{|S_R|}{|S|} G(S_R) \right)
      $$
   - Code Path (`best_split`): Each feature column is sorted using sort_idx = np.argsort(x_column, kind="mergesort"). Midpoint split candidates are computed as threshold = (x_sorted[i] + x_sorted[i + 1]) / 2.0. Class counts are updated dynamically using left_counts and right_counts dictionaries, computing weighted = (n_left / n_samples) * gini_left + (n_right / n_samples) * gini_right to maximize best_gain.
   **Reference Dissection & Non-Trivial Modification (Depth B)**:
   - **Reference Trace (`eriklindermoren/ML-From-Scratch`)**: In `decision_tree_model.py`, `_build_tree()` repeatedly creates sub-arrays via `divide_on_feature()` inside nested loops for each threshold candidate, resulting in \(O(N^2)\) complexity per continuous feature.
   - **My Modification**: I pre-sort continuous features once with `np.argsort` and update class frequency histograms (`left_counts`, `right_counts`) incrementally in a single pass (\(O(N)\)). This algorithmic optimization enables feasible training on the 561-dimensional UCI HAR dataset without memory/runtime bottlenecks.

---

## D. One Controlled Experiment: Impact of Tree Depth & Scikit-Learn Benchmark
Investigating the hyperparameter `max_depth` across \(\{1, 2, 3, 5, 8\}\) on the UCI HAR dataset (7,352 train samples, 2,947 test samples, 561 features):

- **Baseline Floor Performance (R0)**:
  - `DummyClassifier(strategy="most_frequent")` yields Test Macro-F1 = **0.0514** and Test Accuracy = **18.22%**.
- **Underfitting Phase (`max_depth = 1–2`)**:
  - `depth = 1`: From-Scratch yields Test Macro-F1 = **0.2269**, Test Accuracy = **36.27%** (Fit: 48.49s). Scikit-Learn matches identically with Test Macro-F1 = **0.2269**, Test Accuracy = **36.27%** (Fit: 1.31s).
  - `depth = 2`: From-Scratch yields Test Macro-F1 = **0.3685**, Test Accuracy = **53.10%** (Fit: 77.76s), also identical to Scikit-Learn.
- **Effective Learning Phase (`max_depth = 3–5`)**:
  - `depth = 3`: Test Macro-F1 surges to **0.6680** (Test Accuracy = **72.24%**).
  - `depth = 5`: Test Macro-F1 reaches **0.8322** (Train Macro-F1 = **0.9164**, Test Accuracy = **83.85%**).
- **Overfitting Onset Phase (`max_depth = 8`)**:
  - Training performance nears memorization at Train Macro-F1 = **0.9727** and Train Accuracy = **97.33%**, whereas generalization plateaus at Test Macro-F1 = **0.8694** and Test Accuracy = **87.28%** (Fit: 248.00s).
  - Scikit-Learn yields equivalent generalization with Test Macro-F1 = **0.8646** and Test Accuracy = **86.83%** (Fit: 9.58s).
- **Runtime & Efficiency Analysis**:
  - Cython-optimized Scikit-Learn fits approximately **26x faster** than pure interpreted Python at `max_depth = 8` (9.58s vs. 248.00s).
  - Both implementations achieve almost identical decision boundaries and empirical convergence (test metric divergence (less than 0.4\%)).
- **Results table**

| Depth | Scratch Test F1 | Sklearn Test F1 | Scratch Acc | Sklearn Acc | Scratch Fit (s) | Sklearn Fit (s) |
|-------|-----------------|-----------------|-------------|-------------|-----------------|-----------------|
| 1     | 0.2269          | 0.2269          | 0.3627      | 0.3627      | 48.49           | 1.3119          |
| 2     | 0.3685          | 0.3685          | 0.5310      | 0.5310      | 77.76           | 2.3520          |
| 3     | 0.6680          | 0.6680          | 0.7224      | 0.7224      | 98.57           | 3.3489          |
| 5     | 0.8322          | 0.8319          | 0.8385      | 0.8381      | 151.46          | 6.0773          |
| 8     | 0.8694          | 0.8646          | 0.8728      | 0.8683      | 248.00          | 9.5760          |
---

## E. Failure / Misconception
1. **Empirical Confusion Failure Mode (`results/confusion_matrix_scratch.txt`)**:
   - **Dynamic vs. Static Distinction**: The model achieves zero cross-confusion between dynamic activities (*Walking, Walking Upstairs, Walking Downstairs*) and static postures (*Sitting, Standing, Laying*), reaching 100% accuracy on the *Laying* class (537/537 samples).
   - **Systematic Error**: Severe confusion occurs between **Sitting (Class 4)** and **Standing (Class 5)**: 120 sitting instances are misclassified as standing, and 57 standing instances are misclassified as sitting[cite: 15]. Because the static gravitational acceleration vectors captured by the smartphone accelerometer are nearly indistinguishable between these two upright postures, the axis-aligned orthogonal splits of a Decision Tree struggle to separate them without combined tilt/angle interaction features.

2. **Methodological Misconception**:
   - **Misconception**: Decision Trees are completely immune to data leakage because monotonic feature transformations do not alter split thresholds.
   - **Reality**: Applying global preprocessing (such as normalization, scaling, or feature selection) across the entire dataset before performing subject-aware splitting causes entity leakage, artificially inflating validation scores while degrading true generalization to unseen subjects.

---

## F. Written-Exam Capsule
Overfitting occurs when a model captures training noise alongside true patterns, yielding high training accuracy but poor generalization to unseen test data due to high variance. Underfitting happens when a model lacks capacity to capture underlying data structures, resulting in high error across both training and validation sets due to high bias. ID3 uses Information Gain for multiway splits but overfits easily, C4.5 improves ID3 via Gain Ratio and pessimistic pruning; CART uses binary splits with Gini Impurity and cost-complexity pruning to control tree complexity and prevent overfitting.

---

## G. Reflection
- **Mastered Concepts**: Differences between ID3, C4.5, and CART algorithms, and the mathematical origin of the Bias-Variance Decomposition.
- **Open Questions**: Trade-offs between C4.5 fractional weighting and CART surrogate splits for missing data.
- **Next Steps**: Implement the Perceptron linear classifier in Week 5 and compare its linear decision boundary with tree axis-aligned splits.

---

## H. Inquiry Trail (AI-Assisted Protocol)

1. **Prompt**: "In CART, how are continuous features converted into binary 
   splits without evaluating infinite thresholds? Give me the conceptual 
   steps rather than code."
2. **AI Guidance**: AI outlined the 3-step mechanism — (i) sort unique 
   feature values in ascending order, (ii) evaluate midpoints between 
   adjacent distinct values $t = (v_i + v_{i+1})/2$, (iii) select the 
   threshold maximizing Gini reduction $\Delta G(s, t)$.

3. **Verification**: 
   - CO3117 lecture notes, *Decision Trees*, **Slide 56** ("Handling 
     Continuous Attributes" under CART).
   - Cross-checked with **Slide 37–40** (C4.5 threshold selection with 
     Gain Ratio, candidate thresholds $\{64.5, 66.5, \dots, 87.5\}$).

4. **What changed**:
   - Misconception: Thought CART evaluated all real-valued thresholds 
     between consecutive feature values.
   - Correction: Any threshold strictly inside the gap $(v_i, v_{i+1})$ 
     yields the same partition, so only the midpoint is needed — 
     reducing candidates to $n-1$ per feature.