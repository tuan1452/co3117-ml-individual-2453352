import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier as SklearnDT

# ---------------------------------------------------------
# 1. Cấu hình đường dẫn & nạp hàm best_split from-scratch
# ---------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

for path in [PROJECT_ROOT, CURRENT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

    from src.from_scratch.decision_tree import best_split

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "UCI HAR Dataset")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------
# 2. Định nghĩa cấu trúc cây From-Scratch
# ---------------------------------------------------------
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self):
        return self.value is not None


class ScratchDecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _majority_vote(self, y):
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]

    def _build_tree(self, X, y, depth=0):
        n_samples, _ = X.shape
        n_classes = len(np.unique(y))

        if (n_classes <= 1 or 
            n_samples < self.min_samples_split or 
            (self.max_depth is not None and depth >= self.max_depth)):
            return Node(value=self._majority_vote(y))

        feat_idx, thresh = best_split(X, y)
        if feat_idx is None:
            return Node(value=self._majority_vote(y))

        left_mask = X[:, feat_idx] <= thresh
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return Node(value=self._majority_vote(y))

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return Node(feature=feat_idx, threshold=thresh, left=left_child, right=right_child)

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _predict_one(self, x, node):
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])


# ---------------------------------------------------------
# 3. Hàm tiện ích & đo lường
# ---------------------------------------------------------
def load_split(split: str):
    X_path = os.path.join(DATA_DIR, split, f"X_{split}.txt")
    y_path = os.path.join(DATA_DIR, split, f"y_{split}.txt")
    if not os.path.exists(X_path):
        raise FileNotFoundError(f"Không tìm thấy file: {X_path}")
    X = np.loadtxt(X_path)
    y = np.loadtxt(y_path).astype(int).ravel()
    return X, y


def macro_f1(y_true, y_pred):
    classes = np.unique(y_true)
    f1s = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def confusion_matrix(y_true, y_pred, n_classes=6):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t - 1, p - 1] += 1
    return cm


def evaluate(clf, X, y):
    y_pred = clf.predict(X)
    acc = float(np.mean(y_pred == y))
    return acc, macro_f1(y, y_pred), y_pred


# ---------------------------------------------------------
# 4. Thực thi Benchmark đối chuẩn (Stage 5)
# ---------------------------------------------------------
def run_benchmark():
    print(f"-> Đang tải dữ liệu từ {DATA_DIR} ...")
    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")
    print(f"   Train: X={X_train.shape}, y={y_train.shape}")
    print(f"   Test : X={X_test.shape}, y={y_test.shape}")

    depth_candidates = [1, 2, 3, 5, 8]
    results = []

    header = (f"{'Depth':<6} | "
              f"{'Scratch Test F1':<15} | {'Sklearn Test F1':<15} | "
              f"{'Scratch Acc':<11} | {'Sklearn Acc':<11} | "
              f"{'Scratch Fit(s)':<14} | {'Sklearn Fit(s)':<14}")
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for depth in depth_candidates:
        # A. Huấn luyện & đánh giá Scratch Model
        clf_scratch = ScratchDecisionTreeClassifier(max_depth=depth)
        t0 = time.time()
        clf_scratch.fit(X_train, y_train)
        t_fit_scratch = time.time() - t0
        tr_acc_sc, tr_f1_sc, _ = evaluate(clf_scratch, X_train, y_train)
        te_acc_sc, te_f1_sc, y_pred_sc = evaluate(clf_scratch, X_test, y_test)

        # B. Huấn luyện & đánh giá Sklearn Benchmark
        clf_sklearn = SklearnDT(criterion="gini", max_depth=depth, random_state=42)
        t0 = time.time()
        clf_sklearn.fit(X_train, y_train)
        t_fit_sk = time.time() - t0
        tr_acc_sk, tr_f1_sk, _ = evaluate(clf_sklearn, X_train, y_train)
        te_acc_sk, te_f1_sk, y_pred_sk = evaluate(clf_sklearn, X_test, y_test)

        depth_str = str(depth)
        print(f"{depth_str:<6} | "
              f"{te_f1_sc:<15.4f} | {te_f1_sk:<15.4f} | "
              f"{te_acc_sc:<11.4f} | {te_acc_sk:<11.4f} | "
              f"{t_fit_scratch:<14.2f} | {t_fit_sk:<14.4f}")

        results.append({
            "max_depth": depth,
            # Scratch metrics
            "scratch_train_f1": round(tr_f1_sc, 4),
            "scratch_test_f1": round(te_f1_sc, 4),
            "scratch_train_acc": round(tr_acc_sc, 4),
            "scratch_test_acc": round(te_acc_sc, 4),
            "scratch_fit_time_sec": round(t_fit_scratch, 2),
            # Sklearn metrics
            "sklearn_train_f1": round(tr_f1_sk, 4),
            "sklearn_test_f1": round(te_f1_sk, 4),
            "sklearn_train_acc": round(tr_acc_sk, 4),
            "sklearn_test_acc": round(te_acc_sk, 4),
            "sklearn_fit_time_sec": round(t_fit_sk, 4),
        })

        # Lưu confusion matrix của test set ở độ sâu tốt nhất (depth=8)
        if depth == depth_candidates[-1]:
            cm_scratch = confusion_matrix(y_test, y_pred_sc)
            cm_sklearn = confusion_matrix(y_test, y_pred_sk)
            np.savetxt(os.path.join(RESULTS_DIR, "confusion_matrix_scratch.txt"), cm_scratch, fmt="%d")
            np.savetxt(os.path.join(RESULTS_DIR, "confusion_matrix_sklearn.txt"), cm_sklearn, fmt="%d")

    # Lưu metrics.csv
    df = pd.DataFrame(results)
    metrics_path = os.path.join(RESULTS_DIR, "metrics.csv")
    df.to_csv(metrics_path, index=False)
    print("=" * len(header))
    print(f"-> Đã lưu bảng kết quả chi tiết: {metrics_path}")

    # Vẽ đồ thị so sánh đối chứng Validation Curve
    plt.figure(figsize=(10, 5))
    plt.plot(df["max_depth"], df["scratch_test_f1"], 'o-', label="From-Scratch (Test F1)", color="blue", linewidth=2)
    plt.plot(df["max_depth"], df["sklearn_test_f1"], 's--', label="Scikit-Learn (Test F1)", color="red", linewidth=2)
    plt.plot(df["max_depth"], df["scratch_train_f1"], 'o:', label="From-Scratch (Train F1)", color="lightblue", alpha=0.7)
    plt.plot(df["max_depth"], df["sklearn_train_f1"], 's:', label="Scikit-Learn (Train F1)", color="lightcoral", alpha=0.7)

    plt.xlabel("Max Depth")
    plt.ylabel("Macro F1-Score")
    plt.title("Decision Tree: From-Scratch vs Scikit-Learn Benchmark on UCI HAR")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(FIGURES_DIR, "val_curve_dt_benchmark.png")
    plt.savefig(fig_path, dpi=150)
    print(f"-> Đã lưu biểu đồ đối chứng tại: {fig_path}")


if __name__ == "__main__":
    run_benchmark()