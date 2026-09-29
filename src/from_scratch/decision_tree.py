import numpy as np

def gini_index(y):
    if len(y) == 0:
        return 0.0
    _,counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p ** 2)


def best_split(x, y):
    n_samples, n_features = x.shape
    if n_samples == 0:
        return None, None

    parent_gini = gini_index(y)
    best_gain = -1.0
    best_feature = None
    best_threshold = None

    for feature_idx in range(n_features):
        x_column = x[:, feature_idx]

        sort_idx = np.argsort(x_column, kind="mergesort")
        x_sorted = x_column[sort_idx]
        y_sorted = y[sort_idx]

        diff_idx = np.where(np.diff(x_sorted) != 0)[0]
        if len(diff_idx) == 0:
            continue  
        n_left = 0
        left_counts = {}
        right_counts = {}
        for label in y_sorted:
            right_counts[label] = right_counts.get(label, 0) + 1
        n_right = n_samples

        for i in range(n_samples - 1):
            label = y_sorted[i]
            right_counts[label] -= 1
            n_right -= 1
            left_counts[label] = left_counts.get(label, 0) + 1
            n_left += 1

            if x_sorted[i] == x_sorted[i + 1]:
                continue

            threshold = (x_sorted[i] + x_sorted[i + 1]) / 2.0

            gini_left = 1.0 - sum((c / n_left) ** 2 for c in left_counts.values() if c > 0)
            gini_right = 1.0 - sum((c / n_right) ** 2 for c in right_counts.values() if c > 0)

            weighted = (n_left / n_samples) * gini_left + (n_right / n_samples) * gini_right
            gain = parent_gini - weighted

            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold

    return best_feature, best_threshold