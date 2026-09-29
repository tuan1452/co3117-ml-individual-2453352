"""UCI HAR data loader module."""

import os
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "UCI HAR Dataset")
RANDOM_SEED = 2453352


def load_dataset() -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
                            Tuple[np.ndarray, np.ndarray, np.ndarray],
                            List[str]]:
    """Load train and test splits from UCI HAR dataset."""
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

    features_path = os.path.join(DATA_DIR, "features.txt")
    features = pd.read_csv(
        features_path, sep=r"\s+", header=None, names=["idx", "name"]
    )["name"].tolist()

    x_train = np.loadtxt(os.path.join(DATA_DIR, "train", "X_train.txt"))
    y_train = np.loadtxt(os.path.join(DATA_DIR, "train", "y_train.txt"), dtype=int)
    subject_train = np.loadtxt(os.path.join(DATA_DIR, "train", "subject_train.txt"), dtype=int)

    x_test = np.loadtxt(os.path.join(DATA_DIR, "test", "X_test.txt"))
    y_test = np.loadtxt(os.path.join(DATA_DIR, "test", "y_test.txt"), dtype=int)
    subject_test = np.loadtxt(os.path.join(DATA_DIR, "test", "subject_test.txt"), dtype=int)

    return (x_train, y_train, subject_train), (x_test, y_test, subject_test), features


def get_train_val_split(x_train: np.ndarray, 
                        y_train: np.ndarray, 
                        subject_train: np.ndarray, 
                        val_size: float = 0.2, 
                        seed: int = RANDOM_SEED) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Subject-aware Train/Validation split to prevent data leakage."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(gss.split(x_train, y_train, groups=subject_train))

    return x_train[train_idx], y_train[train_idx], x_train[val_idx], y_train[val_idx]


def get_activity_labels() -> dict:
    """Map class index to activity name."""
    return {
        1: "WALKING",
        2: "WALKING_UPSTAIRS",
        3: "WALKING_DOWNSTAIRS",
        4: "SITTING",
        5: "STANDING",
        6: "LAYING",
    }
