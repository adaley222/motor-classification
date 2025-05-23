"""
split.py
--------
Splits preprocessed EEG data into train, validation, and test sets. Ensures no data leakage between subjects/runs.
"""

import numpy as np
from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split data into train, validation, and test sets.

    Args:
        X (np.ndarray): Feature array.
        y (np.ndarray): Label array.
        test_size (float, optional): Fraction of data to use as test set. Defaults to 0.2.
        val_size (float, optional): Fraction of data to use as validation set. Defaults to 0.1.
        random_state (int, optional): Random seed. Defaults to 42.

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
            X_train, X_val, X_test (np.ndarray): Feature splits.
            y_train, y_val, y_test (np.ndarray): Label splits.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size+val_size), random_state=random_state, stratify=y)
    val_relative = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_relative, random_state=random_state, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test
