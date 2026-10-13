import numpy as np
from sklearn.model_selection import KFold

# One fixed, shuffled 5-fold split so every model is scored on exactly the same folds
CV = KFold(n_splits=5, shuffle=True, random_state=42)

def splits_without_outliers(X, outlier_mask, cv=CV):
    # Same folds as CV, but outliers are removed from the training part of each fold only.
    # Every validation fold still contains all houses, so scores stay comparable with models
    # trained on all rows (removing outliers from validation too would make every model look better).
    outlier_mask = np.asarray(outlier_mask)
    return [(train_idx[~outlier_mask[train_idx]], val_idx) for train_idx, val_idx in cv.split(X)]
