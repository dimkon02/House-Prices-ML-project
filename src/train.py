"""Train the final model and write the Kaggle submission.

Run from the project folder:
    python src/train.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from preprocessing import load_data, flag_outliers, full_pipeline, build_preprocessor
from evaluation import splits_without_outliers

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / 'data' / 'train.csv'
TEST_PATH = ROOT / 'data' / 'test.csv'
SUBMISSION_PATH = ROOT / 'submissions' / 'gb_tuned_submission.csv'

# Best settings found by the grid search in notebooks/02_modeling.ipynb
GB_PARAMS = {
    'n_estimators': 300,
    'max_depth': 3,
    'learning_rate': 0.1,
    'subsample': 0.9,
    'random_state': 42,
}

def build_model(X):
    return Pipeline(steps=[
        ('preprocessor', build_preprocessor(X)),
        ('model', GradientBoostingRegressor(**GB_PARAMS))
    ])

def main():
    # 1. Preprocess: imputation statistics are learned on train and reused on test
    X_train, y_train, stats = full_pipeline(TRAIN_PATH)
    X_test, _, _ = full_pipeline(TEST_PATH, stats=stats)
    is_outlier = flag_outliers(load_data(TRAIN_PATH))['is_outlier'].to_numpy()

    # 2. Cross-validation score (outliers left out of training folds only, every house is scored)
    scores = -cross_val_score(build_model(X_train), X_train, y_train,
                              cv=splits_without_outliers(X_train, is_outlier),
                              scoring='neg_root_mean_squared_error')
    print("CV RMSE per fold:", np.round(scores, 4))
    print(f"CV mean RMSE: {scores.mean():.4f}")

    # 3. Train on all houses except the 2 outliers, predict test, convert back from log1p to dollars
    model = build_model(X_train)
    model.fit(X_train[~is_outlier], y_train[~is_outlier])
    predictions = np.expm1(model.predict(X_test))

    # 4. Write the submission
    submission = pd.DataFrame({'Id': load_data(TEST_PATH)['Id'], 'SalePrice': predictions})
    SUBMISSION_PATH.parent.mkdir(exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved {len(submission)} predictions to {SUBMISSION_PATH.relative_to(ROOT)}")

if __name__ == '__main__':
    main()
