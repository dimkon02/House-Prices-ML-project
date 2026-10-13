# House Prices Prediction

Predicting house sale prices in Ames, Iowa from 79 property features, for the Kaggle competition
[House Prices – Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

**Final model:** tuned Gradient Boosting · **Kaggle score: 0.13520** (RMSE on log price) · **Cross-validation: 0.1253**

## Results

All models were evaluated with the same shuffled 5-fold cross-validation on `log1p(SalePrice)`
(lower RMSE is better). The two outlier houses were left out of training but every house was
still scored (see [Evaluating the outliers fairly](#evaluating-the-outliers-fairly)).

| Model | CV RMSE |
|---|---|
| KNN | 0.1761 |
| Linear Regression | 0.1500 |
| XGBoost (default settings) | 0.1439 |
| Random Forest | 0.1430 |
| LightGBM (default settings) | 0.1323 |
| Gradient Boosting (default settings) | 0.1301 |
| LightGBM (tuned) | 0.1294 |
| **Gradient Boosting (tuned)** | **0.1253** |

The tuned Gradient Boosting model (`n_estimators=300, max_depth=3, learning_rate=0.1, subsample=0.9`)
scored **0.13520** on the Kaggle public leaderboard, which means predictions are off by roughly 13–14%
on average. The leaderboard score is a little worse than the CV score, as expected: the grid search
used the same folds to pick the settings and to report the score, so the CV number is slightly optimistic.

## Approach

### 1. Exploratory data analysis
- **Target:** `SalePrice` is strongly right-skewed (skew 1.88). After `log1p` it is nearly
  symmetric (skew 0.12), so all models are trained on `log1p(SalePrice)` and predictions are
  converted back with `expm1`.
- **Missing values:** 19 columns have gaps. Checked each against the data description to separate
  "NA = the house has no such feature" (no pool, no garage, no basement) from genuinely missing values,
  and verified it with cross-checks (e.g. every house with no `PoolQC` has `PoolArea = 0`).
- **Numeric features:** strongest correlations with price are `OverallQual` (0.79), `GrLivArea` (0.71),
  `GarageCars` (0.64), `GarageArea` (0.62) and `TotalBsmtSF` (0.61).
- **Categorical features:** screened all 43 by the spread of average price between categories, then
  checked category counts to avoid false signals from tiny groups. `Neighborhood`, `ExterQual`,
  `KitchenQual` and `FireplaceQu` showed clear patterns; `RoofMatl` (98% one value), `Condition2`,
  `Exterior1st` and `Exterior2nd` were judged unreliable.
- **Outliers:** two houses with over 4,000 sq ft of living area sold for under $300,000. Both were
  partial sales of unfinished new builds, not normal market prices.

### 2. Cleaning
- "No feature" gaps filled with `'None'` (categorical) or `0` (numeric).
- `LotFrontage` filled with the median of the house's neighbourhood.
- Remaining gaps filled with the most common value.
- All fill values are learned from the training data only and reused unchanged on the test data,
  so the test set never influences its own preprocessing.

### 3. Feature engineering
- **Ordinal encoding** of 21 columns with a natural order (e.g. quality `Po < Fa < TA < Gd < Ex`),
  with mappings built from the values that actually occur in the data.
- **Combined features:** `TotalSF`, `TotalBath`, `TotalPorchSF`.
- **Age features:** `HouseAge`, `HouseRemodAge`, `GarageAge`.
- **Binary flags:** `HasGarage`, `HasBsmt`, `HasPool`, `IsNew`, `IsPartialSale` and others (9 in total).
- The 23 unordered categorical columns are one-hot encoded and the numeric columns are standardised,
  inside a scikit-learn `ColumnTransformer` + `Pipeline`, so the transformations are fitted on the
  training folds only during cross-validation. This gives 256 model inputs from 94 features.

### 4. Modelling
- Compared Linear Regression, KNN, Random Forest and Gradient Boosting, then XGBoost and LightGBM.
- Tuned the two best (Gradient Boosting and LightGBM) with a grid search over the number of trees,
  tree depth, learning rate and subsampling.
- The final model is trained on every house except the two outliers.

### Evaluating the outliers fairly
The first version of the outlier experiment removed the two outliers *before* cross-validation, so
they were missing from the validation folds as well. That made every model look much better
(Linear Regression 0.1547 → 0.1281), but mostly because the two hardest houses were no longer being scored.

Removing them only from the **training** part of each fold, while still scoring every house,
gives the real effect: a small but consistent improvement for all four models
(0.0003–0.0025 RMSE). Both outliers fall in validation fold 3, which is the worst fold for most
models, and Random Forest and Gradient Boosting score exactly the same on that fold with or without them.

## Project structure

```
├── notebooks/
│   ├── 01_exploartion.ipynb   # EDA, cleaning and feature engineering, step by step
│   └── 02_modeling.ipynb      # model comparison, outlier experiment, tuning, submission
├── src/
│   ├── preprocessing.py       # cleaning + feature engineering pipeline used by the models
│   ├── evaluation.py          # shared cross-validation folds
│   └── train.py               # retrains the final model and writes the submission
├── submissions/
│   └── gb_tuned_submission.csv
└── requirements.txt
```

## How to run

1. **Set up the environment** (Python 3.11 or newer):
   ```bash
   python -m venv house-prices
   # Windows:      .\house-prices\Scripts\Activate.ps1
   # macOS/Linux:  source house-prices/bin/activate
   pip install -r requirements.txt
   ```

2. **Download the data** into `data/` (needs a Kaggle account and joining the competition):
   ```bash
   kaggle auth login
   kaggle competitions download -c house-prices-advanced-regression-techniques -p data
   ```
   Then unzip it so that `data/train.csv` and `data/test.csv` exist.

3. **Train the final model and create the submission** (takes a minute or two):
   ```bash
   python src/train.py
   ```
   This prints the cross-validation score and writes `submissions/gb_tuned_submission.csv`.

4. **Explore the full analysis** in the notebooks, in order: `01_exploartion.ipynb`, then
   `02_modeling.ipynb` (the grid searches take several minutes).

## What I learned
- **Data leakage is easy to introduce by accident.** Fixes written for specific training rows, and fill
  values computed on the test set itself, both had to be changed so that everything is learned from
  the training data only.
- **How you evaluate matters as much as what you build.** Removing hard examples from the validation
  folds made the outlier experiment look far more successful than it was.
- **CV scores used for tuning are optimistic;** a held-out score (here the Kaggle leaderboard) is the
  honest measure.

## Possible next steps
- Regularised linear models (Ridge, Lasso), which often do well on this dataset.
- Blending a linear model with Gradient Boosting.
- Tuning LightGBM's subsampling properly (`subsample` only takes effect together with `subsample_freq`).
