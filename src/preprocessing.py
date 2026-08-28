import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

#1. Load
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

#2. Clean deterministic data from Phase 2
# def clean_deterministic_data(df):

#     df.loc[332, 'BsmtFinType2'] = 'Unf'
#     df.loc[948, 'BsmtExposure'] = df['BsmtExposure'].mode()[0]

#     cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
#             'GarageFinish', 'GarageQual', 'GarageCond', 'GarageType', 
#             'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual']
#     df[cols] = df[cols].fillna('None')

#     # MasVnrType / MasVnrArea — special handling for 5 inconsistent rows
#     mode_type = df['MasVnrType'].mode()[0]
#     inconsistent_mask = df['MasVnrType'].isnull() & (df['MasVnrArea'] > 0)
#     df.loc[inconsistent_mask, 'MasVnrType'] = mode_type
#     df['MasVnrType'] = df['MasVnrType'].fillna('None')
#     df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

#     df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

#     df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
    
#     return df

def clean_deterministic_data(df):

    df.loc[332, 'BsmtFinType2'] = 'Unf'
    df.loc[948, 'BsmtExposure'] = df['BsmtExposure'].mode()[0]

    cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
            'GarageFinish', 'GarageQual', 'GarageCond', 'GarageType', 
            'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual']
    df[cols] = df[cols].fillna('None')

    # MasVnrType / MasVnrArea — special handling for 5 inconsistent rows
    mode_type = df['MasVnrType'].mode()[0]
    inconsistent_mask = df['MasVnrType'].isnull() & (df['MasVnrArea'] > 0)
    df.loc[inconsistent_mask, 'MasVnrType'] = mode_type
    df['MasVnrType'] = df['MasVnrType'].fillna('None')
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)

    # Basement/Garage numeric columns — only appear missing in test.csv,
    # likely means "no basement"/"no garage", consistent with Phase 2 findings
    zero_fill_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                       'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']
    df[zero_fill_cols] = df[zero_fill_cols].fillna(0)

    # Genuinely missing categoricals, only appear missing in test.csv — fill with mode
    mode_fill_cols = ['MSZoning', 'Exterior1st', 'Exterior2nd', 'SaleType',
                       'KitchenQual', 'Utilities', 'Functional']
    for col in mode_fill_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

# 3. Clean — statistics-based parts (must fit on train, reuse on test)
def fit_imputation_stats(df):
    return {
        'neighborhood_median': df.groupby('Neighborhood')['LotFrontage'].median(),
        'overall_median': df['LotFrontage'].median()
    }

def apply_imputation_stats(df, stats):
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(stats['neighborhood_median'].get(x.name, stats['overall_median']))
    )
    return df

def flag_outliers(df):
    df['is_outlier'] = df.index.isin([523, 1298])
    return df


# 4. Feature engineering (ordinal encoding, combined features, age features, binary flags)

ORDINAL_MAPS = {
    'ExterQual': {'Fa': 0, 'TA': 1, 'Gd': 2, 'Ex': 3},
    'KitchenQual': {'Fa': 0, 'TA': 1, 'Gd': 2, 'Ex': 3},
    'ExterCond': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
    'HeatingQC': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
    'BsmtQual': {'None': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
    'PoolQC': {'None': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
    'BsmtCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'FireplaceQu': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'GarageQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'GarageCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'CentralAir': {'N': 0, 'Y': 1},
    'LandSlope': {'Sev': 0, 'Mod': 1, 'Gtl': 2},
    'Street': {'Grvl': 0, 'Pave': 1},
    'LotShape': {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3},
    'Utilities': {'NoSeWa': 0, 'AllPub': 1},
    'Functional': {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7},
    'GarageFinish': {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
    'PavedDrive': {'N': 0, 'P': 1, 'Y': 2},
    'BsmtExposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
    'BsmtFinType1': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
    'BsmtFinType2': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
}


def encode_ordinal_features(df):
    for col, mapping in ORDINAL_MAPS.items():
        df[f'{col}_encoded'] = df[col].map(mapping)  #f'...' takes the col (ex. 'ExterQual' and makes it to 'ExterQual_econded')
    return df

def add_combined_features(df):
    df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    df['TotalSF'] = df['TotalBsmtSF'] + df['GrLivArea']
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch'] + df['WoodDeckSF']

    return df

def add_age_features(df):
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['HouseRemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']
    df.loc[df['GarageYrBlt'] == 0, 'GarageAge'] = 0
    return df   

def add_binary_flags(df):
    df['Has2ndFloor'] = df['2ndFlrSF'] > 0
    df['HasFireplace'] = df['Fireplaces'] > 0
    df['HasPool'] = df['PoolQC_encoded'] > 0
    df['HasGarage'] = df['GarageArea'] > 0
    df['HasBsmt'] = df['TotalBsmtSF'] > 0
    df['HasFence'] = df['Fence'] != 'None'
    df['HasMasVnr'] = df['MasVnrArea'] > 0
    df['IsNew'] = df['SaleType'] == 'New'
    df['IsPartialSale'] = df['SaleCondition'] == 'Partial'

    binary_cols = ['Has2ndFloor', 'HasFireplace', 'HasPool', 'HasGarage', 'HasBsmt',
                   'HasFence', 'HasMasVnr', 'IsNew', 'IsPartialSale']
    df[binary_cols] = df[binary_cols].astype(int)
    return df

def engineer_features(df):
    df = encode_ordinal_features(df)
    df = add_combined_features(df)
    df = add_age_features(df)
    df = add_binary_flags(df)
    return df

# 5. Drop original text columns now replaced by _encoded versions
def drop_redundant_columns(df):
    cols_to_drop = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
                 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC',
                 'CentralAir', 'LandSlope', 'Street', 'LotShape', 'Utilities',
                 'Functional', 'GarageFinish', 'PavedDrive',
                 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

    df = df.drop(columns=cols_to_drop)
    return df

# 7. Build the ColumnTransformer (one-hot + scaling)

NOMINAL_COLS = ['MSSubClass', 'MSZoning', 'Alley', 'LandContour', 'LotConfig',
                 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
                 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                 'Foundation', 'Heating', 'Electrical', 'GarageType', 'Fence', 'MiscFeature',
                 'SaleType', 'SaleCondition']

def build_preprocessor(X):

    
    numeric_cols = [col for col in X.columns if col not in NOMINAL_COLS]

    preprocessor = ColumnTransformer(transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), NOMINAL_COLS),
        ('scale', StandardScaler(), numeric_cols)
    ])
    
    return preprocessor

def build_preprocessor_scaled(X):

    numeric_cols = [col for col in X.columns if col not in NOMINAL_COLS]

    nominal_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ('scale', StandardScaler(with_mean=False))       
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('nominal', nominal_transformer, NOMINAL_COLS),
        ('numeric', StandardScaler(), numeric_cols)
    ])

    return preprocessor

def split_features_target(df):
    if 'SalePrice' in df.columns:
        y = np.log1p(df['SalePrice'])
        X = df.drop(columns=['SalePrice', 'Id', 'is_outlier'])
    else:
        y = None
        X = df.drop(columns=['Id', 'is_outlier'])
    return X, y

def full_pipeline(file_path, stats=None):
    df = load_data(file_path)
    df = clean_deterministic_data(df)
    
    if stats is None:
        stats = fit_imputation_stats(df)
    
    df = apply_imputation_stats(df, stats)
    df = engineer_features(df)
    df = flag_outliers(df)
    df = drop_redundant_columns(df)
    X, y = split_features_target(df)
    return X, y, stats

if __name__ == '__main__':
    df = load_data('../data/train.csv')
    df = clean_deterministic_data(df)
    stats = fit_imputation_stats(df)
    df = apply_imputation_stats(df, stats)
    df = engineer_features(df)
    df = flag_outliers(df) 
    df = drop_redundant_columns(df)
    X, y = split_features_target(df)
    
    print("Nulls in X:", X.isnull().sum().sum())
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    
    preprocessor = build_preprocessor(X)
    X_transformed = preprocessor.fit_transform(X)
    print("Transformed X shape:", X_transformed.shape)