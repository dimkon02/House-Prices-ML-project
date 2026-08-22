import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#1. Load
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

#2. Clean deterministic data from Phase 2
def clean_deterministic_data(df):
    cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
            'GarageFinish', 'GarageQual', 'GarageCond', 'GarageType', 
            'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual']
    df[cols] = df[cols].fillna('None')
    return df

