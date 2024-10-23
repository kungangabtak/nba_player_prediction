# src/data_preprocessing.py

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def clean_data(df):
    # Select only numeric columns for imputation
    numeric_cols = df.select_dtypes(include=['number']).columns
    imputer = SimpleImputer(strategy='mean')
    
    try:
        df_imputed = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)
    except ValueError as e:
        raise ValueError(f"Imputation failed: {e}")
    
    return df_imputed