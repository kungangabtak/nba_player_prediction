# src/data_preprocessing.py

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def clean_data(df):
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_imputed

def scale_data(df, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df)
    else:
        scaled = scaler.transform(df)
    return pd.DataFrame(scaled, columns=df.columns), scaler