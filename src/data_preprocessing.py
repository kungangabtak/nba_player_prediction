# src/data_preprocessing.py

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import logging

def clean_data(df):
    # Log all columns before cleaning
    logging.info(f"Columns before cleaning: {df.columns.tolist()}")
    
    # Select only numeric columns for imputation
    numeric_cols = df.select_dtypes(include=['number']).columns
    logging.info(f"Numeric columns selected for imputation: {numeric_cols.tolist()}")
    
    # Verify data types
    logging.info(f"Data types of selected columns:\n{df[numeric_cols].dtypes}")
    
    imputer = SimpleImputer(strategy='mean')
    
    # Log the data to be imputed
    logging.debug(f"Data before imputation:\n{df[numeric_cols].head()}")
    
    try:
        df_imputed = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)
        logging.info("Data imputation successful.")
    except ValueError as e:
        logging.error(f"Imputation failed: {e}")
        raise ValueError(f"Imputation failed: {e}")
    
    return df_imputed