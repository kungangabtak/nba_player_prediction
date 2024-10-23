# src/data_preprocessing.py

import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
import logging

def clean_data(df):
    # Log all columns before cleaning
    logging.info(f"Columns before cleaning: {df.columns.tolist()}")
    
    # Replace infinite values with NaN
    num_inf = np.isinf(df.select_dtypes(include=['number'])).sum().sum()
    if num_inf > 0:
        logging.warning(f"Found {num_inf} infinite values. Replacing with NaN.")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Optionally, cap extremely large values to a reasonable maximum
    # Define a threshold (e.g., 99th percentile) to cap values
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        upper_limit = df[col].quantile(0.99)
        if df[col].max() > upper_limit:
            logging.warning(f"Capping values in column '{col}' at {upper_limit}.")
            df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])
    
    # Additional outlier detection
    for col in numeric_cols:
        lower_limit = df[col].quantile(0.01)
        if df[col].min() < lower_limit:
            logging.warning(f"Capping values in column '{col}' at {lower_limit}.")
            df[col] = np.where(df[col] < lower_limit, lower_limit, df[col])
    
    # Check for any remaining infinite values
    if np.isinf(df[numeric_cols]).values.any():
        logging.error("Data contains infinite values even after replacement.")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Select only numeric columns for imputation
    numeric_cols = df.select_dtypes(include=['number']).columns
    logging.info(f"Numeric columns selected for imputation: {numeric_cols.tolist()}")
    
    # Verify data types
    logging.info(f"Data types of selected columns:\n{df[numeric_cols].dtypes}")
    
    # Check for any remaining NaN values
    num_nan = df[numeric_cols].isna().sum().sum()
    if num_nan > 0:
        logging.info(f"Found {num_nan} NaN values. Proceeding with imputation.")
    
    # Use KNN Imputer for more sophisticated imputation
    imputer = KNNImputer(n_neighbors=5)
    
    # Log the data to be imputed
    logging.debug(f"Data before imputation:\n{df[numeric_cols].head()}")
    
    try:
        df_imputed = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)
        logging.info("Data imputation with KNN successful.")
    except ValueError as e:
        logging.error(f"KNN Imputation failed: {e}")
        raise ValueError(f"KNN Imputation failed: {e}")
    
    # Log detailed imputation information
    missing_before = df[numeric_cols].isna().sum()
    missing_after = df_imputed.isna().sum()
    logging.info(f"Missing values before imputation:\n{missing_before}")
    logging.info(f"Missing values after imputation:\n{missing_after}")
    
    # Log summary statistics after cleaning
    logging.info("Summary statistics after cleaning:")
    logging.info(df_imputed.describe())
    
    return df_imputed