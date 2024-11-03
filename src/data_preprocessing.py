# src/data_preprocessing.py

import pandas as pd
from sklearn.impute import SimpleImputer
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
    
    # Cap extremely large and small values to reasonable limits (1st and 99th percentiles)
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        lower_limit = df[col].quantile(0.01)
        upper_limit = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
        logging.debug(f"Capped values in column '{col}' between {lower_limit} and {upper_limit}.")
    
    # Check for any remaining infinite values
    if np.isinf(df[numeric_cols]).values.any():
        logging.error("Data contains infinite values even after replacement.")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Check for any remaining NaN values
    num_nan = df[numeric_cols].isna().sum().sum()
    if num_nan > 0:
        logging.info(f"Found {num_nan} NaN values. Proceeding with imputation.")
    else:
        logging.info("No NaN values found in numeric columns.")
    
    # Use SimpleImputer to handle missing values (impute with mean)
    imputer = SimpleImputer(strategy='mean')
    try:
        df_imputed = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)
        logging.info("Data imputation with mean strategy successful.")
    except ValueError as e:
        logging.error(f"Imputation failed: {e}")
        raise ValueError(f"Imputation failed: {e}")
    
    # Log detailed imputation information
    missing_before = df[numeric_cols].isna().sum()
    missing_after = df_imputed.isna().sum()
    logging.info(f"Missing values before imputation:\n{missing_before}")
    logging.info(f"Missing values after imputation:\n{missing_after}")
    
    # Combine imputed numeric data with non-numeric data
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns
    df_cleaned = pd.concat([df_imputed, df[non_numeric_cols].reset_index(drop=True)], axis=1)
    
    # Log summary statistics after cleaning
    logging.info("Summary statistics after cleaning:")
    logging.info(df_imputed.describe())
    
    return df_cleaned

