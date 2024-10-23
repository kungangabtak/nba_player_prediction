# src/feature_engineering.py

import pandas as pd
import numpy as np
import logging
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor

def engineer_features(df):
    # Log the initial columns
    logging.debug(f"Initial gamelog columns: {df.columns.tolist()}")
    
    # Rename columns from NBA API standards to expected feature names
    rename_dict = {
        'MIN': 'Minutes_Played',
        'FG_PCT': 'FG_Percentage',
        'FT_PCT': 'FT_Percentage',
        'FG3_PCT': 'ThreeP_Percentage',
        'REB': 'REB',
        'AST': 'AST',
        'STL': 'STL',
        'BLK': 'BLK',
        'FGA': 'FGA',
        'FGM': 'FGM',
        'FTA': 'FTA',
        'FTM': 'FTM',
        'TOV': 'TOV',
        'TEAM_ABBREVIATION': 'Opponent_Team'
    }
    
    # Check if all required columns are present before renaming
    missing_cols = [orig for orig in rename_dict.keys() if orig not in df.columns]
    if missing_cols:
        logging.warning(f"The following expected columns are missing from gamelog: {missing_cols}")
        # Depending on requirements, you can choose to skip processing this gamelog or fill missing columns with defaults
        for col in missing_cols:
            df[col] = 0  # Filling missing numerical columns with 0
        logging.info("Filled missing columns with default values (0).")
    
    # Perform the renaming
    df = df.rename(columns=rename_dict)
    
    # Log the columns after renaming
    logging.debug(f"Columns after renaming: {df.columns.tolist()}")
    
    # Create ratio features
    if 'FG3A' in df.columns and 'FGA' in df.columns:
        df['FG3A_FGA_RATIO'] = df['FGA'] / df['FG3A'].replace(0, np.nan)
        df['FG3A_FGA_RATIO'] = df['FG3A_FGA_RATIO'].fillna(0)
    else:
        logging.warning("Columns 'FG3A' or 'FGA' missing. Creating 'FG3A_FGA_RATIO' with default value 0.")
        df['FG3A_FGA_RATIO'] = 0
    
    if 'FTA' in df.columns and 'FGA' in df.columns:
        df['FT_FG_RATIO'] = df['FTA'] / df['FGA'].replace(0, np.nan)
        df['FT_FG_RATIO'] = df['FT_FG_RATIO'].fillna(0)
    else:
        logging.warning("Columns 'FTA' or 'FGA' missing. Creating 'FT_FG_RATIO' with default value 0.")
        df['FT_FG_RATIO'] = 0
    
    # Create an efficiency metric
    required_efficiency_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FGA', 'FGM', 'FTA', 'FTM', 'TOV']
    missing_efficiency_cols = [col for col in required_efficiency_cols if col not in df.columns]
    if missing_efficiency_cols:
        logging.warning(f"The following columns are missing for efficiency calculation: {missing_efficiency_cols}. Filling them with 0.")
        for col in missing_efficiency_cols:
            df[col] = 0
    
    df['EFFICIENCY'] = (
        df['PTS'] + df['REB'] + df['AST'] + df['STL'] + df['BLK']
        - (df['FGA'] - df['FGM'])
        - (df['FTA'] - df['FTM'])
        - df['TOV']
    )
    
    # Drop 'GAME_DATE' or any other non-numeric columns if present
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    if 'GAME_DATE' in non_numeric_cols:
        df = df.drop(columns=['GAME_DATE'])
        logging.info("Dropped 'GAME_DATE' column.")
    
    # Drop 'PLAYER_NAME' if present
    if 'PLAYER_NAME' in df.columns:
        df = df.drop(columns=['PLAYER_NAME'])
        logging.info("Dropped 'PLAYER_NAME' column.")
    
    # Select features for Recursive Feature Elimination
    selected_feature_names = ['Minutes_Played', 'FG_Percentage', 'FT_Percentage', 
                              'ThreeP_Percentage', 'Usage_Rate', 'EFFICIENCY', 'Opponent_Team']
    
    # Check if all selected features are present
    missing_feature_selection_cols = [col for col in selected_feature_names if col not in df.columns]
    if missing_feature_selection_cols:
        logging.warning(f"The following feature selection columns are missing: {missing_feature_selection_cols}. Filling them with 0.")
        for col in missing_feature_selection_cols:
            df[col] = 0
    
    features = df[selected_feature_names]
    target = df['PTS']
    
    # Feature Selection using Recursive Feature Elimination
    model = XGBRegressor(n_estimators=100, random_state=42)
    rfe = RFE(model, n_features_to_select=5)
    fit = rfe.fit(features, target)
    selected_features = features.columns[fit.support_]
    df = df[selected_features.tolist() + ['PTS']]
    logging.info(f"Selected features: {selected_features.tolist()}")
    
    # Log the final columns after feature selection
    logging.debug(f"Final columns after feature selection: {df.columns.tolist()}")
    
    return df