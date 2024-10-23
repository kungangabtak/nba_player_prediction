# src/feature_engineering.py

import pandas as pd
import logging

def engineer_features(df):
    # Create ratio features
    df['FG3A_FGA_RATIO'] = df['FG3A'] / df['FGA']
    df['FT_FG_RATIO'] = df['FTA'] / df['FGA']
    
    # Create an efficiency metric
    df['EFFICIENCY'] = (
        df['Points'] + df['REB'] + df['AST'] + df['STL'] + df['BLK']
        - (df['FGA'] - df['FGM'])
        - (df['FTA'] - df['FTM'])
        - df['TOV']
    )
    
    # Drop 'GAME_DATE' or any other non-numeric columns if present
    if 'GAME_DATE' in df.columns:
        df = df.drop(columns=['GAME_DATE'])
        logging.info("Dropped 'GAME_DATE' column.")
    
    return df