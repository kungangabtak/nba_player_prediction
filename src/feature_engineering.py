# src/feature_engineering.py

import pandas as pd

def engineer_features(df):
    df['FG3A_FGA_RATIO'] = df['FG3A'] / df['FGA']
    df['FT_FG_RATIO'] = df['FTA'] / df['FGA']
    df['EFFICIENCY'] = (
        df['Points'] + df['REB'] + df['AST'] + df['STL'] + df['BLK']
        - (df['FGA'] - df['FGM'])
        - (df['FTA'] - df['FTM'])
        - df['TOV']
    )
    return df