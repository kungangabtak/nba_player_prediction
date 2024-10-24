# src/feature_engineering.py

import pandas as pd
import numpy as np
import logging
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import re

def extract_opponent_team(row, player_team_map):
    """
    Extracts the Opponent_Team from the MATCHUP column.

    Parameters:
        row (pd.Series): A row from the DataFrame containing 'MATCHUP' and 'Player_ID'.
        player_team_map (dict): A dictionary mapping Player_ID to TEAM_ABBREVIATION.

    Returns:
        str or np.nan: The opponent team's abbreviation if successfully extracted, else np.nan.
    """
    player_id = row['Player_ID']
    player_team = player_team_map.get(player_id, None)
    if not player_team:
        logging.warning(f"Player ID {player_id} not found in players_df.")
        return np.nan
    matchup = row['MATCHUP']
    if pd.isna(matchup):
        logging.warning(f"MATCHUP is NaN for Player ID {player_id}.")
        return np.nan
    # MATCHUP format can be 'TEAM vs. OPPONENT' or 'TEAM @ OPPONENT'
    try:
        # Use regular expressions to parse the matchup
        pattern = r'^(?P<team>\w{3})\s+(vs\.?|@)\s+(?P<opponent>\w{3})$'
        match = re.match(pattern, matchup.strip())
        if not match:
            logging.warning(f"Unexpected MATCHUP format: {matchup} for Player ID {player_id}.")
            return np.nan
        team = match.group('team')
        opponent = match.group('opponent')
        if team == player_team:
            return opponent
        elif opponent == player_team:
            return team
        else:
            logging.warning(f"Player team {player_team} not found in MATCHUP: {matchup}.")
            return np.nan
    except Exception as e:
        logging.error(f"Error parsing MATCHUP '{matchup}' for Player ID {player_id}: {e}")
        return np.nan

def engineer_features(df, players_df):
    """
    Engineer features from game logs.

    Parameters:
        df (pd.DataFrame): DataFrame containing concatenated game logs for all players.
        players_df (pd.DataFrame): DataFrame containing player information.

    Returns:
        tuple: (Processed DataFrame, Fitted LabelEncoder)
    """
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
        'PTS': 'PTS'
    }

    # Perform the renaming
    df = df.rename(columns=rename_dict)

    # Log the columns after renaming
    logging.debug(f"Columns after renaming: {df.columns.tolist()}")

    # Extract Opponent_Team from MATCHUP
    # First, get player's team abbreviation from players_df
    player_team_map = players_df.set_index('PERSON_ID')['TEAM_ABBREVIATION'].to_dict()

    df['Opponent_Team'] = df.apply(extract_opponent_team, axis=1, args=(player_team_map,))

    # Drop rows where Opponent_Team could not be extracted
    initial_length = len(df)
    df = df.dropna(subset=['Opponent_Team'])
    dropped_rows = initial_length - len(df)
    if dropped_rows > 0:
        logging.info(f"Dropped {dropped_rows} rows due to inability to extract 'Opponent_Team'.")

    # Ensure Opponent_Team is string type
    df['Opponent_Team'] = df['Opponent_Team'].astype(str)

    # Initialize and fit LabelEncoder for Opponent_Team
    label_encoder = LabelEncoder()
    df['Opponent_Team_Encoded'] = label_encoder.fit_transform(df['Opponent_Team'])

    # Create ratio features
    df['FG3A_FGA_RATIO'] = df['FG3A'] / df['FGA'].replace(0, np.nan)
    df['FG3A_FGA_RATIO'] = df['FG3A_FGA_RATIO'].fillna(0)

    df['FT_FG_RATIO'] = df['FTA'] / df['FGA'].replace(0, np.nan)
    df['FT_FG_RATIO'] = df['FT_FG_RATIO'].fillna(0)

    # Create an efficiency metric
    df['EFFICIENCY'] = (
        df['PTS'] + df['REB'] + df['AST'] + df['STL'] + df['BLK']
        - (df['FGA'] - df['FGM'])
        - (df['FTA'] - df['FTM'])
        - df['TOV']
    )

    # Drop 'GAME_DATE' or any other non-numeric columns if present
    non_numeric_cols = df.select_dtypes(exclude=['number', 'object']).columns.tolist()
    if 'GAME_DATE' in non_numeric_cols:
        df = df.drop(columns=['GAME_DATE'])
        logging.info("Dropped 'GAME_DATE' column.")

    # Drop 'PLAYER_NAME' if present
    if 'PLAYER_NAME' in df.columns:
        df = df.drop(columns=['PLAYER_NAME'])
        logging.info("Dropped 'PLAYER_NAME' column.")

    # Compute Usage Rate
    df['Usage_Rate'] = (df['FGA'] + 0.44 * df['FTA'] + df['TOV']) / df['Minutes_Played'].replace(0, np.nan)
    df['Usage_Rate'] = df['Usage_Rate'].fillna(0)

    # Select features for Recursive Feature Elimination
    selected_feature_names = ['Minutes_Played', 'FG_Percentage', 'FT_Percentage', 
                              'ThreeP_Percentage', 'Usage_Rate', 'EFFICIENCY', 'Opponent_Team_Encoded']

    # Check if all selected features are present
    missing_feature_selection_cols = [col for col in selected_feature_names if col not in df.columns]
    if missing_feature_selection_cols:
        logging.warning(f"The following feature selection columns are missing: {missing_feature_selection_cols}. Filling them with 0.")
        for col in missing_feature_selection_cols:
            df[col] = 0

    features = df[selected_feature_names]
    target = df['PTS']

    # Handle missing values
    if features.isnull().any().any():
        logging.warning("Missing values detected in features. Filling with 0.")
        features = features.fillna(0)

    # Feature Selection using Recursive Feature Elimination
    model = XGBRegressor(n_estimators=100, random_state=42)
    rfe = RFE(model, n_features_to_select=5)
    try:
        fit = rfe.fit(features, target)
    except ValueError as ve:
        logging.error(f"Error during RFE fitting: {ve}")
        return pd.DataFrame(), None

    selected_features = features.columns[fit.support_].tolist()

    # Ensure 'Opponent_Team_Encoded' is included
    if 'Opponent_Team_Encoded' not in selected_features:
        selected_features.append('Opponent_Team_Encoded')

    df = df[selected_features + ['PTS']]
    logging.info(f"Selected features: {selected_features}")

    # Log the final columns after feature selection
    logging.debug(f"Final columns after feature selection: {df.columns.tolist()}")

    return df, label_encoder