# src/feature_engineering.py

import pandas as pd
import numpy as np
import logging
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import re

def extract_opponent_team(row, player_team_map):
    """
    Extracts the Opponent_Team from the MATCHUP column.

    Parameters:
        row (pd.Series): A row from the DataFrame containing 'MATCHUP' and 'Player_ID'.
        player_team_map (dict): A dictionary mapping Player_ID to TEAM_ABBREVIATION.

    Returns:
        int or np.nan: The encoded opponent team's label if successfully extracted, else np.nan.
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
    # Use regular expressions to handle various formats
    try:
        # Pattern to match 'TEAM vs. OPPONENT' or 'TEAM @ OPPONENT' with optional periods
        pattern = r'^(?P<home_team>\w{3})\s*(vs\.?|@)\s*(?P<away_team>\w{3})$'
        match = re.match(pattern, matchup.strip())
        if not match:
            logging.warning(f"Unexpected MATCHUP format: {matchup} for Player ID {player_id}.")
            return np.nan
        home_team = match.group('home_team')
        away_team = match.group('away_team')
        home_away_indicator = match.group(2)

        # Determine if the player is home or away
        # Assuming 'vs' implies player is home, '@' implies player is away
        if home_away_indicator.startswith('vs'):
            player_is_home = True
        elif home_away_indicator.startswith('@'):
            player_is_home = False
        else:
            logging.warning(f"Unknown home/away indicator '{home_away_indicator}' in MATCHUP: {matchup} for Player ID {player_id}.")
            return np.nan

        if player_is_home:
            if player_team != home_team:
                logging.warning(f"Player team {player_team} does not match home team {home_team} in MATCHUP: {matchup}.")
                return np.nan
            opponent = away_team
        else:
            if player_team != away_team:
                logging.warning(f"Player team {player_team} does not match away team {away_team} in MATCHUP: {matchup}.")
                return np.nan
            opponent = home_team

        return opponent
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

    # Check if all required columns are present before renaming
    missing_cols = [orig for orig in rename_dict.keys() if orig not in df.columns]
    if missing_cols:
        logging.warning(f"The following expected columns are missing from gamelog: {missing_cols}")
        # Fill missing numerical columns with 0
        for col in missing_cols:
            df[col] = 0
        logging.info("Filled missing columns with default values (0).")

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
    df['Opponent_Team'] = label_encoder.fit_transform(df['Opponent_Team'])

    # Create ratio features
    if 'FG3A' in df.columns and 'FGA' in df.columns:
        df['FG3A_FGA_RATIO'] = df['FG3A'] / df['FGA'].replace(0, np.nan)
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
    non_numeric_cols = df.select_dtypes(exclude=['number', 'object']).columns.tolist()
    if 'GAME_DATE' in non_numeric_cols:
        df = df.drop(columns=['GAME_DATE'])
        logging.info("Dropped 'GAME_DATE' column.")

    # Drop 'PLAYER_NAME' if present
    if 'PLAYER_NAME' in df.columns:
        df = df.drop(columns=['PLAYER_NAME'])
        logging.info("Dropped 'PLAYER_NAME' column.")

    # Compute Usage Rate
    # Usage Rate Formula:
    # USG% = 100 * ((FGA + 0.44 * FTA + TOV) * (Team Minutes)) / (Minutes Played * (Team FGA + 0.44 * Team FTA + Team TOV))
    # Since team stats are not available, we'll approximate Usage Rate as:
    # USG% = (FGA + 0.44 * FTA + TOV) / Minutes_Played

    # Handle division by zero
    df['Usage_Rate'] = (df['FGA'] + 0.44 * df['FTA'] + df['TOV']) / df['Minutes_Played'].replace(0, np.nan)
    df['Usage_Rate'] = df['Usage_Rate'].fillna(0)

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

    # Verify that all required features are present
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

    selected_features = features.columns[fit.support_]
    df = df[selected_features.tolist() + ['PTS']]
    logging.info(f"Selected features: {selected_features.tolist()}")

    # Log the final columns after feature selection
    logging.debug(f"Final columns after feature selection: {df.columns.tolist()}")

    return df, label_encoder