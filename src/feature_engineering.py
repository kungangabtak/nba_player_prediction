# src/feature_engineering.py

import pandas as pd
import numpy as np
import logging
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import re
from nba_api.stats.static import teams  # Import to get all NBA teams

def extract_opponent_team(row):
    """
    Extracts the Opponent_Team from the MATCHUP column using the TEAM_ABBREVIATION from the game log.

    Parameters:
        row (pd.Series): A row from the DataFrame containing 'MATCHUP', 'TEAM_ABBREVIATION', and 'PLAYER_ID'.

    Returns:
        str or np.nan: The opponent team's abbreviation if successfully extracted, else np.nan.
    """
    player_team = row.get('TEAM_ABBREVIATION', None)
    if not player_team or pd.isna(player_team):
        logging.warning(f"TEAM_ABBREVIATION is missing or NaN for Player ID {row.get('PLAYER_ID', 'Unknown')}.")
        return np.nan
    matchup = row.get('MATCHUP', None)
    if not matchup or pd.isna(matchup):
        logging.warning(f"MATCHUP is missing or NaN for Player ID {row.get('PLAYER_ID', 'Unknown')}.")
        return np.nan
    # MATCHUP format can be 'TEAM_ABBR vs. OPPONENT_ABBR' or 'TEAM_ABBR @ OPPONENT_ABBR'
    try:
        # Use regular expressions to parse the matchup
        pattern = r'^(?P<team>\w{3})\s+(vs\.?|@)\s+(?P<opponent>\w{3})$'
        match = re.match(pattern, matchup.strip())
        if not match:
            logging.warning(f"Unexpected MATCHUP format: {matchup} for Player ID {row.get('PLAYER_ID', 'Unknown')}.")
            return np.nan
        team = match.group('team')
        opponent = match.group('opponent')
        if team == player_team:
            return opponent
        elif opponent == player_team:
            return team
        else:
            logging.warning(f"Player team {player_team} not found in MATCHUP: {matchup} for Player ID {row.get('PLAYER_ID', 'Unknown')}.")
            return np.nan
    except Exception as e:
        logging.error(f"Error parsing MATCHUP '{matchup}' for Player ID {row.get('PLAYER_ID', 'Unknown')}: {e}")
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
    # Check if df is empty
    if df.empty:
        logging.error("Input DataFrame is empty. Cannot engineer features.")
        return pd.DataFrame(), None

    # Check if required columns exist
    required_columns = ['MATCHUP', 'TEAM_ABBREVIATION', 'PLAYER_ID']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing columns in game logs: {missing_columns}")
        return pd.DataFrame(), None

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
        'PTS': 'PTS',
        'FG3A': 'FG3A'
    }

    # Perform the renaming
    df = df.rename(columns=rename_dict)

    # Log the columns after renaming
    logging.debug(f"Columns after renaming: {df.columns.tolist()}")

    # Extract Opponent_Team from MATCHUP using the TEAM_ABBREVIATION from game logs
    df['Opponent_Team'] = df.apply(extract_opponent_team, axis=1)

    # Drop rows where Opponent_Team could not be extracted
    initial_length = len(df)
    df = df.dropna(subset=['Opponent_Team'])
    dropped_rows = initial_length - len(df)
    if dropped_rows > 0:
        logging.info(f"Dropped {dropped_rows} rows due to inability to extract 'Opponent_Team'.")

    # Ensure Opponent_Team is string type
    df['Opponent_Team'] = df['Opponent_Team'].astype(str)

    # Get all NBA team abbreviations
    all_teams = teams.get_teams()
    team_abbreviations = [team['abbreviation'] for team in all_teams]

    # Initialize and fit LabelEncoder for Opponent_Team with all NBA teams
    label_encoder = LabelEncoder()
    label_encoder.fit(team_abbreviations)

    # Transform Opponent_Team using the fitted LabelEncoder
    try:
        df['Opponent_Team_Encoded'] = label_encoder.transform(df['Opponent_Team'])
    except ValueError as e:
        # Handle unknown teams (should not occur if all teams are included)
        logging.error(f"Error encoding Opponent_Team: {e}")
        # Assign -1 to unknown teams
        df['Opponent_Team_Encoded'] = -1

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

    # Compute Usage Rate
    df['Usage_Rate'] = (df['FGA'] + 0.44 * df['FTA'] + df['TOV']) / df['Minutes_Played'].replace(0, np.nan)
    df['Usage_Rate'] = df['Usage_Rate'].fillna(0)

    # Handle missing numeric features
    numeric_features = ['Minutes_Played', 'FG_Percentage', 'FT_Percentage',
                        'ThreeP_Percentage', 'Usage_Rate', 'EFFICIENCY']
    for feature in numeric_features:
        if feature not in df.columns:
            logging.warning(f"Numeric feature '{feature}' is missing. Filling with 0.")
            df[feature] = 0

    # Handle missing values in numeric features
    df[numeric_features] = df[numeric_features].fillna(0)

    # Select features for Recursive Feature Elimination
    selected_feature_names = numeric_features + ['Opponent_Team_Encoded']

    features = df[selected_feature_names]
    target = df['PTS']

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