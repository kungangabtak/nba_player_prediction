# src/feature_engineering.py

import pandas as pd
import numpy as np
import logging
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import re
from nba_api.stats.static import teams  # Import to get all NBA teams

# Open feature_engineering.py and edit the engineer_features function
def engineer_features(df, players_df):
    """
    Engineer features from game logs.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing concatenated game logs for all players.
        players_df (pd.DataFrame): DataFrame containing player information.

    Returns:
        tuple: (Processed DataFrame, Fitted LabelEncoder)
    """
    logging.info("Starting advanced feature engineering.")
    

    # Check if df is empty
    if df.empty:
        logging.error("Input DataFrame is empty. Cannot engineer features.")
        return pd.DataFrame(), None

    # Step 1: Add original functionality from data_preprocessing.feature_engineering
    # Rolling Averages
    df = df.sort_values(by=['PLAYER_ID', 'GAME_DATE'], ascending=[True, False])
    df['PTS_Rolling_Avg'] = df.groupby('PLAYER_ID')['PTS'].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
    df['REB_Rolling_Avg'] = df.groupby('PLAYER_ID')['REB'].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
    df['AST_Rolling_Avg'] = df.groupby('PLAYER_ID')['AST'].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
    logging.debug("Calculated rolling averages for PTS, REB, AST.")

    # Home vs Away Indicator
    df['HOME_GAME'] = df['MATCHUP'].str.contains('@').map({True: 0, False: 1})
    logging.debug("Added HOME_GAME feature.")

    # Opponent Strength
    opponent_pts_allowed = df.groupby('MATCHUP')['PTS'].transform('mean')
    df['Opponent_PTS_Allowed'] = opponent_pts_allowed
    logging.debug("Added Opponent_PTS_Allowed feature.")

    # Step 2: Continue with the original engineer_features functionality in feature_engineering.py
    # (already present in engineer_features function)
    # Extract Opponent_Team from MATCHUP using the TEAM_ABBREVIATION from game logs
    df['Opponent_Team'] = df.apply(extract_opponent_team, axis=1)
    initial_length = len(df)
    df = df.dropna(subset=['Opponent_Team'])
    dropped_rows = initial_length - len(df)
    if dropped_rows > 0:
        logging.info(f"Dropped {dropped_rows} rows due to inability to extract 'Opponent_Team'.")

    # Encode Opponent_Team with LabelEncoder
    all_teams = teams.get_teams()
    team_abbreviations = [team['abbreviation'] for team in all_teams]
    label_encoder = LabelEncoder()
    label_encoder.fit(team_abbreviations)
    try:
        df['Opponent_Team_Encoded'] = label_encoder.transform(df['Opponent_Team'])
    except ValueError as e:
        logging.error(f"Error encoding Opponent_Team: {e}")
        df['Opponent_Team_Encoded'] = -1

    # Ratio Features and Efficiency
    df['FG3A_FGA_RATIO'] = df['FG3A'] / df['FGA'].replace(0, np.nan)
    df['FG3A_FGA_RATIO'] = df['FG3A_FGA_RATIO'].fillna(0)
    df['FT_FG_RATIO'] = df['FTA'] / df['FGA'].replace(0, np.nan)
    df['FT_FG_RATIO'] = df['FT_FG_RATIO'].fillna(0)
    df['EFFICIENCY'] = (
        df['PTS'] + df['REB'] + df['AST'] + df['STL'] + df['BLK']
        - (df['FGA'] - df['FGM'])
        - (df['FTA'] - df['FTM'])
        - df['TOV']
    )
    df['Usage_Rate'] = (df['FGA'] + 0.44 * df['FTA'] + df['TOV']) / df['Minutes_Played'].replace(0, np.nan)
    df['Usage_Rate'] = df['Usage_Rate'].fillna(0)

    # Fill missing values for numeric features
    numeric_features = ['Minutes_Played', 'FG_Percentage', 'FT_Percentage', 'ThreeP_Percentage', 'Usage_Rate', 'EFFICIENCY']
    df[numeric_features] = df[numeric_features].fillna(0)

    # Recursive Feature Elimination (RFE) and final selection
    selected_feature_names = numeric_features + ['Opponent_Team_Encoded']
    features = df[selected_feature_names]
    target = df['PTS']
    model = XGBRegressor(n_estimators=100, random_state=42)
    rfe = RFE(model, n_features_to_select=5)
    fit = rfe.fit(features, target)
    selected_features = features.columns[fit.support_].tolist()

    # Explicitly ensure FT_Percentage and Opponent_Team are included in selected features
    mandatory_features = ['Opponent_Team_Encoded', 'FT_Percentage', 'Opponent_Team']
    for feature in mandatory_features:
        if feature not in selected_features:
            selected_features.append(feature)

    # Update the DataFrame to only use the final selected features and target
    df = df[selected_features + ['PTS']]
    logging.info(f"Final selected features: {selected_features}")

     # At the end, check if required columns are still present
    print("Columns at the end of engineer_features:", df.columns)

    # Return df and label encoder as usual
    return df, label_encoder

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
