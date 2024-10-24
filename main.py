# main.py

import logging
import sys
from nba_api.stats.endpoints import commonallplayers, playergamelog
from tqdm import tqdm
import pandas as pd
from src.feature_engineering import engineer_features
from src.utils import get_player_id, get_full_team_name
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
import joblib
import os

def setup_logging():
    """
    Configures the logging settings for the script.
    Logs are output to both the console and a file named 'training.log'.
    """
    logging.basicConfig(
        level=logging.DEBUG,  # Set to INFO or WARNING in production
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("training.log")
        ]
    )

def fetch_players(season='2024-25'):
    """
    Fetches all players for the specified NBA season.

    Parameters:
        season (str): The NBA season (e.g., '2024-25').

    Returns:
        pd.DataFrame: DataFrame containing all players.
    """
    logging.info(f"Starting data collection for the {season} season.")
    try:
        players_all = commonallplayers.CommonAllPlayers(
            is_only_current_season=0,
            league_id='00',
            season=season
        ).get_data_frames()[0]
        logging.info(f"Retrieved {len(players_all)} players for the {season} season.")
        return players_all
    except Exception as e:
        logging.error(f"Error fetching players for season {season}: {e}")
        sys.exit(1)

def filter_players(players_df, team_abbreviations):
    """
    Filters players based on the specified team abbreviations.

    Parameters:
        players_df (pd.DataFrame): DataFrame containing all players.
        team_abbreviations (list): List of team abbreviations to filter by (e.g., ['LAL', 'BOS']).

    Returns:
        pd.DataFrame: DataFrame containing filtered players.
    """
    logging.info(f"Filtering players from teams: {team_abbreviations}")
    filtered_players = players_df[players_df['TEAM_ABBREVIATION'].isin(team_abbreviations)]
    logging.info(f"Number of players after filtering: {len(filtered_players)}")
    return filtered_players

def fetch_game_logs(filtered_players, season='2024-25'):
    """
    Fetches game logs for each player in the filtered players DataFrame.

    Parameters:
        filtered_players (pd.DataFrame): DataFrame containing filtered players.
        season (str): The NBA season (e.g., '2024-25').

    Returns:
        pd.DataFrame: Concatenated DataFrame of all fetched game logs.
    """
    all_game_logs = []
    for index, row in tqdm(filtered_players.iterrows(), total=filtered_players.shape[0], desc="Fetching game logs"):
        player_id = row['PERSON_ID']
        player_name = row['DISPLAY_FIRST_LAST']
        logging.debug(f"Fetching game logs for player {player_name} (ID: {player_id})")
        try:
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star='Regular Season'
            ).get_data_frames()[0]
            if gamelog.empty:
                logging.warning(f"No game log data for player ID {player_id}. Skipping.")
                continue
            gamelog['PLAYER_NAME'] = player_name
            all_game_logs.append(gamelog)
            logging.info(f"Successfully fetched data for player ID {player_id}.")
        except Exception as e:
            logging.error(f"Error fetching game log for player ID {player_id}: {e}")
            continue
    if not all_game_logs:
        logging.error("No game logs fetched for any players.")
        return pd.DataFrame()
    else:
        logging.info(f"Fetched game logs for {len(all_game_logs)} players.")
        return pd.concat(all_game_logs, ignore_index=True)

def train_and_save_models(processed_data, label_encoder, models_dir='models'):
    """
    Trains machine learning models and saves them along with encoders and scalers.

    Parameters:
        processed_data (pd.DataFrame): DataFrame containing processed features and target.
        label_encoder (LabelEncoder): Fitted LabelEncoder for Opponent_Team.
        models_dir (str): Directory to save the trained models and preprocessors.
    """
    if processed_data.empty:
        logging.error("No data collected after processing filtered players. Exiting the program.")
        sys.exit(1)

    logging.info("Starting feature scaling.")

    # Initialize StandardScaler for numerical features
    scaler = StandardScaler()
    numeric_features = ['Minutes_Played', 'FG_Percentage', 'FT_Percentage', 
                        'ThreeP_Percentage', 'Usage_Rate', 'EFFICIENCY']

    # Ensure that all numeric features exist
    for feature in numeric_features:
        if feature not in processed_data.columns:
            logging.warning(f"Numeric feature '{feature}' is missing. Filling with 0.")
            processed_data[feature] = 0

    processed_data[numeric_features] = scaler.fit_transform(processed_data[numeric_features])

    # Ensure 'Opponent_Team_Encoded' exists
    if 'Opponent_Team_Encoded' not in processed_data.columns:
        logging.error("'Opponent_Team_Encoded' is missing from processed_data after feature engineering.")
        sys.exit(1)

    # Split data into features and target
    X = processed_data[numeric_features + ['Opponent_Team_Encoded']]
    y = processed_data['PTS']

    # Split into train and test sets for regression
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logging.info("Training XGBoost Regressor.")
    regressor = XGBRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train, y_train)
    logging.info("XGBoost Regressor training completed.")

    # For classification, define a target (e.g., PTS > median as binary classification)
    y_class = (y > y.median()).astype(int)
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )

    logging.info("Training XGBoost Classifier.")
    classifier = XGBClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train_cls, y_train_cls)
    logging.info("XGBoost Classifier training completed.")

    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)

    # Save models and preprocessors
    try:
        joblib.dump(regressor, os.path.join(models_dir, 'XGBoostRegressor.joblib'))
        joblib.dump(classifier, os.path.join(models_dir, 'XGBoostClassifier.joblib'))
        joblib.dump(label_encoder, os.path.join(models_dir, 'label_encoder.joblib'))
        joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
        logging.info(f"Models and preprocessors saved in '{models_dir}' directory.")
    except Exception as e:
        logging.error(f"Error saving models: {e}")
        sys.exit(1)

def main():
    """
    The main function orchestrates the data fetching, processing, model training, and saving.
    """
    setup_logging()

    # Define the teams to filter
    teams_to_filter = ['LAL', 'BOS']

    # Fetch all players for the season
    players_all = fetch_players()

    # Filter players by team
    filtered_players = filter_players(players_all, teams_to_filter)

    # Fetch game logs for filtered players
    all_game_logs = fetch_game_logs(filtered_players)

    if all_game_logs.empty:
        logging.error("No game logs were fetched. Exiting the program.")
        sys.exit(1)

    # Process game logs with feature engineering
    # Ensure that 'players_df' passed to engineer_features contains necessary information
    processed_data, label_encoder = engineer_features(all_game_logs, filtered_players)

    if processed_data.empty or label_encoder is None:
        logging.error("Feature engineering failed. Exiting the program.")
        sys.exit(1)

    # Train models and save them
    train_and_save_models(processed_data, label_encoder)

if __name__ == "__main__":
    main()