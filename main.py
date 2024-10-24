# main.py

import logging
import sys
from nba_api.stats.endpoints import commonallplayers, leaguegamelog
import pandas as pd
from src.feature_engineering import engineer_features
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
        level=logging.INFO,  # Set to DEBUG for more detailed logs
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("training.log")
        ]
    )

def fetch_players(season='2023-24'):
    """
    Fetches all players for the specified NBA season.

    Parameters:
        season (str): The NBA season (e.g., '2022-23').

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

def fetch_all_game_logs(season='2023-24'):
    """
    Fetches game logs for all players in the specified NBA season.

    Parameters:
        season (str): The NBA season (e.g., '2022-23').

    Returns:
        pd.DataFrame: DataFrame containing game logs for all players.
    """
    logging.info(f"Fetching game logs for all players for the {season} season.")
    try:
        gamelog = leaguegamelog.LeagueGameLog(
            season=season,
            player_or_team_abbreviation='P',  # 'P' for players, 'T' for teams
            season_type_all_star='Regular Season'
        ).get_data_frames()[0]
        logging.info(f"Retrieved game logs for {len(gamelog)} games.")
        logging.info(f"Columns in all_game_logs: {gamelog.columns.tolist()}")
        return gamelog
    except Exception as e:
        logging.error(f"Error fetching league game logs: {e}")
        sys.exit(1)

def train_and_save_models(processed_data, label_encoder, models_dir='models'):
    """
    Trains machine learning models and saves them along with encoders and scalers.

    Parameters:
        processed_data (pd.DataFrame): DataFrame containing processed features and target.
        label_encoder (LabelEncoder): Fitted LabelEncoder for Opponent_Team.
        models_dir (str): Directory to save the trained models and preprocessors.
    """
    if processed_data.empty:
        logging.error("No data collected after processing players. Exiting the program.")
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

    # Set the season to a completed season with available data
    season = '2022-23'

    # Fetch all players for the season
    players_all = fetch_players(season=season)

    if players_all.empty:
        logging.error("No players fetched. Exiting the program.")
        sys.exit(1)

    # Fetch game logs for all players using LeagueGameLog endpoint
    all_game_logs = fetch_all_game_logs(season=season)

    if all_game_logs.empty:
        logging.error("No game logs were fetched. Exiting the program.")
        sys.exit(1)

    # Check if 'TEAM_ABBREVIATION' exists in all_game_logs
    if 'TEAM_ABBREVIATION' not in all_game_logs.columns:
        logging.error("'TEAM_ABBREVIATION' column is missing from game logs.")
        sys.exit(1)

    # Process game logs with feature engineering
    processed_data, label_encoder = engineer_features(all_game_logs, players_all)

    if processed_data.empty or label_encoder is None:
        logging.error("Feature engineering failed. Exiting the program.")
        sys.exit(1)

    # Train models and save them
    train_and_save_models(processed_data, label_encoder)

if __name__ == "__main__":
    main()