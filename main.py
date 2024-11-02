# main.py

import logging
import sys
import os
import pandas as pd
import time
import joblib

# Import functions from src modules
from src.data_collection import get_all_players, get_all_player_game_logs, get_team_data
from src.knowledge_graph import build_kg
from src.model_training import build_and_train_models


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


def main():
    """
    The main function orchestrates the data fetching, KG building, model training, and saving.
    """
    setup_logging()

    # Set the season to a completed season with available data
    season = '2023-24'  # Use a completed season for available data

    # Fetch all active players and teams
    players_df = get_all_players(season=season)
    teams_df = get_team_data()

    if players_df.empty:
        logging.error("No active players fetched. Exiting the program.")
        sys.exit(1)

    if teams_df.empty:
        logging.error("No teams fetched. Exiting the program.")
        sys.exit(1)

    # Fetch game logs for all players in a single request
    all_game_logs = get_all_player_game_logs(season=season)

    if all_game_logs.empty:
        logging.error("No game logs were fetched. Exiting the program.")
        sys.exit(1)

    # Build Knowledge Graph with game logs
    KG = build_kg(players_df, teams_df, all_game_logs)

    # Proceed with data preprocessing and model training
    result = build_and_train_models(all_game_logs)

    if result is None:
        logging.error("Model training failed. Exiting the program.")
        sys.exit(1)
    else:
        best_reg_pipeline, best_clf_pipeline = result

    # Models and pipelines are already saved within build_and_train_models
    logging.info("All models and pipelines have been saved successfully.")

if __name__ == "__main__":
    main()