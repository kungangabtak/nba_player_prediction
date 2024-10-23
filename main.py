# main.py

import argparse
import datetime
from tqdm import tqdm
from src import utils, model_training, data_collection, feature_engineering, data_preprocessing
import pandas as pd
import logging
import time
from nba_api.stats.static import teams

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='NBA Player Performance Prediction Model')
    parser.add_argument('--season', type=str, help='Season in format YYYY-YY', default=None)
    parser.add_argument('--teams', type=str, nargs='+', help='Team abbreviations to include', default=['LAL', 'BOS'])
    args = parser.parse_args()
    
    # Dynamic Season Input
    if args.season:
        season = args.season
    else:
        current_year = datetime.datetime.now().year
        current_month = datetime.datetime.now().month
        if current_month >= 10:  # NBA season typically starts in October
            # Set to current season (e.g., 2024-25)
            season = f"{current_year}-{str(current_year+1)[-2:]}"
        else:
            # Set to previous season (e.g., 2023-24)
            season = f"{current_year-1}-{str(current_year)[-2:]}"
    
    teams_to_include = args.teams
    logging.info(f"Filtering players from teams: {teams_to_include}")
    
    logging.info(f"Starting data collection for the {season} season.")
    
    players = data_collection.get_all_players(season)
    if players.empty:
        logging.error("No players retrieved for the specified season. Exiting the program.")
        return
    
    # Verify the available columns and unique team abbreviations
    logging.info(f"Available columns in players DataFrame: {players.columns.tolist()}")
    
    if 'TEAM_ABBREVIATION' not in players.columns:
        logging.error("Column 'TEAM_ABBREVIATION' not found in players DataFrame.")
        logging.info("Available team abbreviations:")
        for team in teams.get_teams():
            logging.info(f"{team['abbreviation']} - {team['full_name']}")
        return
    
    unique_teams = players['TEAM_ABBREVIATION'].unique()
    logging.info(f"Unique TEAM_ABBREVIATION values: {unique_teams}")
    
    # Filter players belonging to the specified teams
    players_filtered = players[players['TEAM_ABBREVIATION'].isin(teams_to_include)]
    logging.info(f"Number of players after filtering: {len(players_filtered)}")
    
    if players_filtered.empty:
        logging.error(f"No players found for the specified teams: {teams_to_include}")
        return
    
    all_data = pd.DataFrame()
    skipped_players = []
    
    player_ids = players_filtered['PERSON_ID'].dropna().unique().tolist()
    gamelogs = data_collection.get_all_player_data(player_ids, season)
    
    for player_id, gamelog in tqdm(zip(player_ids, gamelogs), total=len(player_ids), desc="Processing Gamelogs"):
        if gamelog.empty:
            logging.warning(f"No game log data for player ID {player_id}. Skipping.")
            skipped_players.append(player_id)
            continue
        
        # Retrieve player name using player_id
        player_name_series = players_filtered.loc[players_filtered['PERSON_ID'] == player_id, 'DISPLAY_FIRST_LAST']
        if player_name_series.empty:
            logging.warning(f"Player name not found for player ID {player_id}. Skipping.")
            skipped_players.append(player_id)
            continue
        player_name = player_name_series.values[0]
        
        # Log the columns of the gamelog for debugging
        logging.debug(f"Gamelog columns for player {player_name} (ID: {player_id}): {gamelog.columns.tolist()}")
        
        # Check if 'PTS' exists in gamelog
        if 'PTS' not in gamelog.columns:
            logging.warning(f"Player {player_name} (ID: {player_id}) gamelog is missing 'PTS' column. Available columns: {gamelog.columns.tolist()}. Skipping.")
            skipped_players.append(player_name)
            continue
        
        try:
            gamelog = feature_engineering.engineer_features(gamelog)
            # No need to rename 'PTS' to 'PTS' again since it's already handled in feature engineering
            
            # Verify 'PTS' exists after feature engineering
            if 'PTS' not in gamelog.columns:
                logging.warning(f"After feature engineering, 'PTS' column missing for player: {player_name}. Available columns: {gamelog.columns.tolist()}. Skipping.")
                skipped_players.append(player_name)
                continue
            
            # Drop all non-numeric columns
            non_numeric_cols = gamelog.select_dtypes(exclude=['number']).columns.tolist()
            logging.debug(f"Dropping non-numeric columns: {non_numeric_cols}")
            gamelog = gamelog.drop(columns=non_numeric_cols, errors='ignore')
            
            all_data = pd.concat([all_data, gamelog], ignore_index=True)
            logging.info(f"Data appended for player: {player_name}")
        except KeyError as e:
            logging.error(f"Key error while processing data for player: {player_name} - Missing column: {e}")
            skipped_players.append(player_name)
        except Exception as e:
            logging.error(f"Unexpected error for player: {player_name} - {e}")
            skipped_players.append(player_name)
        
        # Optional: Sleep to respect API rate limits
        time.sleep(0.5)  # Adjust delay as needed
    
    if all_data.empty:
        logging.error("No data collected after processing filtered players. Exiting the program.")
        return
    
    logging.info("Sample data before cleaning:")
    logging.info(all_data.head())
    logging.info(f"Data types before cleaning:\n{all_data.dtypes}")
    
    logging.info("Cleaning and preprocessing the collected data.")
    all_data = data_preprocessing.clean_data(all_data)
    logging.info("Sample data after cleaning:")
    logging.info(all_data.head())
    
    # Verify that no non-numeric columns remain
    non_numeric_after = all_data.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric_after:
        logging.error(f"Non-numeric columns still present after cleaning: {non_numeric_after}")
    else:
        logging.info("No non-numeric columns present after cleaning.")
    
    logging.info("Starting model training.")
    reg_model, clf_model, scaler, label_encoder = model_training.build_and_train_models(all_data, threshold=20)
    logging.info("Models trained and saved successfully.")
    
    if skipped_players:
        logging.info(f"Skipped {len(skipped_players)} players due to errors:")
        for player in skipped_players:
            logging.info(f"- {player}")

if __name__ == "__main__":
    main()