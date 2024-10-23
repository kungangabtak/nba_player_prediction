# main.py

from src import utils, model_training, data_collection, feature_engineering, data_preprocessing
import pandas as pd
import logging
import time
from nba_api.stats.static import teams

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    season = '2022-23'
    logging.info(f"Starting data collection for the {season} season.")
    
    players = data_collection.get_all_players(season)
    if players.empty:
        logging.error("No players retrieved. Exiting the program.")
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
    
    # Specify the two teams you want to analyze
    teams_to_include = ['LAL', 'BOS']  # Example: Los Angeles Lakers and Boston Celtics
    logging.info(f"Filtering players from teams: {teams_to_include}")
    
    # Filter players belonging to the specified teams
    players_filtered = players[players['TEAM_ABBREVIATION'].isin(teams_to_include)]
    logging.info(f"Number of players after filtering: {len(players_filtered)}")
    
    if players_filtered.empty:
        logging.error(f"No players found for the specified teams: {teams_to_include}")
        return
    
    all_data = pd.DataFrame()
    skipped_players = []
    
    for _, player in players_filtered.iterrows():
        player_id = player.get('PERSON_ID')
        player_name = player.get('DISPLAY_FIRST_LAST', 'Unknown')
        
        if pd.isna(player_id):
            logging.warning(f"Skipping player with missing PERSON_ID: {player_name}")
            skipped_players.append(player_name)
            continue
        
        gamelog = data_collection.get_player_data(player_id, season)
        if gamelog.empty:
            logging.warning(f"No game log data for player: {player_name} (ID: {player_id}). Skipping.")
            skipped_players.append(player_name)
            continue
        
        try:
            gamelog = feature_engineering.engineer_features(gamelog)
            gamelog = gamelog.rename(columns={
                'PTS': 'Points',
                'REB': 'REB',
                'AST': 'AST',
                'STL': 'STL',
                'BLK': 'BLK',
                'FGA': 'FGA',
                'FGM': 'FGM',
                'FTA': 'FTA',
                'FTM': 'FTM',
                'TOV': 'TOV',
                'TEAM_ABBREVIATION': 'Opponent_Team'
            })
            
            # Drop all non-numeric columns
            non_numeric_cols = gamelog.select_dtypes(exclude=['number']).columns.tolist()
            logging.info(f"Dropping non-numeric columns: {non_numeric_cols}")
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
    
    logging.info("Starting model training.")
    reg_model, clf_model, scaler, label_encoder = model_training.build_and_train_models(all_data, threshold=20)
    logging.info("Models trained and saved successfully.")
    
    if skipped_players:
        logging.info(f"Skipped {len(skipped_players)} players due to errors:")
        for player in skipped_players:
            logging.info(f"- {player}")

if __name__ == "__main__":
    main()