# main.py

from src import utils, model_training, data_collection, feature_engineering, data_preprocessing
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    season = '2022-23'
    logging.info(f"Starting data collection for the {season} season.")
    
    players = data_collection.get_all_players(season)
    if players.empty:
        logging.error("No players retrieved. Exiting the program.")
        return
    
    all_data = pd.DataFrame()
    skipped_players = []

    for _, player in players.iterrows():
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
            gamelog['PLAYER_NAME'] = player_name
            all_data = pd.concat([all_data, gamelog], ignore_index=True)
            logging.info(f"Data appended for player: {player_name}")
        except KeyError as e:
            logging.error(f"Key error while processing data for player: {player_name} - Missing column: {e}")
            skipped_players.append(player_name)
        except Exception as e:
            logging.error(f"Unexpected error for player: {player_name} - {e}")
            skipped_players.append(player_name)
    
    if all_data.empty:
        logging.error("No data collected after processing all players. Exiting the program.")
        return
    
    logging.info("Cleaning and preprocessing the collected data.")
    all_data = data_preprocessing.clean_data(all_data)
    
    logging.info("Starting model training.")
    reg_model, clf_model, scaler, label_encoder = model_training.build_and_train_models(all_data, threshold=20)
    logging.info("Models trained and saved successfully.")
    
    if skipped_players:
        logging.info(f"Skipped {len(skipped_players)} players due to errors:")
        for player in skipped_players:
            logging.info(f"- {player}")

if __name__ == "__main__":
    main()