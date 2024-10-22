# src/data_collection.py

from nba_api.stats.endpoints import commonallplayers, playergamelog
import pandas as pd
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_all_players(season='2022-23'):
    try:
        players = commonallplayers.CommonAllPlayers(is_only_current_season=1).get_data_frames()[0]
        logging.info(f"Retrieved {len(players)} players for the {season} season.")
        return players
    except Exception as e:
        logging.error(f"Error fetching all players: {e}")
        return pd.DataFrame()

def get_player_data(player_id, season='2022-23', opponent_team=None, retries=3, delay=5):
    attempt = 0
    while attempt < retries:
        try:
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star='Regular Season'
            ).get_data_frames()[0]
            
            if opponent_team:
                gamelog = gamelog[gamelog['MATCHUP'].str.contains(opponent_team)]
            
            logging.info(f"Successfully fetched data for player ID {player_id}.")
            return gamelog
        except Exception as e:
            attempt += 1
            logging.warning(f"Attempt {attempt} - Error fetching data for player ID {player_id}: {e}")
            if attempt < retries:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logging.error(f"Failed to fetch data for player ID {player_id} after {retries} attempts.")
                return pd.DataFrame()