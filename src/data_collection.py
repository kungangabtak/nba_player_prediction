# src/data_collection.py

from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog
import pandas as pd
import logging

def get_all_players(season='2023-24'):
    """
    Fetches all players for the given season.

    Parameters:
        season (str): NBA season (e.g., '2023-24').

    Returns:
        pd.DataFrame: DataFrame of all players.
    """
    try:
        players_list = players.get_players()
        players_df = pd.DataFrame(players_list)
        logging.info(f"Fetched {len(players_df)} players for the season {season}.")
        return players_df
    except Exception as e:
        logging.error(f"Error fetching players: {e}")
        return pd.DataFrame()

def get_player_data(player_id, season='2023-24'):
    """
    Fetches game logs for a specific player and season.

    Parameters:
        player_id (int): NBA player ID.
        season (str): NBA season (e.g., '2023-24').

    Returns:
        pd.DataFrame: DataFrame of game logs.
    """
    try:
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star='Regular Season',
            timeout=30
        ).get_data_frames()[0]
        logging.info(f"Fetched {len(gamelog)} game logs for player ID {player_id} in season {season}.")
        return gamelog
    except Exception as e:
        logging.error(f"Error fetching game logs for player ID {player_id}: {e}")
        return pd.DataFrame()