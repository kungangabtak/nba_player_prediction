# src/utils.py

from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from nba_api.stats.endpoints import commonplayerinfo, playergamelog, commonallplayers
from nba_api.stats.static import teams
import joblib
import os
import logging
from functools import lru_cache

def get_player_id(full_name, players_df):
    """
    Retrieves the PERSON_ID for a given player's full name.

    Parameters:
        full_name (str): The full name of the player (e.g., "LeBron James").
        players_df (pd.DataFrame): DataFrame containing player information.

    Returns:
        int or None: The PERSON_ID if found, else None.
    """
    matched_players = players_df[players_df['DISPLAY_FIRST_LAST'].str.lower() == full_name.lower()]
    if not matched_players.empty:
        if len(matched_players) > 1:
            logging.warning(f"Multiple players found with the name '{full_name}'. Selecting the first match.")
        return matched_players.iloc[0]['PERSON_ID']
    return None

@lru_cache(maxsize=1)
def load_team_dict():
    return teams.get_teams()

def get_full_team_name(abbreviation):
    team_dict = teams.get_teams()
    for team in team_dict:
        if team['abbreviation'] == abbreviation:
            return team['full_name']
    return None
