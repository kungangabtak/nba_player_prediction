# src/utils.py

import pandas as pd
import logging
from nba_api.stats.static import teams

def get_player_id(player_name, players_df):
    """
    Retrieves the player ID for a given player name.

    Parameters:
        player_name (str): Full name of the player (e.g., 'LeBron James').
        players_df (pd.DataFrame): DataFrame containing player information.

    Returns:
        int or None: Player ID if found, else None.
    """
    try:
        player_row = players_df[players_df['full_name'].str.lower() == player_name.lower()]
        if not player_row.empty:
            player_id = player_row.iloc[0]['id']
            logging.info(f"Found player ID {player_id} for player '{player_name}'.")
            return player_id
        else:
            logging.warning(f"Player '{player_name}' not found in players DataFrame.")
            return None
    except Exception as e:
        logging.error(f"Error retrieving player ID for '{player_name}': {e}")
        return None

def get_opponent_teams(player_id, season, gamelogs):
    """
    Retrieves a list of unique opponent teams the player has faced in the season.

    Parameters:
        player_id (int): NBA player ID.
        season (str): NBA season (e.g., '2023-24').
        gamelogs (pd.DataFrame): DataFrame of player's game logs.

    Returns:
        list: List of opponent team abbreviations.
    """
    try:
        # Extract opponent team from MATCHUP column
        # MATCHUP format: 'TEAM vs. OPPONENT' or 'TEAM @ OPPONENT'
        opponents = gamelogs['MATCHUP'].apply(lambda x: x.split(' ')[-1] if pd.notnull(x) else None)
        opponent_abbr = opponents.dropna().unique().tolist()
        logging.info(f"Player ID {player_id} has faced opponents: {opponent_abbr}")
        return opponent_abbr
    except Exception as e:
        logging.error(f"Error retrieving opponent teams for player ID {player_id}: {e}")
        return []

def get_full_team_name(team_abbr):
    """
    Converts a team abbreviation to its full team name.

    Parameters:
        team_abbr (str): Team abbreviation (e.g., 'LAL').

    Returns:
        str or None: Full team name (e.g., 'Los Angeles Lakers') if found, else None.
    """
    try:
        team = teams.find_team_by_abbreviation(team_abbr.upper())
        if team:
            full_name = team['full_name']
            logging.info(f"Converted abbreviation '{team_abbr}' to full team name '{full_name}'.")
            return full_name
        else:
            logging.warning(f"No team found with abbreviation '{team_abbr}'.")
            return None
    except Exception as e:
        logging.error(f"Error converting team abbreviation '{team_abbr}': {e}")
        return None