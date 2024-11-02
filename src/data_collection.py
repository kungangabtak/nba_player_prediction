# src/data_collection.py

from functools import lru_cache
from nba_api.stats.static import teams
from nba_api.stats.endpoints import commonteamroster, PlayerGameLog, LeagueGameLog  # Updated import
import pandas as pd
import logging
import time

def validate_season_format(season):
    if not isinstance(season, str) or '-' not in season:
        logging.error(f"Invalid season format: {season}. Expected format 'YYYY-YY'.")
        return False
    return True

@lru_cache(maxsize=None)
def get_all_players(season='2023-24'):
    """
    Fetches all active players with valid team assignments for the given season.

    Parameters:
        season (str): NBA season (e.g., '2023-24').

    Returns:
        pd.DataFrame: DataFrame of active players with valid team assignments.
    """
    if not validate_season_format(season):
        return pd.DataFrame()

    try:
        # Fetch team data and get valid team IDs
        teams_list = teams.get_teams()
        teams_df = pd.DataFrame(teams_list)
        teams_df['id'] = teams_df['id'].astype(str)
        valid_team_ids = teams_df['id'].tolist()

        all_players_list = []

        for team_id in valid_team_ids:
            try:
                # Fetch team roster
                roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season).get_data_frames()[0]
                # Add 'team_id' to the roster DataFrame
                roster['team_id'] = team_id
                all_players_list.append(roster)
                logging.info(f"Fetched roster for team ID {team_id}.")
                time.sleep(0.5)  # Reduced sleep time to prevent rate limiting
            except Exception as e:
                logging.error(f"Error fetching roster for team ID {team_id}: {e}")
                time.sleep(0.5)  # Sleep even on error to be cautious with rate limits

        if not all_players_list:
            logging.error("No players fetched from team rosters.")
            return pd.DataFrame()

        players_df = pd.concat(all_players_list, ignore_index=True)
        # Rename columns for consistency
        players_df = players_df.rename(columns={'PLAYER_ID': 'id', 'PLAYER': 'full_name'})
        # Ensure 'team_id' and 'id' are of type string
        players_df['team_id'] = players_df['team_id'].astype(str)
        players_df['id'] = players_df['id'].astype(str)

        logging.info(f"Fetched {len(players_df)} players from team rosters for season {season}.")

        return players_df
    except Exception as e:
        logging.error(f"Error fetching active players: {e}")
        return pd.DataFrame()

@lru_cache(maxsize=None)
def get_player_game_logs(player_name, season='2023-24'):
    """
    Fetches game logs for a specific player in a given season.

    Parameters:
        player_name (str): Full name of the player (e.g., 'LeBron James').
        season (str): NBA season (e.g., '2023-24').

    Returns:
        pd.DataFrame: DataFrame of the player's game logs.
    """
    from src.utils import get_player_id  # Import here to prevent circular imports

    # Fetch all players to get player ID
    players_df = get_all_players(season=season)
    player_id = get_player_id(player_name, players_df=players_df)

    if player_id is None:
        logging.error(f"Cannot fetch game logs for player '{player_name}' because player ID not found.")
        return pd.DataFrame()

    try:
        # Fetch game logs for the player using PlayerGameLog
        gamelog = PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
        if gamelog.empty:
            logging.error(f"No game logs found for player '{player_name}' in season {season}.")
            return pd.DataFrame()
        logging.info(f"Fetched {len(gamelog)} game logs for player '{player_name}' in season {season}.")
        return gamelog
    except Exception as e:
        logging.error(f"Error fetching game logs for player '{player_name}': {e}")
        return pd.DataFrame()

@lru_cache(maxsize=None)
def get_all_player_game_logs(season='2023-24'):
    """
    Fetches game logs for all players in a given season.

    Parameters:
        season (str): NBA season (e.g., '2023-24').

    Returns:
        pd.DataFrame: DataFrame of all players' game logs.
    """
    if not validate_season_format(season):
        return pd.DataFrame()

    try:
        # Fetch all player game logs for the season using LeagueGameLog
        gamelog = LeagueGameLog(season=season, player_or_team_abbreviation='P').get_data_frames()[0]
        logging.info(f"Fetched {len(gamelog)} total game logs for all players in season {season}.")
        return gamelog
    except Exception as e:
        logging.error(f"Error fetching all player game logs for season {season}: {e}")
        return pd.DataFrame()

@lru_cache(maxsize=None)
def get_team_data():
    """
    Fetches all NBA team data.

    Returns:
        pd.DataFrame: DataFrame of NBA teams.
    """
    try:
        teams_list = teams.get_teams()
        teams_df = pd.DataFrame(teams_list)
        teams_df['id'] = teams_df['id'].astype(str)
        logging.info(f"Fetched {len(teams_df)} teams.")
        return teams_df
    except Exception as e:
        logging.error(f"Error fetching team data: {e}")
        return pd.DataFrame()