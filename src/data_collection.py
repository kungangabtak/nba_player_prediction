# src/data_collection.py

from nba_api.stats.endpoints import commonallplayers, playergamelog
import pandas as pd
import logging
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from functools import lru_cache
import asyncio
import aiohttp

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_players_data(season):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
                      ' Chrome/58.0.3029.110 Safari/537.3',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://stats.nba.com/',
        'Origin': 'https://stats.nba.com',
        'Connection': 'keep-alive',
    }
    response = requests.get(
        'https://stats.nba.com/stats/commonallplayers',
        params={
            'IsOnlyCurrentSeason': '0',
            'LeagueID': '00',
            'Season': season
        },
        headers=headers,
        timeout=60
    )
    response.raise_for_status()
    players_data = commonallplayers.CommonAllPlayers(season=season, is_only_current_season=0).get_normalized_dict()
    players_df = pd.DataFrame(players_data['CommonAllPlayers'])
    return players_df

def get_all_players(season='2023-24'):
    try:
        players_df = fetch_players_data(season)
        logging.info(f"Retrieved {len(players_df)} players for the {season} season.")
        return players_df
    except Exception as e:
        logging.error(f"Error fetching all players: {e}")
        return pd.DataFrame()

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_player_gamelog(player_id, season):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
                      ' Chrome/58.0.3029.110 Safari/537.3',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://stats.nba.com/',
        'Origin': 'https://stats.nba.com',
        'Connection': 'keep-alive',
    }
    gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season, headers=headers)
    gamelog_df = gamelog.get_data_frames()[0]
    return gamelog_df

async def fetch_player_gamelog_async(session, player_id, season):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
                      ' Chrome/58.0.3029.110 Safari/537.3',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://stats.nba.com/',
        'Origin': 'https://stats.nba.com',
        'Connection': 'keep-alive',
    }
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season, headers=headers)
        gamelog_df = gamelog.get_data_frames()[0]
        logging.info(f"Successfully fetched data for player ID {player_id}.")
        return gamelog_df
    except Exception as e:
        logging.error(f"Error fetching data for player ID {player_id}: {e}")
        return pd.DataFrame()

async def fetch_all_gamelogs(player_ids, season):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_player_gamelog_async(session, pid, season) for pid in player_ids]
        return await asyncio.gather(*tasks)

def get_all_player_data(player_ids, season='2023-24'):
    return asyncio.run(fetch_all_gamelogs(player_ids, season))

@lru_cache(maxsize=128)
def get_player_data(player_id, season='2023-24'):
    try:
        gamelog_df = fetch_player_gamelog(player_id, season)
        logging.info(f"Successfully fetched data for player ID {player_id}.")
        return gamelog_df
    except Exception as e:
        logging.error(f"Error fetching data for player ID {player_id}: {e}")
        return pd.DataFrame()