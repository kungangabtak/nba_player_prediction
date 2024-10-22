# src/utils.py

from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from nba_api.stats.endpoints import commonplayerinfo, playergamelog, commonallplayers
from nba_api.stats.static import teams
import pickle
import os

def preprocess_data(df, threshold, label_encoder=None, scaler=None):
    features = df[['Minutes_Played', 'FG_Percentage', 'FT_Percentage', 'ThreeP_Percentage', 'Usage_Rate', 'PER', 'Opponent_Team']]
    
    if label_encoder is None:
        label_encoder = LabelEncoder()
        features['Opponent_Team'] = label_encoder.fit_transform(features['Opponent_Team'])
    else:
        features['Opponent_Team'] = label_encoder.transform(features['Opponent_Team'])
    
    target_classification = (df['Points'] > threshold).astype(int)
    
    if scaler is None:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
    else:
        features_scaled = scaler.transform(features)
    
    return features_scaled, target_classification, scaler, label_encoder

def get_player_id(full_name):
    players = commonallplayers.CommonAllPlayers(is_only_current_season=1).get_data_frames()[0]
    player = players[players['DISPLAY_FIRST_LAST'] == full_name]
    if not player.empty:
        return player.iloc[0]['PERSON_ID']
    return None

def get_full_team_name(abbreviation):
    team_dict = teams.get_teams()
    for team in team_dict:
        if team['abbreviation'] == abbreviation:
            return team['full_name']
    return None

def fetch_player_data(player_id, season, opponent_team=None):
    gamelog = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star='Regular Season'
    ).get_data_frames()[0]
    
    if opponent_team:
        gamelog = gamelog[gamelog['MATCHUP'].str.contains(opponent_team)]
    
    return gamelog

def get_opponent_teams(player_id, season):
    gamelog = fetch_player_data(player_id, season)
    if gamelog.empty:
        return []
    opponents = gamelog['MATCHUP'].str.split(' @ | vs ').str[1].unique().tolist()
    return opponents