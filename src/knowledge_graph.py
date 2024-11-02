# src/knowledge_graph.py

import networkx as nx
import logging
import pandas as pd

def build_kg(players_df, teams_df, game_logs_df):
    """
    Constructs a Knowledge Graph from player, team, and game data.

    Parameters:
        players_df (pd.DataFrame): DataFrame containing player information.
        teams_df (pd.DataFrame): DataFrame containing team information.
        game_logs_df (pd.DataFrame): DataFrame containing game logs.

    Returns:
        nx.Graph: A NetworkX graph representing the KG.
    """
    KG = nx.Graph()

    # Ensure team and player IDs are strings
    teams_df['id'] = teams_df['id'].astype(str)
    players_df['team_id'] = players_df['team_id'].astype(str)
    players_df['id'] = players_df['id'].astype(str)

    # Add team nodes
    for index, row in teams_df.iterrows():
        team_id = row['id']
        team_name = row['full_name']
        team_abbr = row['abbreviation']
        KG.add_node(team_id, type='Team', name=team_name, abbreviation=team_abbr)
        logging.debug(f"Added Team node: {team_name} (ID: {team_id})")

    # Add player nodes and 'plays_for' relationships
    for index, row in players_df.iterrows():
        player_id = row['id']
        player_name = row['full_name']
        team_id = row['team_id']

        if team_id in KG.nodes:
            KG.add_node(player_id, type='Player', name=player_name)
            KG.add_edge(player_id, team_id, relation='plays_for')
            logging.debug(f"Added Player node: {player_name} (ID: {player_id}) and 'plays_for' edge to Team ID {team_id}")
        else:
            logging.warning(f"Team ID {team_id} not found in KG for Player {player_name} (ID: {player_id})")

    # Add game nodes and relationships
    for index, row in game_logs_df.iterrows():
        game_id = row['GAME_ID']
        game_date = row['GAME_DATE']
        team_id = row['TEAM_ID']
        matchup = row['MATCHUP']

        if pd.notnull(matchup):
            # Assuming 'MATCHUP' format: 'TEAM_ABBR vs. OPPONENT' or 'TEAM_ABBR @ OPPONENT'
            parts = matchup.split(' ')
            if len(parts) >= 3:
                opponent_abbr = parts[-1]
                opponent_team = teams_df[teams_df['abbreviation'] == opponent_abbr]
                if not opponent_team.empty:
                    opponent_id = opponent_team.iloc[0]['id']
                    KG.add_node(game_id, type='Game', date=game_date)
                    KG.add_edge(team_id, game_id, relation='participated_in')
                    KG.add_edge(opponent_id, game_id, relation='opponent_in')
                    logging.debug(f"Added Game node: {game_id} on {game_date} with Team ID {team_id} vs Opponent ID {opponent_id}")

    logging.info("Knowledge Graph constructed with entities and relationships.")
    return KG