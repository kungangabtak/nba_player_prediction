# src/knowledge_graph.py

import networkx as nx
import logging
import pandas as pd
import re
import community  # Import the Louvain method from python-louvain

def parse_matchup(matchup):
    """
    Parses the MATCHUP string to extract the opponent team's abbreviation.

    Parameters:
        matchup (str): The MATCHUP string (e.g., 'LAL vs. BOS' or 'LAL @ BOS').

    Returns:
        str or None: Opponent team abbreviation if successfully parsed, else None.
    """
    if not isinstance(matchup, str):
        return None
    # Regex to match formats like 'LAL vs. BOS' or 'LAL @ BOS'
    pattern = r'^\w{3}\s+(vs\.|@)\s+(\w{3})$'
    match = re.match(pattern, matchup.strip())
    if match:
        return match.group(2)
    return None

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

    # Define required columns for validation
    required_team_columns = {'id', 'full_name', 'abbreviation'}
    required_player_columns = {'id', 'full_name', 'team_id'}
    required_game_columns = {'GAME_ID', 'GAME_DATE', 'TEAM_ID', 'MATCHUP', 'PLAYER_ID'}

    # Validate team DataFrame
    if not required_team_columns.issubset(teams_df.columns):
        missing = required_team_columns - set(teams_df.columns)
        logging.error(f"Missing columns in teams_df: {missing}")
        raise KeyError(f"Missing columns in teams_df: {missing}")

    # Validate player DataFrame
    if not required_player_columns.issubset(players_df.columns):
        missing = required_player_columns - set(players_df.columns)
        logging.error(f"Missing columns in players_df: {missing}")
        raise KeyError(f"Missing columns in players_df: {missing}")

    # Validate game_logs DataFrame
    if not required_game_columns.issubset(game_logs_df.columns):
        missing = required_game_columns - set(game_logs_df.columns)
        logging.error(f"Missing columns in game_logs_df: {missing}")
        raise KeyError(f"Missing columns in game_logs_df: {missing}")

    # Convert IDs to strings
    teams_df = teams_df.astype({'id': str})
    players_df = players_df.astype({'id': str, 'team_id': str})
    game_logs_df = game_logs_df.astype({'GAME_ID': str, 'TEAM_ID': str, 'PLAYER_ID': str})

    # Add team nodes
    for index, row in teams_df.iterrows():
        team_id = row['id']
        team_name = row['full_name']
        team_abbr = row['abbreviation']
        if not all([team_id, team_name, team_abbr]):
            logging.warning(f"Incomplete team data at index {index}: {row}")
            continue
        KG.add_node(team_name, type='Team', abbreviation=team_abbr)
        logging.debug(f"Added Team node: {team_name}")
    
    # Add player nodes and 'plays_for' relationships
    for index, row in players_df.iterrows():
        player_id = row['id']
        player_name = row['full_name']
        team_id = row['team_id']

        if not all([player_id, player_name, team_id]):
            logging.warning(f"Incomplete player data at index {index}: {row}")
            continue

        # Retrieve team name using team_id
        team_row = teams_df[teams_df['id'] == team_id]
        if not team_row.empty:
            team_name = team_row.iloc[0]['full_name']
            # Handle potential duplicate player names by appending team name
            unique_player_name = f"{player_name} ({team_name})"
            KG.add_node(unique_player_name, type='Player')
            KG.add_edge(unique_player_name, team_name, relation='plays_for')
            logging.debug(f"Added Player node: {unique_player_name} and 'plays_for' edge to Team '{team_name}'")
        else:
            logging.warning(f"Team ID {team_id} not found in KG for Player {player_name} (ID: {player_id})")

    # Add game nodes and relationships
    for index, row in game_logs_df.iterrows():
        game_id = row['GAME_ID']
        game_date = row['GAME_DATE']
        team_id = row['TEAM_ID']
        matchup = row['MATCHUP']

        if not all([game_id, game_date, team_id, matchup]):
            logging.warning(f"Incomplete game log data at index {index}: {row}")
            continue

        # Add game node with date as attribute
        game_node_name = f"Game {game_id} ({game_date})"
        KG.add_node(game_node_name, type='Game', date=game_date)
        logging.debug(f"Added Game node: {game_node_name}")

        # Add venue node and relationship if available
        venue_name = row.get('ARENA_NAME')
        if venue_name:
            KG.add_node(venue_name, type='Venue', name=venue_name)
            KG.add_edge(game_node_name, venue_name, relation='played_at')
            logging.debug(f"Added Venue node: {venue_name} for Game '{game_node_name}'")

        # Parse matchup to find opponent team
        opponent_abbr = parse_matchup(matchup)
        if opponent_abbr:
            opponent_team = teams_df[teams_df['abbreviation'] == opponent_abbr]
            if not opponent_team.empty:
                opponent_team_name = opponent_team.iloc[0]['full_name']
                # Assuming 'team_name' is the home team; adjust if necessary
                team_row = teams_df[teams_df['id'] == team_id]
                if not team_row.empty:
                    team_name = team_row.iloc[0]['full_name']
                    KG.add_edge(team_name, game_node_name, relation='participated_in')
                    KG.add_edge(opponent_team_name, game_node_name, relation='opponent_in')
                    logging.debug(f"Added 'participated_in' edge from Team '{team_name}' to Game '{game_node_name}'")
                    logging.debug(f"Added 'opponent_in' edge from Opponent Team '{opponent_team_name}' to Game '{game_node_name}'")
                else:
                    logging.warning(f"Team ID {team_id} not found for Game '{game_node_name}'")
            else:
                logging.warning(f"Opponent abbreviation '{opponent_abbr}' not found in teams_df.")
        else:
            logging.warning(f"Failed to parse opponent abbreviation from MATCHUP '{matchup}'.")

        # Add performance stats as Game node attributes
        player_id = row['PLAYER_ID']
        if player_id:
            # Retrieve player name
            player_row = players_df[players_df['id'] == player_id]
            if not player_row.empty:
                player_name = player_row.iloc[0]['full_name']
                team_id = player_row.iloc[0]['team_id']
                team_row = teams_df[teams_df['id'] == team_id]
                if not team_row.empty:
                    team_name = team_row.iloc[0]['full_name']
                    unique_player_name = f"{player_name} ({team_name})"
                    if unique_player_name in KG.nodes:
                        stats = {
                            'Minutes_Played': row.get('Minutes_Played', 0),
                            'FG_Percentage': row.get('FG_Percentage', 0),
                            'FT_Percentage': row.get('FT_Percentage', 0),
                            'ThreeP_Percentage': row.get('ThreeP_Percentage', 0),
                            'Usage_Rate': row.get('Usage_Rate', 0),
                            'EFFICIENCY': row.get('EFFICIENCY', 0),
                            'PTS': row.get('PTS', 0),
                            'PTS_Rolling_Avg': row.get('PTS_Rolling_Avg', 0),
                            'REB_Rolling_Avg': row.get('REB_Rolling_Avg', 0),
                            'AST_Rolling_Avg': row.get('AST_Rolling_Avg', 0),
                            'HOME_GAME': row.get('HOME_GAME', 0),
                            'Opponent_PTS_Allowed': row.get('Opponent_PTS_Allowed', 0),
                            'FG3A_FGA_RATIO': row.get('FG3A_FGA_RATIO', 0),
                            'FT_FG_RATIO': row.get('FT_FG_RATIO', 0)
                        }
                        for stat_name, stat_value in stats.items():
                            KG.nodes[game_node_name][f'{unique_player_name}_{stat_name}'] = stat_value
                            logging.debug(f"Added stat '{stat_name}' with value {stat_value} for Player '{unique_player_name}' in Game '{game_node_name}'")
                    else:
                        logging.warning(f"Unique Player name '{unique_player_name}' not found in KG.")
            else:
                logging.warning(f"Player ID {player_id} not found in players_df.")

        # Add historical performance node if not present
        if player_id:
            player_row = players_df[players_df['id'] == player_id]
            if not player_row.empty:
                player_name = player_row.iloc[0]['full_name']
                team_id = player_row.iloc[0]['team_id']
                team_row = teams_df[teams_df['id'] == team_id]
                if not team_row.empty:
                    team_name = team_row.iloc[0]['full_name']
                    unique_player_name = f"{player_name} ({team_name})"
                    season_avg_node = f"{unique_player_name} Season Average"
                    if not KG.has_node(season_avg_node):
                        KG.add_node(season_avg_node, type='HistoricalPerformance', name='SeasonAverage')
                        KG.add_edge(unique_player_name, season_avg_node, relation='has_historical_performance')
                        logging.debug(f"Added historical performance node for Player '{unique_player_name}'")

        # Add matchup relationships directly without creating separate nodes
        if player_id and opponent_abbr:
            opponent_team = teams_df[teams_df['abbreviation'] == opponent_abbr]
            if not opponent_team.empty:
                opponent_team_name = opponent_team.iloc[0]['full_name']
                player_row = players_df[players_df['id'] == player_id]
                if not player_row.empty:
                    player_name = player_row.iloc[0]['full_name']
                    team_id = player_row.iloc[0]['team_id']
                    team_row = teams_df[teams_df['id'] == team_id]
                    if not team_row.empty:
                        team_name = team_row.iloc[0]['full_name']
                        unique_player_name = f"{player_name} ({team_name})"
                        KG.add_edge(unique_player_name, opponent_team_name, relation='matched_up_against', game_id=game_node_name)
                        logging.debug(f"Connected Player '{unique_player_name}' with Opponent Team '{opponent_team_name}' in Game '{game_node_name}'")

    # Detect communities in the graph using Louvain method
    logging.info("Applying Louvain clustering to detect communities.")
    partition = community.best_partition(KG)
    # Assign cluster labels to nodes
    for node, cluster_id in partition.items():
        KG.nodes[node]['cluster'] = cluster_id
    logging.info(f"Detected {max(partition.values()) + 1} communities.")

    logging.info("Knowledge Graph constructed with entities, relationships, and cluster labels.")
    return KG