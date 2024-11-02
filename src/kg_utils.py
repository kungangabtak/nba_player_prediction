# src/kg_utils.py

import networkx as nx
import logging

def extract_context_subgraph(KG, player_id, opponent_team_abbr, model_manager):
    """
    Extracts a subgraph relevant to the given player and opponent team.
    
    Parameters:
        KG (nx.Graph): The Knowledge Graph.
        player_id (str): The ID of the player.
        opponent_team_abbr (str): The abbreviation of the opponent team.
        model_manager (ModelManager): Instance of ModelManager to access utility methods.
    
    Returns:
        nx.Graph: Subgraph containing relevant context.
    """
    if not KG.has_node(player_id):
        logging.warning(f"Player ID {player_id} not found in KG.")
        return nx.Graph()  # Return an empty graph if player not found
    
    nodes = set()
    nodes.add(player_id)
    
    # Add the player's team node
    for neighbor in KG.neighbors(player_id):
        edge_data = KG.get_edge_data(player_id, neighbor)
        if edge_data and edge_data.get('relation') == 'plays_for':
            nodes.add(neighbor)
            logging.debug(f"Player ID {player_id} plays for Team ID {neighbor}.")
            break  # Assuming a player plays for only one team
    
    # Add the opponent team node
    opponent_team_id = model_manager.get_team_id_from_abbr(opponent_team_abbr)
    if opponent_team_id:
        if KG.has_node(opponent_team_id):
            nodes.add(opponent_team_id)
            logging.debug(f"Added Opponent Team ID {opponent_team_id} to subgraph.")
        else:
            logging.warning(f"Opponent Team ID {opponent_team_id} not found in KG.")
    else:
        logging.warning(f"Opponent team abbreviation '{opponent_team_abbr}' could not be resolved to a team ID.")
    
    # Extract the subgraph
    subgraph = KG.subgraph(nodes).copy()
    logging.debug(f"Extracted context subgraph with nodes: {nodes}")
    
    return subgraph