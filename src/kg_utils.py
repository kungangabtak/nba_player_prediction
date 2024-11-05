# src/kg_utils.py

import networkx as nx
import logging

def extract_context_subgraph(KG, player_name, opponent_team_abbr, model_manager, depth=2):
    """
    Extracts a subgraph relevant to the given player and opponent team up to a specified depth,
    filtering nodes by their cluster to enhance efficiency.

    Parameters:
        KG (nx.Graph): The Knowledge Graph.
        player_name (str): The full name of the player (e.g., 'LeBron James (Los Angeles Lakers)').
        opponent_team_abbr (str): The abbreviation of the opponent team (e.g., 'BOS').
        model_manager (ModelManager): Instance of ModelManager to access utility methods.
        depth (int): The depth of relationships to traverse.

    Returns:
        nx.Graph: Subgraph containing relevant context within the same cluster.
    """
    if not KG.has_node(player_name):
        logging.warning(f"Player '{player_name}' not found in KG.")
        return nx.Graph()  # Return an empty graph if player not found

    # Retrieve the cluster of the player
    player_cluster = KG.nodes[player_name].get('cluster')
    if player_cluster is None:
        logging.warning(f"Player '{player_name}' does not have a cluster assigned.")
        return nx.Graph()

    logging.info(f"Player '{player_name}' is in cluster {player_cluster}.")

    # Filter nodes by the same cluster
    cluster_nodes = [n for n, attr in KG.nodes(data=True) if attr.get('cluster') == player_cluster]
    subgraph = KG.subgraph(cluster_nodes).copy()
    logging.info(f"Subgraph created with {subgraph.number_of_nodes()} nodes from cluster {player_cluster}.")

    # Further extract context up to the specified depth
    try:
        # Use NetworkX's ego_graph to include neighbors up to a certain depth for the player
        player_subgraph = nx.ego_graph(subgraph, player_name, radius=depth, center=True, undirected=True)

        # Resolve opponent team name using abbreviation
        opponent_team_name = model_manager.get_team_name_from_abbr(opponent_team_abbr)
        if opponent_team_name and subgraph.has_node(opponent_team_name):
            opponent_subgraph = nx.ego_graph(subgraph, opponent_team_name, radius=depth, center=True, undirected=True)
            # Combine player and opponent subgraphs
            combined_subgraph = nx.compose(player_subgraph, opponent_subgraph)
            logging.info(f"Combined subgraph includes player and opponent within cluster {player_cluster}.")
        else:
            combined_subgraph = player_subgraph
            if not opponent_team_abbr:
                logging.warning(f"Opponent team abbreviation '{opponent_team_abbr}' is invalid.")
            else:
                logging.warning(f"Opponent team '{opponent_team_abbr}' not found in KG within cluster {player_cluster}.")

        logging.debug(f"Extracted context subgraph with {combined_subgraph.number_of_nodes()} nodes.")
        return combined_subgraph
    except Exception as e:
        logging.error(f"Error extracting context subgraph: {e}")
        return nx.Graph()