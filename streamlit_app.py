# streamlit_app.py

import streamlit as st
import logging
from src import prediction, utils, data_collection
from src.kg_utils import extract_context_subgraph
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@st.cache_resource
def load_models(season='2023-24'):
    """
    Loads the ModelManager, including models and the Knowledge Graph.

    Parameters:
        season (str): NBA season to build the Knowledge Graph for.

    Returns:
        ModelManager: An instance of the ModelManager class with loaded models and KG.
    """
    try:
        model_manager = prediction.ModelManager(season=season)
        model_manager.load_models()             # Loads the entire pipelines
        model_manager.build_knowledge_graph()   # Builds the Knowledge Graph with clustering
        return model_manager
    except FileNotFoundError as e:
        logging.error(f"Model file missing: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading models: {e}")
        raise

def visualize_subgraph(subgraph):
    """
    Visualizes a NetworkX subgraph using PyVis within Streamlit.
    Labels nodes with their actual names instead of IDs.

    Parameters:
        subgraph (networkx.Graph): The subgraph to visualize.
    """
    if subgraph.number_of_nodes() == 0:
        st.warning("No subgraph available to display.")
        return
    
    try:
        # Initialize PyVis Network
        net = Network(height='600px', width='100%', notebook=False, directed=False)

        # Add nodes with labels and colors based on type
        for node, data in subgraph.nodes(data=True):
            label = node  # Node identifier is already the name
            node_type = data.get('type', 'Unknown')

            # Define node color and shape based on type
            if node_type == 'Player':
                color = 'blue'
                shape = 'ellipse'
            elif node_type == 'Team':
                color = 'green'
                shape = 'box'
            elif node_type == 'Game':
                color = 'red'
                shape = 'diamond'
            elif node_type == 'Venue':
                color = 'orange'
                shape = 'triangle'
            elif node_type == 'HistoricalPerformance':
                color = 'purple'
                shape = 'dot'
            else:
                color = 'grey'
                shape = 'ellipse'
            
            # Optionally, use cluster information for coloring
            cluster_id = data.get('cluster')
            if cluster_id is not None:
                # Assign a color based on cluster ID
                # For simplicity, use a palette or generate colors dynamically
                # Here, we'll use a basic mapping for demonstration
                palette = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
                           '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
                           '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
                           '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080']
                color = palette[cluster_id % len(palette)]
            
            net.add_node(node, label=label, color=color, shape=shape)

        # Add edges with labels
        for u, v, data in subgraph.edges(data=True):
            relation = data.get('relation', '')
            net.add_edge(u, v, title=relation)
        
        # Enable physics for better layout
        net.toggle_physics(True)
        
        # Generate the graph HTML
        graph_html = net.generate_html()

        # Display the graph in Streamlit
        components.html(graph_html, height=700, width=900)
    except Exception as e:
        logging.error(f"Error visualizing subgraph: {e}")
        st.error("Error visualizing the Knowledge Graph subgraph.")

def main():
    st.title("NBA Player Performance Prediction")

    # User selects season first to load appropriate data
    season = st.selectbox('Select Season:', ['2023-24', '2022-23', '2021-22'], index=0)

    # Load models and related components
    try:
        model_manager = load_models(season=season)
    except FileNotFoundError as e:
        st.error(f"Model file missing: {e}")
        return
    except Exception as e:
        st.error(f"An unexpected error occurred while loading models: {e}")
        return

    player_name_input = st.text_input("Enter Player Name (e.g., LeBron James)")

    if player_name_input:
        # Fetch all players for the selected season
        players_df = data_collection.get_all_players(season)
        if players_df.empty:
            st.error(f"No player data available for season {season}.")
            return

        # Retrieve player ID
        player_id = utils.get_player_id(player_name_input, players_df=players_df)
        if player_id is None:
            st.error("Player not found. Please check the name and try again.")
            return

        # Fetch all NBA teams for the selected season
        all_teams = data_collection.get_team_data()
        if all_teams.empty:
            st.error("No team data available.")
            return

        team_abbreviations = all_teams['abbreviation'].tolist()
        opponent_abbr = st.selectbox("Select Opponent Team", team_abbreviations)

        st.header(f"Predict Performance for {player_name_input} against {opponent_abbr}")

        if st.button("Predict"):
            try:
                # Prepare input data
                X = prediction.prepare_input(player_name_input, opponent_abbr, season=season)

                # Make predictions using the pipelines
                reg_pred = prediction.predict_regression(X, model_manager.regressor_pipeline)
                clf_pred = prediction.predict_classification(X, model_manager.classifier_pipeline)

                # Display predictions
                st.write(f"**Predicted Points:** {reg_pred:.2f}")
                st.write(f"**Will Exceed Threshold:** {'Yes' if clf_pred == 1 else 'No'}")

                # Generate explanation using OpenAI
                prediction_result = {
                    'points': reg_pred,
                    'exceeds_threshold': bool(clf_pred)
                }

                # Retrieve player full name with team for uniqueness
                player_row = players_df[players_df['id'] == player_id]
                if not player_row.empty:
                    player_full_name = player_row.iloc[0]['full_name']
                    team_id = player_row.iloc[0]['team_id']
                    team_row = all_teams[all_teams['id'] == team_id]
                    if not team_row.empty:
                        team_name = team_row.iloc[0]['full_name']
                        unique_player_name = f"{player_full_name} ({team_name})"
                    else:
                        unique_player_name = player_full_name
                else:
                    unique_player_name = player_name_input

                # Resolve opponent team name from abbreviation
                opponent_team_name = model_manager.get_team_name_from_abbr(opponent_abbr)

                # Extract context from KG with increased depth
                subgraph = extract_context_subgraph(
                    KG=model_manager.KG,
                    player_name=unique_player_name,
                    opponent_team_abbr=opponent_abbr,
                    model_manager=model_manager,
                    depth=3
                )

                context_info = {
                    'Player': unique_player_name,
                    'Opponent': opponent_team_name,
                    'Relationships': [
                        {
                            'source': u,
                            'target': v,
                            'relation': d['relation']
                        }
                        for u, v, d in subgraph.edges(data=True)
                    ]
                }

                explanation = model_manager.generate_explanation(player_name_input, opponent_abbr, prediction_result, context_info)
                st.subheader("Prediction Explanation")
                st.write(explanation)

                # Visualize the subgraph
                st.subheader("Knowledge Graph Subgraph")
                visualize_subgraph(subgraph)

            except FileNotFoundError as e:
                st.error(f"Model file missing: {e}")
            except ValueError as e:
                st.error(f"Input error: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
    else:
        st.info("Please enter a player's name to begin prediction.")

if __name__ == "__main__":
    main()