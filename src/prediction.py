# src/prediction.py

import joblib
import pandas as pd
import os
import logging
from src.data_collection import get_all_players, get_player_game_logs, get_all_player_game_logs, get_team_data
from src.utils import get_opponent_teams, get_full_team_name
from src.knowledge_graph import build_kg
from src.kg_utils import extract_context_subgraph
import networkx as nx
import openai  # type: ignore # Ensure you have openai installed and set up
import json
import tiktoken


from dotenv import load_dotenv # type: ignore
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure this environment variable is set

class ModelManager:
    def __init__(self, models_dir='models', season='2023-24'):
        self.models_dir = models_dir
        self.regressor_pipeline = None
        self.classifier_pipeline = None
        self.KG = None
        self.season = season
        self.teams_df = None  # Initialize teams_df

    def load_models(self):
        """
        Loads the regression and classification pipeline models.
        """
        try:
            self.regressor_pipeline = joblib.load(os.path.join(self.models_dir, 'XGBoostRegressor_pipeline.joblib'))
            self.classifier_pipeline = joblib.load(os.path.join(self.models_dir, 'XGBoostClassifier_pipeline.joblib'))
            logging.info("Models and pipelines loaded successfully.")
        except FileNotFoundError as e:
            logging.error(f"Model file missing: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error loading models: {e}")
            raise
    
    def get_team_name_from_abbr(self, abbr):
        """
        Retrieves the full team name based on its abbreviation.

        Parameters:
            abbr (str): Team abbreviation (e.g., 'BOS').

        Returns:
            str or None: Full team name if found, else None.
        """
        for node, data in self.KG.nodes(data=True):
            if data.get('type') == 'Team' and data.get('abbreviation') == abbr:
                return data.get('full_name')  # Changed from 'name' to 'full_name'
        return None
    
    def build_knowledge_graph(self):
        """
        Builds or loads the Knowledge Graph for the specified season.
        """
        try:
            players_df = get_all_players(season=self.season)
            teams_df = get_team_data()
            self.teams_df = teams_df  # Store teams_df as an instance attribute
            game_logs_df = get_all_player_game_logs(season=self.season)
            self.KG = build_kg(players_df, teams_df, game_logs_df)
            logging.info("Knowledge Graph built successfully.")
        except Exception as e:
            logging.error(f"Error building Knowledge Graph: {e}")
            raise

    def get_player_team(self, player_id):
        """
        Retrieves the team ID for a given player ID from the Knowledge Graph.
        """
        try:
            for neighbor in self.KG.neighbors(player_id):
                edge_data = self.KG.get_edge_data(player_id, neighbor)
                if edge_data and edge_data.get('relation') == 'plays_for':
                    return neighbor
            return None
        except Exception as e:
            logging.error(f"Error retrieving team for player ID {player_id}: {e}")
            return None
    
    def get_team_id_from_abbr(self, team_abbr):
        """
        Retrieves the team ID based on the team abbreviation.

        Parameters:
            team_abbr (str): The abbreviation of the team.

        Returns:
            str or None: Team ID if found, else None.
        """
        try:
            team_row = self.teams_df[self.teams_df['abbreviation'] == team_abbr.upper()]
            if not team_row.empty:
                team_id = team_row.iloc[0]['id']
                logging.info(f"Found Team ID {team_id} for abbreviation '{team_abbr}'.")
                return team_id
            else:
                logging.warning(f"Team abbreviation '{team_abbr}' not found in teams DataFrame.")
                return None
        except Exception as e:
            logging.error(f"Error retrieving team ID for abbreviation '{team_abbr}': {e}")
            return None

    def generate_explanation(self, player_name, opponent_team, prediction_result, context_info, max_retries=5):
        """
        Generates an explanation for the prediction using OpenAI's API.

        Parameters:
            player_name (str): Name of the player.
            opponent_team (str): Name of the opponent team.
            prediction_result (dict): Dictionary containing prediction results.
            context_info (dict): Dictionary containing context information.
            max_retries (int): Maximum number of retries for rate limit errors.

        Returns:
            str: Generated explanation or an error message.
        """
        # Construct a concise prompt
        relationships = "; ".join([
            f"{rel['source']} {rel['relation']} {rel['target']}"
            for rel in context_info.get('Relationships', [])
            if rel.get('relation') in ['matched_up_against', 'plays_for']
        ])

        prompt = (
            f"Provide an explanation for the prediction that {player_name} will "
            f"{'exceed' if prediction_result['exceeds_threshold'] else 'not exceed'} the "
            f"points threshold against {opponent_team}. "
            f"Consider the following relationships: {relationships}."
        )

        # Count tokens to ensure prompt is within limits
        token_count = self.count_tokens(prompt)
        logging.debug(f"Prompt token count: {token_count}")

        if token_count > 4000:  # Adjust based on model's max tokens
            logging.warning("Prompt is too long. Trimming the context.")
            # Trim the relationships or other parts as necessary
            relationships = "; ".join([
                f"{rel['source']} {rel['relation']} {rel['target']}"
                for rel in context_info.get('Relationships', [])[:10]  # Limit to first 10 relationships
                if rel.get('relation') in ['matched_up_against', 'plays_for']
            ])
            prompt = (
                f"Provide an explanation for the prediction that {player_name} will "
                f"{'exceed' if prediction_result['exceeds_threshold'] else 'not exceed'} the "
                f"points threshold against {opponent_team}. "
                f"Consider the following relationships: {relationships}."
            )

        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                explanation = response.choices[0].message['content']
                return explanation
            except openai.error.RateLimitError:
                wait_time = 2 ** attempt
                logging.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            except openai.error.InvalidRequestError as e:
                logging.error(f"Invalid request: {e}")
                break
            except Exception as e:
                logging.error(f"Error generating explanation with OpenAI: {e}")
                break
        return "Explanation could not be generated at this time. Please try again later."

    def count_tokens(self, text):
        """
        Counts the number of tokens in the given text using OpenAI's tiktoken.

        Parameters:
            text (str): The text to count tokens for.

        Returns:
            int: Number of tokens.
        """
        try:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            return len(encoding.encode(text))
        except Exception as e:
            logging.error(f"Error counting tokens: {e}")
            return 0  # Or handle as appropriate
def prepare_input(player_name, opponent_abbr, season='2023-24'):
    """
    Prepares the input features for prediction based on player name, opponent team, and season.
    
    Parameters:
        player_name (str): Full name of the player (e.g., 'LeBron James').
        opponent_abbr (str): Opponent team abbreviation (e.g., 'LAL').
        season (str): NBA season (e.g., '2023-24').
    
    Returns:
        pd.DataFrame: DataFrame containing the prepared input features.
    """
    try:
        # Fetch player data
        players_df = get_all_players(season=season)
        player_id = get_player_id(player_name, players_df=players_df)
        if player_id is None:
            raise ValueError(f"Player '{player_name}' not found for season {season}.")

        # Fetch player's game logs
        gamelogs = get_player_game_logs(player_name, season=season)
        if gamelogs.empty:
            raise ValueError(f"No game logs found for player '{player_name}' in season {season}.")

        # Aggregate player statistics (e.g., average points, usage rate, efficiency)
        stats = {
            'Minutes_Played': gamelogs['MIN'].astype(float).mean(),
            'FG_Percentage': gamelogs['FG_PCT'].astype(float).mean(),
            'FT_Percentage': gamelogs['FT_PCT'].astype(float).mean(),
            'ThreeP_Percentage': gamelogs['FG3_PCT'].astype(float).mean(),
            'Usage_Rate': (gamelogs['FGA'].astype(float).mean() + 0.44 * gamelogs['FTA'].astype(float).mean() + gamelogs['TOV'].astype(float).mean()) / gamelogs['MIN'].astype(float).mean() if gamelogs['MIN'].astype(float).mean() > 0 else 0,
            'EFFICIENCY': (
                gamelogs['PTS'].astype(float).mean() +
                gamelogs['REB'].astype(float).mean() +
                gamelogs['AST'].astype(float).mean() +
                gamelogs['STL'].astype(float).mean() +
                gamelogs['BLK'].astype(float).mean() -
                (gamelogs['FGA'].astype(float).mean() - gamelogs['FG_PCT'].astype(float).mean() * gamelogs['FGA'].astype(float).mean()) -
                (gamelogs['FTA'].astype(float).mean() - gamelogs['FT_PCT'].astype(float).mean() * gamelogs['FTA'].astype(float).mean()) -
                gamelogs['TOV'].astype(float).mean()
            ),
            'Opponent_Team': opponent_abbr
        }

        # Create DataFrame
        input_df = pd.DataFrame([stats])
        logging.info(f"Input features prepared for player '{player_name}' against '{opponent_abbr}'.")
        return input_df
    except Exception as e:
        logging.error(f"Error preparing input: {e}")
        raise

def predict_regression(input_df, regressor_pipeline):
    """
    Makes a regression prediction (predicted points) using the pre-trained pipeline.
    
    Parameters:
        input_df (pd.DataFrame): Prepared input features.
        regressor_pipeline: Loaded regression pipeline.
    
    Returns:
        float: Predicted points.
    """
    try:
        prediction = regressor_pipeline.predict(input_df)[0]
        logging.info(f"Regression prediction made: {prediction:.2f} points.")
        return prediction
    except Exception as e:
        logging.error(f"Error during regression prediction: {e}")
        raise

def predict_classification(input_df, classifier_pipeline):
    """
    Makes a classification prediction (will exceed threshold) using the pre-trained pipeline.
    
    Parameters:
        input_df (pd.DataFrame): Prepared input features.
        classifier_pipeline: Loaded classification pipeline.
    
    Returns:
        int: 1 if will exceed threshold, else 0.
    """
    try:
        prediction = classifier_pipeline.predict(input_df)[0]
        logging.info(f"Classification prediction made: {'Yes' if prediction == 1 else 'No'}.")
        return prediction
    except Exception as e:
        logging.error(f"Error during classification prediction: {e}")
        raise

def get_player_id(player_name, players_df):
    """
    Retrieves the player ID for a given player name from the players DataFrame.
    
    Parameters:
        player_name (str): Full name of the player.
        players_df (pd.DataFrame): DataFrame containing player information.
    
    Returns:
        str or None: Player ID if found, else None.
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

