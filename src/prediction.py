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
import openai  # Ensure you have openai installed and set up
import json

from dotenv import load_dotenv
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

    def generate_explanation(self, player_name, opponent_team, prediction, context_info):
        """
        Generates an explanation for the prediction using OpenAI's GPT model.
        
        Parameters:
            player_name (str): Name of the player.
            opponent_team (str): Opponent team abbreviation.
            prediction (dict): Dictionary containing regression and classification predictions.
            context_info (dict): Additional context extracted from KG.
        
        Returns:
            str: Generated explanation text.
        """
        try:
            messages = [
                {"role": "system", "content": "You are an NBA analyst."},
                {"role": "user", "content": f"""
                Provide a detailed explanation for the performance prediction of {player_name} against {opponent_team}.

                Prediction Details:
                - Predicted Points: {prediction['points']:.2f}
                - Will Exceed Threshold: {'Yes' if prediction['exceeds_threshold'] else 'No'}

                Context Information:
                {json.dumps(context_info, indent=4)}

                Using the available data, explain how the player's recent performance, team dynamics, and historical matchups influence this prediction.
                                """}
            ]

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=250,
                temperature=0.7,
            )
            explanation = response['choices'][0]['message']['content'].strip()
            return explanation
        except Exception as e:
            logging.error(f"Error generating explanation with OpenAI: {e}")
            return "Unable to generate explanation at this time."

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

# src/data_preprocessing.py

import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
import logging

def clean_data(df):
    # Log all columns before cleaning
    logging.info(f"Columns before cleaning: {df.columns.tolist()}")
    
    # Replace infinite values with NaN
    num_inf = np.isinf(df.select_dtypes(include=['number'])).sum().sum()
    if num_inf > 0:
        logging.warning(f"Found {num_inf} infinite values. Replacing with NaN.")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Cap extremely large and small values to reasonable limits (1st and 99th percentiles)
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        lower_limit = df[col].quantile(0.01)
        upper_limit = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
        logging.debug(f"Capped values in column '{col}' between {lower_limit} and {upper_limit}.")
    
    # Check for any remaining infinite values
    if np.isinf(df[numeric_cols]).values.any():
        logging.error("Data contains infinite values even after replacement.")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Check for any remaining NaN values
    num_nan = df[numeric_cols].isna().sum().sum()
    if num_nan > 0:
        logging.info(f"Found {num_nan} NaN values. Proceeding with imputation.")
    else:
        logging.info("No NaN values found in numeric columns.")
    
    # Use SimpleImputer to handle missing values (impute with mean)
    imputer = SimpleImputer(strategy='mean')
    try:
        df_imputed = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)
        logging.info("Data imputation with mean strategy successful.")
    except ValueError as e:
        logging.error(f"Imputation failed: {e}")
        raise ValueError(f"Imputation failed: {e}")
    
    # Log detailed imputation information
    missing_before = df[numeric_cols].isna().sum()
    missing_after = df_imputed.isna().sum()
    logging.info(f"Missing values before imputation:\n{missing_before}")
    logging.info(f"Missing values after imputation:\n{missing_after}")
    
    # Combine imputed numeric data with non-numeric data
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns
    df_cleaned = pd.concat([df_imputed, df[non_numeric_cols].reset_index(drop=True)], axis=1)
    
    # Log summary statistics after cleaning
    logging.info("Summary statistics after cleaning:")
    logging.info(df_imputed.describe())
    
    return df_cleaned

def feature_engineering(df):
    """
    Enhances the dataset with additional features derived from available data.

    Parameters:
        df (pd.DataFrame): Cleaned DataFrame.

    Returns:
        pd.DataFrame: DataFrame with additional engineered features.
    """
    logging.info("Starting feature engineering.")

    # Example: Calculate rolling averages for the last 5 games
    df = df.sort_values(by=['PLAYER_ID', 'GAME_DATE'], ascending=[True, False])
    df['PTS_Rolling_Avg'] = df.groupby('PLAYER_ID')['PTS'].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
    df['REB_Rolling_Avg'] = df.groupby('PLAYER_ID')['REB'].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
    df['AST_Rolling_Avg'] = df.groupby('PLAYER_ID')['AST'].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
    logging.debug("Calculated rolling averages for PTS, REB, AST.")

    # Example: Home vs Away
    df['HOME_GAME'] = df['MATCHUP'].str.contains('@').map({True: 0, False: 1})  # Assuming '@' indicates away
    logging.debug("Added HOME_GAME feature.")

    # Example: Opponent Strength (average PTS allowed by opponent)
    opponent_abbr = df['Opponent_Team']
    opponent_pts_allowed = df.groupby('Opponent_Team')['PTS'].mean().rename('Opponent_PTS_Allowed')
    df = df.join(opponent_pts_allowed, on='Opponent_Team')
    logging.debug("Added Opponent_PTS_Allowed feature.")

    logging.info("Feature engineering completed.")
    return df