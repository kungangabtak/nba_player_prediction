# src/prediction.py

import joblib
import os
import logging
import pandas as pd
from nba_api.stats.static import teams  # Import to get all NBA teams
from sklearn.preprocessing import LabelEncoder

class ModelManager:
    def __init__(self):
        self.models_dir = 'models'
        self.regressor_path = os.path.join(self.models_dir, 'XGBoostRegressor.joblib')
        self.classifier_path = os.path.join(self.models_dir, 'XGBoostClassifier.joblib')
        self.label_encoder_path = os.path.join(self.models_dir, 'label_encoder.joblib')
        self.scaler_path = os.path.join(self.models_dir, 'scaler.joblib')
        self.label_encoder = None
        self.scaler = None

    def load_model(self, model_type):
        if model_type == 'Regressor':
            path = self.regressor_path
        elif model_type == 'Classifier':
            path = self.classifier_path
        else:
            raise ValueError("Model type must be 'Regressor' or 'Classifier'")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file {path} does not exist.")

        try:
            model = joblib.load(path)
            logging.info(f"Loaded {model_type} model from {path}.")
        except Exception as e:
            logging.error(f"Error loading {model_type} model from {path}: {e}")
            raise
        return model

    def load_label_encoder(self):
        if self.label_encoder is None:
            if not os.path.exists(self.label_encoder_path):
                raise FileNotFoundError(f"Label encoder file {self.label_encoder_path} does not exist.")
            try:
                self.label_encoder = joblib.load(self.label_encoder_path)
                logging.info("Loaded LabelEncoder for Opponent_Team.")
            except Exception as e:
                logging.error(f"Error loading label encoder from {self.label_encoder_path}: {e}")
                raise
        return self.label_encoder

    def load_scaler(self):
        if self.scaler is None:
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler file {self.scaler_path} does not exist.")
            try:
                self.scaler = joblib.load(self.scaler_path)
                logging.info("Loaded StandardScaler.")
            except Exception as e:
                logging.error(f"Error loading scaler from {self.scaler_path}: {e}")
                raise
        return self.scaler

    def prepare_input(self, minutes, fg_pct, ft_pct, threep_pct, usg_pct, per, opponent):
        # Load necessary encoders and scalers
        self.load_label_encoder()
        self.load_scaler()

        # Get all NBA team abbreviations
        all_teams = teams.get_teams()
        team_abbreviations = [team['abbreviation'] for team in all_teams]

        # Ensure the opponent team abbreviation is in uppercase
        opponent = opponent.upper()

        # Check if opponent is a valid NBA team
        if opponent not in team_abbreviations:
            logging.error(f"Opponent team '{opponent}' is not a valid NBA team abbreviation.")
            raise ValueError(f"Opponent team '{opponent}' is not a valid NBA team abbreviation.")

        try:
            opponent_encoded = self.label_encoder.transform([opponent])[0]
            logging.info(f"Encoded Opponent_Team '{opponent}' as {opponent_encoded}.")
        except ValueError as e:
            logging.error(f"Error encoding Opponent_Team '{opponent}': {e}")
            raise

        input_data = pd.DataFrame({
            'Minutes_Played': [minutes],
            'FG_Percentage': [fg_pct],
            'FT_Percentage': [ft_pct],
            'ThreeP_Percentage': [threep_pct],
            'Usage_Rate': [usg_pct],
            'EFFICIENCY': [per],
            'Opponent_Team_Encoded': [opponent_encoded]
        })

        # Scale numerical features
        numeric_features = ['Minutes_Played', 'FG_Percentage', 'FT_Percentage', 
                            'ThreeP_Percentage', 'Usage_Rate', 'EFFICIENCY']
        input_data[numeric_features] = self.scaler.transform(input_data[numeric_features])

        logging.info("Input data prepared for prediction.")
        return input_data

    def predict_regression(self, input_data):
        reg_model = self.load_model('Regressor')
        try:
            predictions = reg_model.predict(input_data)
            logging.info(f"Regression prediction: {predictions}")
        except Exception as e:
            logging.error(f"Error during regression prediction: {e}")
            raise
        return predictions

    def predict_classification(self, input_data):
        clf_model = self.load_model('Classifier')
        try:
            predictions = clf_model.predict(input_data)
            logging.info(f"Classification prediction: {predictions}")
        except Exception as e:
            logging.error(f"Error during classification prediction: {e}")
            raise
        return predictions

# Initialize a global ModelManager instance
model_manager = ModelManager()

def load_model(model_type):
    return model_manager.load_model(model_type)

def load_label_encoder():
    return model_manager.load_label_encoder()

def prepare_input(minutes, fg_pct, ft_pct, threep_pct, usg_pct, per, opponent):
    return model_manager.prepare_input(minutes, fg_pct, ft_pct, threep_pct, usg_pct, per, opponent)

def predict_regression(input_data):
    return model_manager.predict_regression(input_data)

def predict_classification(input_data):
    return model_manager.predict_classification(input_data)