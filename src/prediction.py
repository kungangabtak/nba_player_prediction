# src/prediction.py

import pickle
import pandas as pd
import os
import logging

class ModelManager:
    def __init__(self):
        self.reg_model = self.load_model('Regressor')
        self.clf_model = self.load_model('Classifier')
        self.scaler = self.load_scaler()
        self.label_encoder = self.load_label_encoder()
    
    def load_model(self, model_type):
        model_path = os.path.join('models', f'XGBoost{model_type}.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found.")
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    
    def load_scaler(self):
        scaler_path = os.path.join('models', 'scaler.pkl')
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file '{scaler_path}' not found.")
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
        return scaler
    
    def load_label_encoder(self):
        label_encoder_path = os.path.join('models', 'label_encoder.pkl')
        if not os.path.exists(label_encoder_path):
            raise FileNotFoundError(f"Label encoder file '{label_encoder_path}' not found.")
        with open(label_encoder_path, 'rb') as file:
            label_encoder = pickle.load(file)
        return label_encoder
    
    def predict_regression(self, input_data):
        features_scaled = self.scaler.transform(input_data)
        prediction = self.reg_model.predict(features_scaled)
        return prediction
    
    def predict_classification(self, input_data):
        features_scaled = self.scaler.transform(input_data)
        prediction = self.clf_model.predict(features_scaled)
        return prediction
    
    def prepare_input(self, minutes, fg_pct, ft_pct, threep_pct, usg_pct, per, opponent_team):
        try:
            opponent_encoded = self.label_encoder.transform([opponent_team])[0]
        except ValueError:
            raise ValueError(f"Opponent team '{opponent_team}' was not seen during training.")
        
        input_df = pd.DataFrame({
            'Minutes_Played': [minutes],
            'FG_Percentage': [fg_pct],
            'FT_Percentage': [ft_pct],
            'ThreeP_Percentage': [threep_pct],
            'Usage_Rate': [usg_pct],
            'PER': [per],
            'Opponent_Team': [opponent_encoded]
        })
        return input_df

model_manager = ModelManager()

def predict_regression(input_data):
    return model_manager.predict_regression(input_data)

def predict_classification(input_data):
    return model_manager.predict_classification(input_data)

def prepare_input(minutes, fg_pct, ft_pct, threep_pct, usg_pct, per, opponent_team):
    """
    Prepares the input dataframe for prediction.

    Parameters:
        minutes (float): Minutes played.
        fg_pct (float): Field goal percentage.
        ft_pct (float): Free throw percentage.
        threep_pct (float): Three-point percentage.
        usg_pct (float): Usage rate.
        per (float): Player Efficiency Rating.
        opponent_team (str): Opponent team abbreviation.

    Returns:
        pd.DataFrame: Prepared input dataframe.
    """
    return model_manager.prepare_input(minutes, fg_pct, ft_pct, threep_pct, usg_pct, per, opponent_team)

def predict_regression_blocks(input_data):
    return model_manager.predict_regression(input_data)

def predict_classification_blocks(input_data):
    return model_manager.predict_classification(input_data)

def predict_regression_assists(input_data):
    return model_manager.predict_regression(input_data)

def predict_classification_assists(input_data):
    return model_manager.predict_classification(input_data)

def predict_regression_rebounds(input_data):
    return model_manager.predict_regression(input_data)

def predict_classification_rebounds(input_data):
    return model_manager.predict_classification(input_data)

def predict_regression_steals(input_data):
    return model_manager.predict_regression(input_data)

def predict_classification_steals(input_data):
    return model_manager.predict_classification(input_data)