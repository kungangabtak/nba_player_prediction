# src/prediction.py

import pickle
import os
import logging

class ModelManager:
    def __init__(self):
        self.models_dir = 'models'
        self.regressor_path = os.path.join(self.models_dir, 'XGBoostRegressor.pkl')
        self.classifier_path = os.path.join(self.models_dir, 'XGBoostClassifier.pkl')
        self.scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        self.label_encoder_path = os.path.join(self.models_dir, 'label_encoder.pkl')
        self.scaler = None
        self.label_encoder = None

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
            with open(path, 'rb') as file:
                model = pickle.load(file)
        except pickle.UnpicklingError as e:
            logging.error(f"Error unpickling the {model_type} model from {path}: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error loading {model_type} model from {path}: {e}")
            raise
        return model

    def load_scaler(self):
        if self.scaler is None:
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler file {self.scaler_path} does not exist.")
            try:
                with open(self.scaler_path, 'rb') as file:
                    self.scaler = pickle.load(file)
            except pickle.UnpicklingError as e:
                logging.error(f"Error unpickling the scaler from {self.scaler_path}: {e}")
                raise
            except Exception as e:
                logging.error(f"Unexpected error loading scaler from {self.scaler_path}: {e}")
                raise
        return self.scaler

    def load_label_encoder(self):
        if self.label_encoder is None:
            if not os.path.exists(self.label_encoder_path):
                raise FileNotFoundError(f"Label encoder file {self.label_encoder_path} does not exist.")
            try:
                with open(self.label_encoder_path, 'rb') as file:
                    self.label_encoder = pickle.load(file)
            except pickle.UnpicklingError as e:
                logging.error(f"Error unpickling the label encoder from {self.label_encoder_path}: {e}")
                raise
            except Exception as e:
                logging.error(f"Unexpected error loading label encoder from {self.label_encoder_path}: {e}")
                raise
        return self.label_encoder

    def prepare_input(self, minutes, fg_pct, ft_pct, threep_pct, usg_pct, per, opponent):
        import pandas as pd
        import numpy as np
        scaler = self.load_scaler()
        label_encoder = self.load_label_encoder()

        try:
            opponent_encoded = label_encoder.transform([opponent])[0]
        except ValueError as e:
            logging.error(f"Error encoding opponent team '{opponent}': {e}")
            raise

        input_data = pd.DataFrame({
            'Minutes_Played': [minutes],
            'FG_Percentage': [fg_pct],
            'FT_Percentage': [ft_pct],
            'ThreeP_Percentage': [threep_pct],
            'Usage_Rate': [usg_pct],
            'EFFICIENCY': [per],
            'Opponent_Team': [opponent_encoded]
        })

        try:
            input_scaled = scaler.transform(input_data)
        except Exception as e:
            logging.error(f"Error scaling input data: {e}")
            raise

        return input_scaled

    def predict_regression(self, input_data):
        reg_model = self.load_model('Regressor')
        try:
            predictions = reg_model.predict(input_data)
        except Exception as e:
            logging.error(f"Error during regression prediction: {e}")
            raise
        return predictions

    def predict_classification(self, input_data):
        clf_model = self.load_model('Classifier')
        try:
            predictions = clf_model.predict(input_data)
        except Exception as e:
            logging.error(f"Error during classification prediction: {e}")
            raise
        return predictions

# Initialize a global ModelManager instance
model_manager = ModelManager()

def load_model(model_type):
    return model_manager.load_model(model_type)

def load_scaler():
    return model_manager.load_scaler()

def load_label_encoder():
    return model_manager.load_label_encoder()

def prepare_input(minutes, fg_pct, ft_pct, threep_pct, usg_pct, per, opponent):
    return model_manager.prepare_input(minutes, fg_pct, ft_pct, threep_pct, usg_pct, per, opponent)

def predict_regression(input_data):
    return model_manager.predict_regression(input_data)

def predict_classification(input_data):
    return model_manager.predict_classification(input_data)