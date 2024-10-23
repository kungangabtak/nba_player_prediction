# src/prediction.py

import pickle
import pandas as pd
import os

def load_model(model_type):
    model_path = os.path.join('models', f'XGBoost{model_type}.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def load_scaler():
    scaler_path = os.path.join('models', 'scaler.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file '{scaler_path}' not found.")
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    return scaler

def load_label_encoder():
    label_encoder_path = os.path.join('models', 'label_encoder.pkl')
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Label encoder file '{label_encoder_path}' not found.")
    with open(label_encoder_path, 'rb') as file:
        label_encoder = pickle.load(file)
    return label_encoder

def predict_regression(input_data):
    model = load_model('Regressor')
    scaler = load_scaler()
    features_scaled = scaler.transform(input_data)
    prediction = model.predict(features_scaled)
    return prediction

def predict_classification(input_data):
    model = load_model('Classifier')
    scaler = load_scaler()
    features_scaled = scaler.transform(input_data)
    prediction = model.predict(features_scaled)
    return prediction

def prepare_input(minutes, fg_pct, ft_pct, threep_pct, usg_pct, per, opponent_team):
    label_encoder = load_label_encoder()
    try:
        opponent_encoded = label_encoder.transform([opponent_team])[0]
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

if __name__ == "__main__":
    # Example usage
    minutes = 35
    fg_pct = 0.5
    ft_pct = 0.8
    threep_pct = 0.4
    usg_pct = 25
    per = 20
    opponent_team = 'LAL'  # Example team abbreviation
    
    try:
        input_df = prepare_input(minutes, fg_pct, ft_pct, threep_pct, usg_pct, per, opponent_team)
        reg_pred = predict_regression(input_df)[0]
        clf_pred = predict_classification(input_df)[0]
        print(f"Regression Prediction: {reg_pred}")
        print(f"Classification Prediction: {'Yes' if clf_pred == 1 else 'No'}")
    except Exception as e:
        print(f"Error during prediction: {e}")