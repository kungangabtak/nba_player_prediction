# src/prediction.py

import pickle
import pandas as pd

def load_model(model_type):
    if model_type == 'Regressor':
        with open('models/XGBoostRegressor.pkl', 'rb') as file:
            model = pickle.load(file)
    elif model_type == 'Classifier':
        with open('models/XGBoostClassifier.pkl', 'rb') as file:
            model = pickle.load(file)
    return model

def load_scaler():
    with open('models/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return scaler

def load_label_encoder():
    with open('models/label_encoder.pkl', 'rb') as file:
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
    opponent_encoded = label_encoder.transform([opponent_team])[0]
    
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
    
    input_df = prepare_input(minutes, fg_pct, ft_pct, threep_pct, usg_pct, per, opponent_team)
    reg_pred = predict_regression(input_df)[0]
    clf_pred = predict_classification(input_df)[0]
    print(f"Regression Prediction: {reg_pred}")
    print(f"Classification Prediction: {'Yes' if clf_pred == 1 else 'No'}")