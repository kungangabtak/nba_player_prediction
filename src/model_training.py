# src/model_training.py

import pandas as pd
import pickle
import os
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

def build_and_train_models(data, threshold=20):
    # Rename columns
    data = data.rename(columns={
        'MIN': 'Minutes_Played',
        'FG_PCT': 'FG_Percentage',
        'FT_PCT': 'FT_Percentage',
        'FG3_PCT': 'ThreeP_Percentage',
        'USG_PCT': 'Usage_Rate',
        'PER': 'PER',
        'PTS': 'Points',
        'TEAM_ABBREVIATION': 'Opponent_Team'
    })
    
    # Initialize Label Encoder for Opponent_Team
    label_encoder = LabelEncoder()
    data['Opponent_Team'] = label_encoder.fit_transform(data['Opponent_Team'])
    
    features = data[['Minutes_Played', 'FG_Percentage', 'FT_Percentage', 'ThreeP_Percentage', 'Usage_Rate', 'PER', 'Opponent_Team']]
    target_points = data['Points']
    target_class = (target_points > threshold).astype(int)
    
    # Split the data
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        features, target_points, target_class, test_size=0.2, random_state=42
    )
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Regression Model
    reg_model = XGBRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train_scaled, y_reg_train)
    reg_pred = reg_model.predict(X_test_scaled)
    print("Regression R2 Score:", r2_score(y_reg_test, reg_pred))
    
    # Classification Model
    clf_model = XGBClassifier(n_estimators=100, random_state=42)
    clf_model.fit(X_train_scaled, y_clf_train)
    clf_pred = clf_model.predict(X_test_scaled)
    print("Classification Report:\n", classification_report(y_clf_test, clf_pred))
    
    # Save models, scaler, and label encoder
    os.makedirs('models', exist_ok=True)
    with open('models/XGBoostRegressor.pkl', 'wb') as f:
        pickle.dump(reg_model, f)
    with open('models/XGBoostClassifier.pkl', 'wb') as f:
        pickle.dump(clf_model, f)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    return reg_model, clf_model, scaler, label_encoder