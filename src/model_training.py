# src/model_training.py

import pandas as pd
import pickle
import os
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from nba_api.stats.static import teams
import logging
import matplotlib.pyplot as plt
import seaborn as sns

def build_and_train_models(data, threshold=20):
    # Rename columns (handled in feature engineering)
    data = data.rename(columns={
        'Minutes_Played': 'Minutes_Played',
        'FG_Percentage': 'FG_Percentage',
        'FT_Percentage': 'FT_Percentage',
        'ThreeP_Percentage': 'ThreeP_Percentage',
        'Usage_Rate': 'Usage_Rate',
        'EFFICIENCY': 'EFFICIENCY',
        'PTS': 'PTS',
        'Opponent_Team': 'Opponent_Team'
    })
    
    # Get all team abbreviations
    all_team_abbreviations = [team['abbreviation'] for team in teams.get_teams()]
    
    # Initialize Label Encoder for Opponent_Team with all team abbreviations
    label_encoder = LabelEncoder()
    label_encoder.fit(all_team_abbreviations)
    
    # Check for any opponent teams not in the label encoder
    unique_opponents = data['Opponent_Team'].unique()
    # Since Opponent_Team is already encoded, we need to reverse transform to get abbreviations
    try:
        unique_opponent_abbr = label_encoder.inverse_transform(unique_opponents)
    except ValueError as e:
        logging.error(f"Error inverse transforming Opponent_Team: {e}")
        unique_opponent_abbr = []
    
    unknown_teams = set()
    for abbr in unique_opponent_abbr:
        if abbr not in all_team_abbreviations:
            unknown_teams.add(abbr)
    
    if unknown_teams:
        logging.warning(f"The following opponent teams were not found in the LabelEncoder training set: {unknown_teams}")
        # Exclude these rows
        data = data[~data['Opponent_Team'].isin(label_encoder.transform(list(unknown_teams)))]
    
    # Transform Opponent_Team (already encoded during feature engineering)
    # Assuming 'Opponent_Team' is already encoded correctly
    
    # Features and Targets
    features = data[['Minutes_Played', 'FG_Percentage', 'FT_Percentage', 'ThreeP_Percentage', 'Usage_Rate', 'EFFICIENCY', 'Opponent_Team']]
    target_pts = data['PTS']
    target_class = (target_pts > threshold).astype(int)
    
    # Split the data
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        features, target_pts, target_class, test_size=0.2, random_state=42
    )
    
    # Scaling and model training within a pipeline
    scaler = StandardScaler()
    
    # Regression Model with Hyperparameter Tuning
    reg_pipeline = Pipeline([
        ('scaler', scaler),
        ('regressor', XGBRegressor(random_state=42))
    ])
    
    reg_params = {
        'regressor__n_estimators': [100, 200],
        'regressor__learning_rate': [0.05, 0.1],
        'regressor__max_depth': [3, 5]
    }
    
    reg_grid = GridSearchCV(reg_pipeline, reg_params, cv=5, scoring='r2', n_jobs=-1)
    reg_grid.fit(X_train, y_reg_train)
    best_reg_model = reg_grid.best_estimator_
    reg_pred = best_reg_model.predict(X_test)
    logging.info("Regression R2 Score: {:.4f}".format(r2_score(y_reg_test, reg_pred)))
    
    # Classification Model with Hyperparameter Tuning and Handling Class Imbalance
    clf_pipeline = Pipeline([
        ('scaler', scaler),
        ('classifier', XGBClassifier(random_state=42, scale_pos_weight=(len(y_clf_train)-sum(y_clf_train))/sum(y_clf_train)))
    ])
    
    clf_params = {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__max_depth': [3, 5]
    }
    
    clf_grid = GridSearchCV(clf_pipeline, clf_params, cv=5, scoring='accuracy', n_jobs=-1)
    clf_grid.fit(X_train, y_clf_train)
    best_clf_model = clf_grid.best_estimator_
    clf_pred = best_clf_model.predict(X_test)
    logging.info("Classification Report:\n{}".format(classification_report(y_clf_test, clf_pred)))
    
    # Feature Importance Analysis
    plt.figure(figsize=(10,6))
    importance = best_reg_model.named_steps['regressor'].feature_importances_
    sns.barplot(x=importance, y=features.columns)
    plt.title('Feature Importance from XGBoost Regressor')
    plt.tight_layout()
    plt.savefig('models/feature_importance_regressor.png')
    plt.close()
    
    # Save evaluation metrics
    with open(os.path.join('models', 'regression_metrics.txt'), 'w') as f:
        f.write(f"Regression R2 Score: {r2_score(y_reg_test, reg_pred):.4f}\n")
    
    with open(os.path.join('models', 'classification_report.txt'), 'w') as f:
        f.write(classification_report(y_clf_test, clf_pred))
    
    # Save models, scaler, and label encoder
    os.makedirs('models', exist_ok=True)
    with open(os.path.join('models', 'XGBoostRegressor.pkl'), 'wb') as f:
        pickle.dump(best_reg_model, f)
    with open(os.path.join('models', 'XGBoostClassifier.pkl'), 'wb') as f:
        pickle.dump(best_clf_model, f)
    with open(os.path.join('models', 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join('models', 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    return best_reg_model, best_clf_model, scaler, label_encoder