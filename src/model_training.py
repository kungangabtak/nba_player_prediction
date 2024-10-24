# src/model_training.py

import pandas as pd
import joblib
import os
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from nba_api.stats.static import teams
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import clean_data  # Import the clean_data function

def build_and_train_models(data, threshold=20):
    # Clean the data
    data = clean_data(data)

    # Rename columns (ensure that this mapping aligns with your cleaned data)
    data = data.rename(columns={
        'MIN': 'Minutes_Played',
        'FG_PCT': 'FG_Percentage',
        'FT_PCT': 'FT_Percentage',
        'FG3_PCT': 'ThreeP_Percentage',
        'REB': 'REB',  # Assuming 'REB' is already named correctly
        'AST': 'AST',
        'STL': 'STL',
        'BLK': 'BLK',
        'FGA': 'FGA',
        'FGM': 'FGM',
        'FTA': 'FTA',
        'FTM': 'FTM',
        'TOV': 'TOV',
        'PTS': 'PTS',
        'MATCHUP': 'Opponent_Team'  # Assuming 'MATCHUP' is the column to extract opponent
    })

    # Extract 'Opponent_Team' from 'MATCHUP'
    # Assuming 'MATCHUP' is in the format 'TEAM vs. OPPONENT' or 'TEAM @ OPPONENT'
    data['Opponent_Team'] = data['Opponent_Team'].apply(lambda x: x.split(' ')[-1] if pd.notnull(x) else 'UNK')

    # Handle any unknown opponents
    data['Opponent_Team'] = data['Opponent_Team'].replace({'UNK': 'NOP'})  # Replace 'UNK' with a default team, e.g., 'NOP'

    # Get all team abbreviations
    all_team_abbreviations = [team['abbreviation'] for team in teams.get_teams()]

    # Initialize Label Encoder for Opponent_Team with all team abbreviations
    label_encoder = LabelEncoder()
    label_encoder.fit(all_team_abbreviations)

    # Ensure Opponent_Team is string type before encoding
    data['Opponent_Team'] = data['Opponent_Team'].astype(str)

    try:
        # Transform Opponent_Team to integer encoding
        data['Opponent_Team'] = label_encoder.transform(data['Opponent_Team'])
    except ValueError as e:
        logging.error(f"Error encoding Opponent_Team: {e}")
        # Identify which teams are causing the issue
        unique_opponents = data['Opponent_Team'].unique()
        unknown_teams = [team for team in unique_opponents if team not in label_encoder.classes_]
        if unknown_teams:
            logging.warning(f"The following opponent teams were not found in the LabelEncoder training set: {unknown_teams}")
            # Exclude these rows
            data = data[~data['Opponent_Team'].isin(unknown_teams)]
        else:
            logging.error("Unknown error during Opponent_Team encoding.")
            raise e

    # Features and Targets
    required_features = ['Minutes_Played', 'FG_Percentage', 'FT_Percentage', 
                         'ThreeP_Percentage', 'Usage_Rate', 'EFFICIENCY', 'Opponent_Team']
    missing_features = [feat for feat in required_features if feat not in data.columns]
    if missing_features:
        logging.error(f"The following required features are missing from the data: {missing_features}")
        raise KeyError(f"The following required features are missing from the data: {missing_features}")

    features = data[required_features]
    target_pts = data['PTS']
    target_class = (target_pts > threshold).astype(int)

    # Split the data
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        features, target_pts, target_class, test_size=0.2, random_state=42
    )

    # Define numerical and categorical features
    numerical_features = ['Minutes_Played', 'FG_Percentage', 'FT_Percentage', 
                          'ThreeP_Percentage', 'Usage_Rate', 'EFFICIENCY']
    categorical_features = ['Opponent_Team']

    # Create ColumnTransformer to scale numerical features and passthrough categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', 'passthrough', categorical_features)
        ]
    )

    # Regression Model with Hyperparameter Tuning
    reg_pipeline = Pipeline([
        ('preprocessor', preprocessor),
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
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(random_state=42, 
                                      scale_pos_weight=(len(y_clf_train)-sum(y_clf_train))/sum(y_clf_train)))
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
    sns.barplot(x=importance, y=required_features)
    plt.title('Feature Importance from XGBoost Regressor')
    plt.tight_layout()
    os.makedirs('models', exist_ok=True)
    plt.savefig(os.path.join('models', 'feature_importance_regressor.png'))
    plt.close()

    # Save evaluation metrics
    with open(os.path.join('models', 'regression_metrics.txt'), 'w') as f:
        f.write(f"Regression R2 Score: {r2_score(y_reg_test, reg_pred):.4f}\n")

    with open(os.path.join('models', 'classification_report.txt'), 'w') as f:
        f.write(classification_report(y_clf_test, clf_pred))

    # Save models, preprocessor, and label encoder using joblib
    joblib.dump(best_reg_model, os.path.join('models', 'XGBoostRegressor.joblib'))
    joblib.dump(best_clf_model, os.path.join('models', 'XGBoostClassifier.joblib'))
    joblib.dump(preprocessor, os.path.join('models', 'preprocessor.joblib'))
    joblib.dump(label_encoder, os.path.join('models', 'label_encoder.joblib'))

    return best_reg_model, best_clf_model, preprocessor, label_encoder