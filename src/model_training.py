# src/model_training.py

import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from nba_api.stats.static import teams
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preprocessing import clean_data
from src.feature_engineering import engineer_features

def build_and_train_models(game_logs_df, players_df, threshold=20):
    """
    Builds and trains regression and classification models based on game logs.

    Parameters:
        game_logs_df (pd.DataFrame): DataFrame containing all players' game logs.
        players_df (pd.DataFrame): DataFrame containing player information.
        threshold (int): Threshold for classification target.

    Returns:
        tuple: Best regression pipeline, best classification pipeline.
    """
    # Clean the data
    data = clean_data(game_logs_df)

    

    # Rename columns to standardize names BEFORE feature engineering
    data = data.rename(columns={
        'MIN': 'Minutes_Played',
        'FG_PCT': 'FG_Percentage',
        'FT_PCT': 'FT_Percentage',
        'FG3_PCT': 'ThreeP_Percentage',
        'REB': 'REB',
        'AST': 'AST',
        'STL': 'STL',
        'BLK': 'BLK',
        'FGA': 'FGA',
        'FGM': 'FGM',
        'FTA': 'FTA',
        'FTM': 'FTM',
        'TOV': 'TOV',
        'PTS': 'PTS',
        'MATCHUP': 'MATCHUP'
    })

    # Verify renaming worked
    print("Columns after renaming:", data.columns)

    # Extract 'Opponent_Team' from 'MATCHUP' before engineer_features
    if 'MATCHUP' in data.columns:
        data['Opponent_Team'] = data['MATCHUP'].apply(lambda x: x.split(' ')[-1] if pd.notnull(x) else 'UNK')
        data['Opponent_Team'] = data['Opponent_Team'].replace({'UNK': 'NOP'})
    else:
        raise KeyError("MATCHUP column is missing, cannot extract Opponent_Team.")

    

    # Now call engineer_features with the renamed data
    data, label_encoder = engineer_features(data, players_df)

    # Check if required features are present after feature engineering
    required_features = ['Minutes_Played', 'FG_Percentage', 'FT_Percentage', 'ThreeP_Percentage',
                         'Usage_Rate', 'EFFICIENCY', 'Opponent_Team']
    missing_features = [feat for feat in required_features if feat not in data.columns]
    if missing_features:
        logging.error(f"The following required features are missing from the data: {missing_features}")
        raise KeyError(f"The following required features are missing from the data: {missing_features}")

    # Features and Targets
    features = data[required_features]
    target_pts = data['PTS']
    target_class = (target_pts > threshold).astype(int)

    # Step 6: Split the data for regression and classification
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        features, target_pts, target_class, test_size=0.2, random_state=42
    )

    # Step 7: Define preprocessor with numerical and categorical transformers
    numerical_features = ['Minutes_Played', 'FG_Percentage', 'FT_Percentage',
                          'ThreeP_Percentage', 'Usage_Rate', 'EFFICIENCY']
    categorical_features = ['Opponent_Team']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Step 8: Define and train Regression Pipeline
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
    best_reg_pipeline = reg_grid.best_estimator_
    reg_pred = best_reg_pipeline.predict(X_test)
    logging.info("Regression R2 Score: {:.4f}".format(r2_score(y_reg_test, reg_pred)))

    # Step 9: Define and train Classification Pipeline
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
    best_clf_pipeline = clf_grid.best_estimator_
    clf_pred = best_clf_pipeline.predict(X_test)
    logging.info("Classification Report:\n{}".format(classification_report(y_clf_test, clf_pred)))

    # Step 10: Feature Importance Plot
    plt.figure(figsize=(10,6))
    importance = best_reg_pipeline.named_steps['regressor'].feature_importances_
    cat_features = best_reg_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = numerical_features + list(cat_features)
    sns.barplot(x=importance, y=feature_names)
    plt.title('Feature Importance from XGBoost Regressor')
    plt.tight_layout()
    os.makedirs('models', exist_ok=True)
    plt.savefig(os.path.join('models', 'feature_importance_regressor.png'))
    plt.close()

    # Step 11: Save evaluation metrics and models
    with open(os.path.join('models', 'regression_metrics.txt'), 'w') as f:
        f.write(f"Regression R2 Score: {r2_score(y_reg_test, reg_pred):.4f}\n")

    with open(os.path.join('models', 'classification_report.txt'), 'w') as f:
        f.write(classification_report(y_clf_test, clf_pred))
        # Save the trained pipelines using joblib
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_reg_pipeline, os.path.join('models', 'XGBoostRegressor_pipeline.joblib'))
    joblib.dump(best_clf_pipeline, os.path.join('models', 'XGBoostClassifier_pipeline.joblib'))

    logging.info("Models and pipelines saved successfully.")

    return best_reg_pipeline, best_clf_pipeline