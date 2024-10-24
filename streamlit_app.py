# streamlit_app.py

import streamlit as st
import pandas as pd
from src import prediction, utils, data_collection
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@st.cache_resource
def load_models():
    try:
        reg_model = prediction.load_model('Regressor')
        clf_model = prediction.load_model('Classifier')
        label_encoder = prediction.load_label_encoder()
        scaler = prediction.model_manager.load_scaler()
        return reg_model, clf_model, label_encoder, scaler
    except FileNotFoundError as e:
        logging.error(f"Model file missing: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading models: {e}")
        raise

def main():
    st.title("NBA Player Performance Prediction")

    try:
        reg_model, clf_model, label_encoder, scaler = load_models()
    except FileNotFoundError as e:
        st.error(f"Model file missing: {e}")
        return
    except Exception as e:
        st.error(f"An unexpected error occurred while loading models: {e}")
        return

    season = '2023-24'
    player_name = st.text_input("Enter Player Name (e.g., LeBron James)")

    if player_name:
        players_df = data_collection.get_all_players(season)
        player_id = utils.get_player_id(player_name, players_df)
        if player_id is None:
            st.error("Player not found. Please check the name and try again.")
            return

        gamelogs = data_collection.get_player_data(player_id, season)
        opponents = utils.get_opponent_teams(player_id, season, gamelogs)
        if not opponents:
            st.warning("No game log data available for the selected player.")
            return

        opponent = st.selectbox("Select Opponent Team", opponents)

        st.header(f"Predict Performance for {player_name} against {opponent}")

        minutes = st.number_input("Minutes Played", min_value=0, max_value=48, value=35)
        fg_pct = st.slider("FG Percentage", 0.0, 1.0, 0.5)
        ft_pct = st.slider("FT Percentage", 0.0, 1.0, 0.8)
        threep_pct = st.slider("3PT Percentage", 0.0, 1.0, 0.4)
        usg_pct = st.number_input("Usage Rate", min_value=0.0, max_value=100.0, value=25.0)
        per = st.number_input("Player Efficiency Rating (PER)", min_value=0.0, max_value=40.0, value=20.0)

        threshold_metric = st.selectbox("Select Metric to Predict", ['Points'])
        threshold_value = st.number_input(f"Enter Threshold for {threshold_metric}", min_value=0.0, value=20.0)

        if st.button("Predict"):
            try:
                # Prepare input data
                input_df = prediction.prepare_input(
                    minutes=minutes,
                    fg_pct=fg_pct,
                    ft_pct=ft_pct,
                    threep_pct=threep_pct,
                    usg_pct=usg_pct,
                    per=per,
                    opponent=opponent
                )

                # Make predictions
                reg_pred = prediction.predict_regression(input_df)[0]
                clf_pred = prediction.predict_classification(input_df)[0]
                st.write(f"**Predicted {threshold_metric}:** {reg_pred:.2f}")
                st.write(f"**Will Exceed {threshold_value} {threshold_metric}:** {'Yes' if clf_pred == 1 else 'No'}")
            except FileNotFoundError as e:
                st.error(f"Model file missing: {e}")
            except ValueError as e:
                st.error(f"Input error: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
    else:
        st.info("Please enter a player's name to begin prediction.")

if __name__ == "__main__":
    main()