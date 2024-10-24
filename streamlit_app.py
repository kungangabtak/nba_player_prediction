

import streamlit as st
import pandas as pd
from src import prediction, utils, data_collection
import logging
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@st.cache_resource
def load_models():
    try:
        reg_model = prediction.load_model('Regressor')
        clf_model = prediction.load_model('Classifier')
        scaler = prediction.load_scaler()
        label_encoder = prediction.load_label_encoder()
        return reg_model, clf_model, scaler, label_encoder
    except FileNotFoundError as e:
        logging.error(f"Model file missing: {e}")
        raise
    except pickle.UnpicklingError as e:
        logging.error(f"Error unpickling model: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading models: {e}")
        raise

def main():
    st.title("NBA Player Performance Prediction")

    try:
        reg_model, clf_model, scaler, label_encoder = load_models()
    except FileNotFoundError as e:
        st.error(f"Model file missing: {e}")
        return
    except pickle.UnpicklingError as e:
        st.error(f"Error loading model: {e}")
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
        usg_pct = st.number_input("Usage Rate", min_value=0, max_value=100, value=25)
        per = st.number_input("Player Efficiency Rating (PER)", min_value=0.0, max_value=40.0, value=20.0)

        threshold_metric = st.selectbox("Select Metric to Predict", ['Points', 'Blocks', 'Assists', 'Rebounds', 'Steals'])
        threshold_value = st.number_input(f"Enter Threshold for {threshold_metric}", min_value=0.0, value=20.0)

        if st.button("Predict"):
            try:
                input_scaled = prediction.prepare_input(minutes, fg_pct, ft_pct, threep_pct, usg_pct, per, opponent)
                if threshold_metric == 'Points':
                    reg_pred = prediction.predict_regression(input_scaled)[0]
                    clf_pred = prediction.predict_classification(input_scaled)[0]
                    st.write(f"**Predicted Points:** {reg_pred:.2f}")
                    st.write(f"**Will Score Above {threshold_value} Points:** {'Yes' if clf_pred == 1 else 'No'}")
                elif threshold_metric == 'Blocks':
                    reg_pred = prediction.predict_regression(input_scaled)[0]
                    clf_pred = prediction.predict_classification(input_scaled)[0]
                    st.write(f"**Predicted Blocks:** {reg_pred:.2f}")
                    st.write(f"**Will Block Above {threshold_value} Times:** {'Yes' if clf_pred == 1 else 'No'}")
                elif threshold_metric == 'Assists':
                    reg_pred = prediction.predict_regression(input_scaled)[0]
                    clf_pred = prediction.predict_classification(input_scaled)[0]
                    st.write(f"**Predicted Assists:** {reg_pred:.2f}")
                    st.write(f"**Will Assist Above {threshold_value} Times:** {'Yes' if clf_pred == 1 else 'No'}")
                elif threshold_metric == 'Rebounds':
                    reg_pred = prediction.predict_regression(input_scaled)[0]
                    clf_pred = prediction.predict_classification(input_scaled)[0]
                    st.write(f"**Predicted Rebounds:** {reg_pred:.2f}")
                    st.write(f"**Will Rebound Above {threshold_value} Times:** {'Yes' if clf_pred == 1 else 'No'}")
                elif threshold_metric == 'Steals':
                    reg_pred = prediction.predict_regression(input_scaled)[0]
                    clf_pred = prediction.predict_classification(input_scaled)[0]
                    st.write(f"**Predicted Steals:** {reg_pred:.2f}")
                    st.write(f"**Will Steal Above {threshold_value} Times:** {'Yes' if clf_pred == 1 else 'No'}")
                else:
                    st.error("Invalid metric selected.")
            except FileNotFoundError as e:
                st.error(f"Model file missing: {e}")
            except pickle.UnpicklingError as e:
                st.error(f"Error loading model: {e}")
            except ValueError as e:
                st.error(f"Input error: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
    else:
        st.info("Please enter a player's name to begin prediction.")

if __name__ == "__main__":
    main()