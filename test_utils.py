# streamlit_app.py

import streamlit as st
import pandas as pd
from src import prediction, utils, data_collection
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    st.title("NBA Player Performance Prediction")
    
    season = '2022-23'
    
    player_name = st.text_input("Enter Player Name (e.g., LeBron James)")
    
    if player_name:
        player_id = utils.get_player_id(player_name)
        if player_id is None:
            st.error("Player not found. Please check the name and try again.")
            return
        
        opponents = utils.get_opponent_teams(player_id, season)
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
            if threshold_metric == 'Points':
                input_df = prediction.prepare_input(minutes, fg_pct, ft_pct, threep_pct, usg_pct, per, opponent)
                reg_pred = prediction.predict_regression(input_df)[0]
                clf_pred = prediction.predict_classification(input_df)[0]
                st.write(f"**Predicted Points:** {reg_pred:.2f}")
                st.write(f"**Will Score Above {threshold_value} Points:** {'Yes' if clf_pred == 1 else 'No'}")
            elif threshold_metric == 'Blocks':
                # Placeholder for Blocks prediction
                st.warning("Blocks prediction not implemented yet.")
            elif threshold_metric == 'Assists':
                # Placeholder for Assists prediction
                st.warning("Assists prediction not implemented yet.")
            elif threshold_metric == 'Rebounds':
                # Placeholder for Rebounds prediction
                st.warning("Rebounds prediction not implemented yet.")
            elif threshold_metric == 'Steals':
                # Placeholder for Steals prediction
                st.warning("Steals prediction not implemented yet.")
            else:
                st.error("Invalid metric selected.")
    
    else:
        st.info("Please enter a player's name to begin prediction.")

if __name__ == "__main__":
    main()