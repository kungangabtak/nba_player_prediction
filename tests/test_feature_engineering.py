# tests/test_feature_engineering_pipeline.py

import unittest
import pandas as pd
from src.feature_engineering import engineer_features

class TestFeatureEngineeringPipeline(unittest.TestCase):
    def test_full_pipeline(self):
        # Mock players_df
        players_df = pd.DataFrame({
            'PERSON_ID': [1627759, 1631108],
            'TEAM_ABBREVIATION': ['BOS', 'LAL'],
            'DISPLAY_FIRST_LAST': ['Jaylen Brown', 'Max Christie']
        })
        
        # Mock game logs
        game_logs = pd.DataFrame({
            'SEASON_ID': ['2024-25'],
            'Player_ID': [1627759, 1631108],
            'Game_ID': ['0022400001', '0022400002'],
            'GAME_DATE': ['2024-10-19', '2024-10-20'],
            'MATCHUP': ['BOS vs. NYK', 'LAL vs. MIN'],
            'WL': ['W', 'L'],
            'MIN': [35, 30],
            'FGM': [10, 8],
            'FGA': [20, 15],
            'FG_PCT': [0.5, 0.533],
            'FG3M': [2, 1],
            'FG3A': [5, 3],
            'FG3_PCT': [0.4, 0.333],
            'FTM': [5, 4],
            'FTA': [6, 5],
            'FT_PCT': [0.833, 0.8],
            'OREB': [1, 2],
            'DREB': [5, 4],
            'REB': [6, 6],
            'AST': [7, 5],
            'STL': [2, 1],
            'BLK': [1, 0],
            'TOV': [3, 2],
            'PF': [2, 3],
            'PTS': [27, 21],
            'PLUS_MINUS': [5, -3],
            'VIDEO_AVAILABLE': [False, False],
            'PLAYER_NAME': ['Jaylen Brown', 'Max Christie']
        })
        
        processed_data, label_encoder = engineer_features(game_logs, players_df)
        
        # Assertions
        self.assertFalse(processed_data.empty)
        self.assertIsNotNone(label_encoder)
        self.assertIn('Opponent_Team', processed_data.columns)
        self.assertTrue(processed_data['Opponent_Team'].dtype == int)
        self.assertIn('PTS', processed_data.columns)

if __name__ == '__main__':
    unittest.main()