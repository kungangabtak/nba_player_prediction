# test_utils.py

import unittest
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src import utils

class TestUtils(unittest.TestCase):

    def test_preprocess_data(self):
        df = pd.DataFrame({
            'Minutes_Played': [30, 35],
            'FG_Percentage': [0.45, 0.5],
            'FT_Percentage': [0.75, 0.8],
            'ThreeP_Percentage': [0.35, 0.4],
            'Usage_Rate': [20, 25],
            'PER': [18, 20],
            'Points': [18, 25],
            'Opponent_Team': ['LAL', 'BOS']
        })
        threshold = 20
        features_scaled, target_class, scaler, label_encoder = utils.preprocess_data(df, threshold)
        
        # Check the shape of the scaled features
        self.assertEqual(features_scaled.shape, (2, 7))
        
        # Check target_class values
        expected_class = pd.Series([0, 1])
        pd.testing.assert_series_equal(target_class.reset_index(drop=True), expected_class)
        
        # Check if scaler is fitted
        self.assertIsNotNone(scaler)
        
        # Check if label_encoder is fitted correctly
        self.assertIsInstance(label_encoder, LabelEncoder)
        self.assertListEqual(label_encoder.classes_.tolist(), ['BOS', 'LAL'])

    def test_get_player_id_valid(self):
        player_name = 'LeBron James'
        player_id = utils.get_player_id(player_name)
        self.assertIsInstance(player_id, int)

    def test_get_player_id_invalid(self):
        player_name = 'Invalid Player'
        player_id = utils.get_player_id(player_name)
        self.assertIsNone(player_id)

    def test_get_full_team_name_valid(self):
        abbreviation = 'LAL'
        team_name = utils.get_full_team_name(abbreviation)
        self.assertEqual(team_name, 'Los Angeles Lakers')

    def test_get_full_team_name_invalid(self):
        abbreviation = 'XXX'
        team_name = utils.get_full_team_name(abbreviation)
        self.assertIsNone(team_name)

    def test_fetch_player_data_valid(self):
        player_id = 2544  # LeBron James
        season = '2022-23'
        gamelog = utils.fetch_player_data(player_id, season)
        self.assertIsInstance(gamelog, pd.DataFrame)
        self.assertFalse(gamelog.empty)

    def test_fetch_player_data_invalid(self):
        player_id = 0  # Invalid ID
        season = '2022-23'
        gamelog = utils.fetch_player_data(player_id, season)
        self.assertIsInstance(gamelog, pd.DataFrame)
        self.assertTrue(gamelog.empty)

    def test_get_opponent_teams_valid(self):
        player_id = 2544  # LeBron James
        season = '2022-23'
        opponents = utils.get_opponent_teams(player_id, season)
        self.assertIsInstance(opponents, list)
        self.assertTrue(len(opponents) > 0)
        # Note: The actual opponents may vary, so this is a generic check
        self.assertIn('LAL', opponents)  # Example check

    def test_get_opponent_teams_invalid(self):
        player_id = 0  # Invalid ID
        season = '2022-23'
        opponents = utils.get_opponent_teams(player_id, season)
        self.assertIsInstance(opponents, list)
        self.assertEqual(len(opponents), 0)

if __name__ == '__main__':
    unittest.main()