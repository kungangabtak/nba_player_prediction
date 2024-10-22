# test_data_collection.py

import unittest
from src import data_collection
import pandas as pd

class TestDataCollection(unittest.TestCase):

    def test_get_all_players(self):
        players = data_collection.get_all_players(season='2022-23')
        self.assertIsInstance(players, pd.DataFrame)
        self.assertFalse(players.empty)

    def test_get_player_data_valid(self):
        player_id = 2544  # LeBron James
        season = '2022-23'
        data = data_collection.get_player_data(player_id, season)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)

    def test_get_player_data_invalid(self):
        player_id = 0  # Invalid ID
        season = '2022-23'
        data = data_collection.get_player_data(player_id, season)
        self.assertTrue(data.empty)

if __name__ == '__main__':
    unittest.main()