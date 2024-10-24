# test_data_preprocessing.py

import unittest
import pandas as pd
import numpy as np
from src.data_preprocessing import clean_data

class TestDataPreprocessing(unittest.TestCase):
    def test_clean_data(self):
        # Create a sample DataFrame
        data = {
            'Points': [25, 30, np.inf, 20, 15],
            'REB': [5, 7, 10, -np.inf, 3],
            'AST': [7, 8, 9, 6, 4],
            'Opponent_Team': [1, 2, 3, 4, 5]
        }
        df = pd.DataFrame(data)
        
        # Clean the data
        cleaned_df = clean_data(df)
        
        # Check for NaN values
        self.assertFalse(cleaned_df.isna().values.any(), "There should be no NaN values after cleaning.")
        
        # Check that no infinite values exist
        self.assertFalse(np.isinf(cleaned_df).values.any(), "There should be no infinite values after cleaning.")
        
        # Check that data types are correct
        self.assertTrue(all(cleaned_df.dtypes == 'float64'), "All columns should be of type float64.")

if __name__ == '__main__':
    unittest.main()