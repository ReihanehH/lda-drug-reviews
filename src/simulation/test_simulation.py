import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.simulation.simulation import Simulation


class TestSimulation(unittest.TestCase):
    def setUp(self):
        # Initialize Simulation with a dummy config path
        self.simulation = Simulation(config_path="./config/config.yaml")

    @patch("src.config.yaml.load_yaml_config")
    def test_clean_text_df(self, mock_load_yaml_config):
        # Sample input data
        sample_data = {"text": ["This is a test.", "Another test case."]}
        sample_df = pd.DataFrame(sample_data)

        # Mock the configuration to avoid dependencies
        mock_load_yaml_config.return_value = {}

        # Expected output after cleaning
        expected_data = {"text": ["test.", "anoth test case."]}
        expected_df = pd.DataFrame(expected_data)

        # Run the method
        result_df = self.simulation.clean_text_df(sample_df["text"])

        # Check if the result matches the expected output
        pd.testing.assert_frame_equal(result_df.to_frame(), expected_df)

    @patch("src.config.yaml.load_yaml_config")
    @patch("pandas.read_csv")
    def test_create_train_df(self, mock_read_csv, mock_load_yaml_config):
        # Sample input data
        sample_data = {
            "review": ["This is a test.", "Another test case."],
            "drugName": ["drug1", "drug2"],
        }

        # Expected output (cleaned series of reviews)
        expected_df = pd.Series(["test.", "anoth test case."], name="clean_review")

        sample_df = pd.DataFrame(sample_data)

        # Mock the configuration to avoid dependencies
        mock_load_yaml_config.return_value = {}
        mock_read_csv.return_value = sample_df

        # Run the method
        result_df = self.simulation.create_train_df(
            dataset_path="dummy_path",
            num_of_samples=2,
            sampling_seed=42,
            cache_enabled=False,
        )

        # Check if the result matches the expected output
        pd.testing.assert_series_equal(result_df, expected_df)

    @patch("src.config.yaml.load_yaml_config")
    def test_create_train_dtm(self, mock_load_yaml_config):
        # Sample input data
        sample_data = pd.Series(["test.", "anoth test case."], name="clean_review")

        # Mock the configuration to avoid dependencies
        mock_load_yaml_config.return_value = {}

        # Expected output for vocab and dtm
        vocab = ["anoth", "case", "test"]
        dtm = np.array([[0, 0, 1], [1, 1, 1]])

        # Run the method
        result_dtm, result_vocab = self.simulation.create_train_dtm(
            sample_data, min_df_threshold=0.0
        )

        # Check if the result matches the expected output
        np.testing.assert_array_equal(result_dtm, dtm)
        np.testing.assert_array_equal(result_vocab, vocab)


if __name__ == "__main__":
    unittest.main()
