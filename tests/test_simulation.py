import unittest
from unittest.mock import MagicMock, patch

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

    # @patch("os.makedirs")
    # def test_ensure_directories_exist(self, mock_makedirs):
    #     # Create a mock configuration
    #     mock_config = MagicMock()
    #     mock_config.data.train.cache_dir_path = "/tmp/cache"
    #     mock_config.lda.models_dir_path = "/tmp/models"
    #     mock_config.result.results_dir_path = "/tmp/results"

    #     # Create an instance of the Simulation class with the mock configuration
    #     simulation = Simulation(config_path=mock_config)

    #     # Call the method to test
    #     simulation.ensure_directories_exist()

    #     # Assert that os.makedirs was called with the correct paths and exist_ok=True
    #     mock_makedirs.assert_any_call("/tmp/cache", exist_ok=True)
    #     mock_makedirs.assert_any_call("/tmp/models", exist_ok=True)
    #     mock_makedirs.assert_any_call("/tmp/results", exist_ok=True)
    #     self.assertEqual(mock_makedirs.call_count, 3)


if __name__ == "__main__":
    unittest.main()
