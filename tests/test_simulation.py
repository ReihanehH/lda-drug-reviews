import os
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

    @patch("os.makedirs")
    def test_ensure_directories_exist(self, mock_makedirs):
        """
        Test the ensure_directories_exist method of the Simulation class.

        This test verifies that the ensure_directories_exist method correctly creates
        the necessary directories for caching, models, and results. It uses the unittest.mock
        library to patch the os.makedirs function and check that it is called with the correct
        paths and the exist_ok=True argument.

        Args:
            mock_makedirs (MagicMock): Mocked version of os.makedirs.
        """
        # Create a mock configuration
        mock_config = MagicMock()
        mock_config.data.train.cache_dir_path = "/tmp/cache"
        mock_config.lda.models_dir_path = "/tmp/models"
        mock_config.result.results_dir_path = "/tmp/results"

        # Create an instance of the Simulation class with the mock configuration
        simulation = Simulation()
        simulation.config = mock_config

        # Call the method to test
        simulation.ensure_directories_exist()

        # Assert that os.makedirs was called with the correct paths and exist_ok=True
        mock_makedirs.assert_any_call("/tmp/cache", exist_ok=True)
        mock_makedirs.assert_any_call("/tmp/models", exist_ok=True)
        mock_makedirs.assert_any_call("/tmp/results", exist_ok=True)
        self.assertEqual(mock_makedirs.call_count, 3)

    @patch("src.simulation.simulation.LDA")
    def test_train_or_load_lda_model_training_enabled(self, MockLDA):
        """
        Test the train_or_load_lda_model method when training is enabled.

        This test verifies that the LDA model is trained and saved correctly when
        the enable_training configuration is set to True. It checks that the LDA
        instance is created with the correct parameters, the fit method is called
        with the document-term matrix, and the save_model method is called with the
        correct path. Additionally, it ensures that the returned LDA model is the
        mock instance.
        """
        # Create a mock LDA instance
        mock_lda = MockLDA.return_value
        mock_lda.get_model_id.return_value = "mock_model_id"

        # Create a mock configuration
        mock_config = MagicMock()
        mock_config.lda.topics = 10
        mock_config.lda.iterations = 100
        mock_config.lda.alpha = 0.1
        mock_config.lda.beta = 0.01
        mock_config.lda.models_dir_path = "/mock/path"
        mock_config.lda.enable_traning = True

        # Create a sample document-term matrix
        dtm = np.array([[1, 2, 3], [4, 5, 6]])

        # Create an instance of the Simulation class
        simulation_instance = Simulation()
        simulation_instance.config = mock_config

        # Call the method
        lda_model = simulation_instance.train_or_load_lda_model(dtm)

        # Check if the fit method was called with the document-term matrix
        mock_lda.fit.assert_called_once_with(dtm)

        # Check if the save_model method was called with the correct path
        expected_model_path = os.path.join("/mock/path", "mock_model_id.pkl")
        mock_lda.save_model.assert_called_once_with(expected_model_path)

        # Check if the returned lda_model is the mock instance
        self.assertEqual(lda_model, mock_lda)

    @patch("src.simulation.simulation.LDA")
    def test_train_or_load_lda_model_training_disabled(self, MockLDA):
        """
        Test the train_or_load_lda_model method when training is disabled.

        This test verifies that the LDA model is loaded correctly when the
        enable_training configuration is set to False. It checks that the LDA
        instance's load_model method is called with the correct path, and ensures
        that the fit and save_model methods are not called. Additionally, it
        confirms that the returned LDA model is the mock instance.
        """
        # Create a mock LDA instance
        mock_lda = MockLDA.return_value
        mock_lda.get_model_id.return_value = "mock_model_id"

        # Create a mock configuration
        mock_config = MagicMock()
        mock_config.lda.topics = 10
        mock_config.lda.iterations = 100
        mock_config.lda.alpha = 0.1
        mock_config.lda.beta = 0.01
        mock_config.lda.models_dir_path = "/mock/path"
        mock_config.lda.enable_traning = False

        # Create a sample document-term matrix
        dtm = np.array([[1, 2, 3], [4, 5, 6]])

        # Create an instance of the Simulation class
        simulation_instance = Simulation()
        simulation_instance.config = mock_config

        # Call the method
        lda_model = simulation_instance.train_or_load_lda_model(dtm)

        # Check if the load_model method was called with the correct path
        expected_model_path = os.path.join("/mock/path", "mock_model_id.pkl")
        mock_lda.load_model.assert_called_once_with(expected_model_path)

        # Check if the fit and save_model methods were not called
        mock_lda.fit.assert_not_called()
        mock_lda.save_model.assert_not_called()

        # Check if the returned lda_model is the mock instance
        self.assertEqual(lda_model, mock_lda)


if __name__ == "__main__":
    unittest.main()
