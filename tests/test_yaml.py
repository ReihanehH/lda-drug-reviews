import os
import tempfile
import unittest

from pydantic import ValidationError
from yaml import YAMLError, dump, safe_dump

from src.config.yaml import (
    Config,
    DataConfig,
    DTMConfig,
    LDAConfig,
    ResultConfig,
    SamplingConfig,
    TrainConfig,
    load_yaml_config,
)


class TestConfig(unittest.TestCase):
    """
    The TestConfig class contains unit tests for various configuration classes and functions.
    It ensures that these classes and functions correctly initialize with valid parameters,
    raise appropriate exceptions with invalid parameters, and properly load and parse YAML
    configuration files. The tests cover attributes verification and exception handling for
    classes such as TrainConfig, DataConfig, DTMConfig, LDAConfig, ResultConfig, and the
    load_yaml_config function.
    """

    def test_sampling_config(self):
        """
        This method tests the SamplingConfig class to ensure it correctly initializes with valid parameters
        and raises appropriate exceptions with invalid parameters. It verifies the attributes 'number' and
        'seed' are set correctly and checks for ValidationError when an invalid 'number' parameter is provided.
        """
        config = SamplingConfig(number=10, seed=123)

        self.assertEqual(config.number, 10)
        self.assertEqual(config.seed, 123)

        with self.assertRaises(ValidationError):
            SamplingConfig(number="invalid", seed=123)

    def test_train_config(self):
        """
        This method tests the TrainConfig class to ensure it correctly initializes with valid parameters
        and raises appropriate exceptions with invalid parameters. It verifies the attributes 'cache_dir_path',
        'path', 'enable_caching', and 'sampling' are set correctly and checks for ValidationError when an
        invalid 'sampling' parameter is provided.
        """
        sampling_config = SamplingConfig(**{"number": 10, "seed": 123})
        config = TrainConfig(
            **{
                "cache-dir-path": "/tmp/cache",
                "path": "data/train",
                "enable-caching": False,
                "sampling": sampling_config,
            }
        )
        self.assertEqual(config.cache_dir_path, "/tmp/cache")
        self.assertEqual(config.path, "data/train")
        self.assertFalse(config.enable_caching)
        self.assertEqual(config.sampling, sampling_config)

        with self.assertRaises(ValidationError):
            TrainConfig(
                **{
                    "cache-dir-path": "/tmp/cache",
                    "path": "data/train",
                    "enable-caching": False,
                    "sampling": "invalid",
                }
            )

    def test_data_config(self):
        """
        This method tests the DataConfig class to ensure it correctly initializes with valid parameters
        and raises appropriate exceptions with invalid parameters. It verifies that the 'train' attribute
        is set correctly when provided with a valid TrainConfig instance and checks for ValidationError
        when an invalid 'train' parameter is provided.
        """
        sampling_config = SamplingConfig(number=10, seed=123)
        train_config = TrainConfig(
            path="data/train", enable_caching=False, sampling=sampling_config
        )
        config = DataConfig(train=train_config)
        self.assertEqual(config.train, train_config)

        with self.assertRaises(ValidationError):
            DataConfig(train="invalid")

    def test_dtm_config(self):
        """
        This method tests the DTMConfig class to ensure it correctly initializes with valid parameters.
        It verifies that the 'min_df_threshold' attribute is set correctly when provided with a valid
        configuration dictionary.
        """
        map_config = {"min-df-threshold": 0.01}
        config = DTMConfig(**map_config)
        self.assertEqual(config.min_df_threshold, 0.01)

    def test_lda_config(self):
        """
        This method tests the LDAConfig class to ensure it correctly initializes with valid parameters
        and raises appropriate exceptions with invalid parameters. It verifies the attributes 'enable_traning',
        'models_dir_path', 'topics', 'iterations', 'alpha', and 'beta' are set correctly and checks for
        ValidationError when an invalid 'topics' parameter is provided.
        """
        config = LDAConfig(
            **{
                "enable-traning": True,
                "models-dir-path": "/tmp/models",
                "topics": 5,
                "iterations": 100,
                "alpha": 0.5,
                "beta": 0.1,
            }
        )
        self.assertTrue(config.enable_traning)
        self.assertEqual(config.models_dir_path, "/tmp/models")
        self.assertEqual(config.topics, 5)
        self.assertEqual(config.iterations, 100)
        self.assertEqual(config.alpha, 0.5)
        self.assertEqual(config.beta, 0.1)

        with self.assertRaises(ValidationError):
            LDAConfig(
                **{
                    "enable-traning": True,
                    "models-dir-path": "/tmp/models",
                    "topics": 5.4,
                    "iterations": 100,
                    "alpha": 0.5,
                    "beta": 0.1,
                }
            )

    def test_result_config(self):
        """
        This method tests the ResultConfig class to ensure it correctly initializes with valid parameters
        and raises appropriate exceptions with invalid parameters. It verifies the attributes 'results_dir_path'
        and 'top_words' are set correctly and checks for ValidationError when an invalid 'top_words' parameter
        is provided.
        """
        config = ResultConfig(**{"results-dir-path": "/tmp/results", "top-words": 20})
        self.assertEqual(config.results_dir_path, "/tmp/results")
        self.assertEqual(config.top_words, 20)

        with self.assertRaises(ValidationError):
            ResultConfig(**{"results-dir-path": "/tmp/results", "top-words": "invalid"})

    def test_config(self):
        """
        This method tests the Config class to ensure it correctly initializes with valid parameters
        and raises appropriate exceptions with invalid parameters. It verifies that the attributes
        'data', 'dtm', 'lda', and 'result' are set correctly when provided with valid configuration
        instances and checks for ValidationError when an invalid 'data' parameter is provided.
        """
        sampling_config = SamplingConfig(**{"number": 10, "seed": 123})
        train_config = TrainConfig(
            **{
                "path": "data/train",
                "enable-caching": False,
                "sampling": sampling_config,
            }
        )
        data_config = DataConfig(train=train_config)
        dtm_config = DTMConfig(min_df_threshold=0.01)
        lda_config = LDAConfig(
            **{
                "enable-traning": True,
                "topics": 5,
                "iterations": 100,
                "alpha": 0.5,
                "beta": 0.1,
            }
        )
        result_config = ResultConfig(**{"top-words": 20})

        config = Config(
            **{
                "data": data_config,
                "dtm": dtm_config,
                "lda": lda_config,
                "result": result_config,
            }
        )
        self.assertEqual(config.data, data_config)
        self.assertEqual(config.dtm, dtm_config)
        self.assertEqual(config.lda, lda_config)
        self.assertEqual(config.result, result_config)

        with self.assertRaises(ValidationError):
            Config(
                **{
                    "data": "invalid",
                    "dtm": dtm_config,
                    "lda": lda_config,
                    "result": result_config,
                }
            )

    def test_load_yaml_config(self):
        """
        This method tests the load_yaml_config function to ensure it correctly loads and parses a YAML
        configuration file into a Config object. It verifies that the attributes of the Config object
        are set correctly based on the YAML content. The method also checks for appropriate exceptions
        when the YAML file is missing or contains invalid data.
        """
        yaml_content = {
            "data": {
                "train": {
                    "path": "data/train",
                    "enable-caching": False,
                    "cache-dir-path": "/tmp/cache",
                    "sampling": {"number": 10, "seed": 123},
                }
            },
            "dtm": {"min-df-threshold": 0.01},
            "lda": {
                "enable-traning": True,
                "models-dir-path": "/tmp/models",
                "topics": 5,
                "iterations": 100,
                "alpha": 0.5,
                "beta": 0.1,
            },
            "result": {"results-dir-path": "/tmp/results", "top-words": 20},
        }

        with tempfile.NamedTemporaryFile("w", delete=False) as temp_file:
            safe_dump(yaml_content, temp_file)
            temp_file.close()
            config = load_yaml_config(temp_file.name)
            self.assertEqual(config.data.train.path, "data/train")
            self.assertEqual(config.data.train.cache_dir_path, "/tmp/cache")
            self.assertFalse(config.data.train.enable_caching)
            self.assertEqual(config.data.train.sampling.number, 10)
            self.assertEqual(config.data.train.sampling.seed, 123)
            self.assertEqual(config.dtm.min_df_threshold, 0.01)
            self.assertTrue(config.lda.enable_traning)
            self.assertEqual(config.lda.models_dir_path, "/tmp/models")
            self.assertEqual(config.lda.topics, 5)
            self.assertEqual(config.lda.iterations, 100)
            self.assertEqual(config.lda.alpha, 0.5)
            self.assertEqual(config.lda.beta, 0.1)
            self.assertEqual(config.result.top_words, 20)
            self.assertEqual(config.result.results_dir_path, "/tmp/results")
            os.remove(temp_file.name)

        with self.assertRaises(FileNotFoundError):
            load_yaml_config("non_existent_file.yaml")

        with tempfile.NamedTemporaryFile("w", delete=False) as temp_file:
            faulty_yaml_string = safe_dump({"data": {"train": "invalid"}})
            temp_file.write(faulty_yaml_string)
            temp_file.close()
            with self.assertRaises(ValidationError):
                load_yaml_config(temp_file.name)
            os.remove(temp_file.name)


if __name__ == "__main__":
    unittest.main()
