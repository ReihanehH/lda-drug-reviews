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

    def test_sampling_config(self):
        config = SamplingConfig(number=10, seed=123)
        self.assertEqual(config.number, 10)
        self.assertEqual(config.seed, 123)

        with self.assertRaises(ValidationError):
            SamplingConfig(number="invalid", seed=123)

    def test_train_config(self):
        sampling_config = SamplingConfig(**{"number": 10, "seed": 123})
        config = TrainConfig(
            **{
                "path": "data/train",
                "enable-caching": False,
                "sampling": sampling_config,
            }
        )
        self.assertEqual(config.path, "data/train")
        self.assertFalse(config.enable_caching)
        self.assertEqual(config.sampling, sampling_config)

        with self.assertRaises(ValidationError):
            TrainConfig(
                **{
                    "path": "data/train",
                    "enable-caching": False,
                    "sampling": "invalid",
                }
            )

    def test_data_config(self):
        sampling_config = SamplingConfig(number=10, seed=123)
        train_config = TrainConfig(
            path="data/train", enable_caching=False, sampling=sampling_config
        )
        config = DataConfig(train=train_config)
        self.assertEqual(config.train, train_config)

        with self.assertRaises(ValidationError):
            DataConfig(train="invalid")

    def test_dtm_config(self):
        map_config = {"min-df-threshold": 0.01}
        config = DTMConfig(**map_config)
        self.assertEqual(config.min_df_threshold, 0.01)

        # with self.assertRaises(ValidationError):
        #     DTMConfig(**{"min-df-threshold": 0.01})

    def test_lda_config(self):
        config = LDAConfig(
            **{
                "enable-traning": True,
                "topics": 5,
                "iterations": 100,
                "alpha": 0.5,
                "beta": 0.1,
            }
        )
        self.assertTrue(config.enable_traning)
        self.assertEqual(config.topics, 5)
        self.assertEqual(config.iterations, 100)
        self.assertEqual(config.alpha, 0.5)
        self.assertEqual(config.beta, 0.1)

        with self.assertRaises(ValidationError):
            LDAConfig(
                **{
                    "enable-traning": True,
                    "topics": 5.4,
                    "iterations": 100,
                    "alpha": 0.5,
                    "beta": 0.1,
                }
            )

    def test_result_config(self):
        config = ResultConfig(**{"top-words": 20})
        self.assertEqual(config.top_words, 20)

        with self.assertRaises(ValidationError):
            ResultConfig(**{"top-words": "invalid"})

    def test_config(self):
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
        yaml_content = {
            "data": {
                "train": {
                    "path": "data/train",
                    "enable-caching": False,
                    "sampling": {"number": 10, "seed": 123},
                }
            },
            "dtm": {"min-df-threshold": 0.01},
            "lda": {
                "enable-traning": True,
                "topics": 5,
                "iterations": 100,
                "alpha": 0.5,
                "beta": 0.1,
            },
            "result": {"top-words": 20},
        }

        with tempfile.NamedTemporaryFile("w", delete=False) as temp_file:
            safe_dump(yaml_content, temp_file)
            temp_file.close()
            config = load_yaml_config(temp_file.name)
            self.assertEqual(config.data.train.path, "data/train")
            self.assertFalse(config.data.train.enable_caching)
            self.assertEqual(config.data.train.sampling.number, 10)
            self.assertEqual(config.data.train.sampling.seed, 123)
            self.assertEqual(config.dtm.min_df_threshold, 0.01)
            self.assertTrue(config.lda.enable_traning)
            self.assertEqual(config.lda.topics, 5)
            self.assertEqual(config.lda.iterations, 100)
            self.assertEqual(config.lda.alpha, 0.5)
            self.assertEqual(config.lda.beta, 0.1)
            self.assertEqual(config.result.top_words, 20)
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
