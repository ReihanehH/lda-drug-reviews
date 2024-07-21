import unittest

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from src.algorithms.lda import LDA


class TestLDA(unittest.TestCase):
    def setUp(self):
        self.num_topics = 3
        self.num_iterations = 5
        self.alpha = 0.1
        self.beta = 0.01
        self.verbose = True
        self.lda = LDA(
            num_topics=self.num_topics,
            num_iterations=self.num_iterations,
            alpha=self.alpha,
            beta=self.beta,
            verbose=self.verbose,
        )

        # Example corpus
        self.corpus = [
            "apple banana apple",
            "banana fruit fruit banana",
            "apple fruit apple banana fruit",
        ]

        self.vectorizer = CountVectorizer()
        self.dtm = self.vectorizer.fit_transform(self.corpus).toarray()

    def test_initialization(self):
        self.assertEqual(self.lda.num_topics, self.num_topics)
        self.assertEqual(self.lda.num_iterations, self.num_iterations)
        self.assertEqual(self.lda.alpha, self.alpha)
        self.assertEqual(self.lda.beta, self.beta)
        self.assertEqual(self.lda.verbose, self.verbose)

    def test_fit(self):
        self.lda.fit(self.dtm)
        self.assertEqual(self.lda.train_data_shape, self.dtm.shape)
        self.assertIsNotNone(self.lda.topic_word_counts)
        self.assertEqual(self.lda.topic_word_counts.shape[0], self.num_topics)
        self.assertEqual(self.lda.topic_word_counts.shape[1], self.dtm.shape[1])

    def test_save_and_load_model(self):
        self.lda.fit(self.dtm)
        model_path = "/tmp/lda_model.npy"
        self.lda.save_model(model_path)

        lda_loaded = LDA(num_topics=self.num_topics)
        lda_loaded.load_model(model_path)

        np.testing.assert_array_almost_equal(
            self.lda.topic_word_counts, lda_loaded.topic_word_counts
        )

    def test_get_model_id(self):
        model_id = self.lda.get_model_id()
        expected_id = f"lda_topics-{self.num_topics}_it-{self.num_iterations}_aplha-{self.alpha}_beta-{self.beta}"
        self.assertEqual(model_id, expected_id)


if __name__ == "__main__":
    unittest.main()
