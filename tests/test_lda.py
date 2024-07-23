import unittest

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from src.algorithms.lda import LDA


class TestLDA(unittest.TestCase):
    """
    The TestLDA class contains unit tests for the LDA (Latent Dirichlet Allocation) class.
    It ensures that the LDA class correctly initializes with valid parameters and performs
    topic modeling on a given corpus. The setUp method initializes the LDA instance and
    prepares a sample corpus along with its document-term matrix (DTM) for testing.
    The test_initialization method verifies that the LDA instance attributes are set correctly.
    """

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
        """
        This method tests the initialization of the LDA class. It verifies that the attributes
        'num_topics', 'num_iterations', 'alpha', 'beta', and 'verbose' are correctly set during
        the initialization of an LDA instance. The test ensures that these attributes match the
        expected values provided during the setup.
        """
        self.assertEqual(self.lda.num_topics, self.num_topics)
        self.assertEqual(self.lda.num_iterations, self.num_iterations)
        self.assertEqual(self.lda.alpha, self.alpha)
        self.assertEqual(self.lda.beta, self.beta)
        self.assertEqual(self.lda.verbose, self.verbose)

    def test_fit(self):
        """
        This method tests the fit method of the LDA class. It verifies that the LDA model
        correctly processes the document-term matrix (DTM) and updates its internal state.
        The test checks that the 'train_data_shape' attribute matches the shape of the input DTM,
        and that the 'topic_word_counts' attribute is not None and has the correct dimensions
        corresponding to the number of topics and the number of words in the DTM.
        """
        self.lda.fit(self.dtm)
        self.assertEqual(self.lda.train_data_shape, self.dtm.shape)
        self.assertIsNotNone(self.lda.topic_word_counts)
        self.assertEqual(self.lda.topic_word_counts.shape[0], self.num_topics)
        self.assertEqual(self.lda.topic_word_counts.shape[1], self.dtm.shape[1])

    def test_save_and_load_model(self):
        """
        This method tests the save_model and load_model methods of the LDA class. It verifies that
        the LDA model can be saved to a file and subsequently loaded from that file without loss
        of information. The test ensures that the 'topic_word_counts' attribute of the loaded model
        matches the 'topic_word_counts' attribute of the original model after fitting the DTM.
        """
        self.lda.fit(self.dtm)
        model_path = "/tmp/lda_model.npy"
        self.lda.save_model(model_path)

        lda_loaded = LDA(num_topics=self.num_topics)
        lda_loaded.load_model(model_path)

        np.testing.assert_array_almost_equal(
            self.lda.topic_word_counts, lda_loaded.topic_word_counts
        )

    def test_get_model_id(self):
        """
        This method tests the get_model_id method of the LDA class. It verifies that the method
        correctly generates a unique model identifier string based on the LDA model's parameters.
        The test checks that the generated model ID matches the expected format, which includes
        the number of topics, number of iterations, alpha, and beta values.
        """
        model_id = self.lda.get_model_id()
        expected_id = f"lda_topics-{self.num_topics}_it-{self.num_iterations}_aplha-{self.alpha}_beta-{self.beta}"
        self.assertEqual(model_id, expected_id)


if __name__ == "__main__":
    unittest.main()
