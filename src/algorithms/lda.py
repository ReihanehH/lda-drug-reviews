import time
from typing import Tuple

import numpy as np


class LDA:
    def __init__(
        self,
        num_topics: int,
        num_iterations: int = 1000,
        verbose: bool = True,
        alpha: float = 0.1,
        beta: float = 0.01,
        random_seed: int = 42,
    ):
        """
        Latent Dirichlet Allocation (LDA) model for topic modeling.

        Parameters:
        - num_topics: int
            The number of topics to be extracted from the documents.
        - num_iterations: int, optional (default=1000)
            The number of iterations for Gibbs sampling.
        - verbose: bool, optional (default=True)
            Whether to print progress during training.
        - alpha: float, optional (default=0.1)
            The hyperparameter for the Dirichlet prior on the per-document topic distributions.
        - beta: float, optional (default=0.01)
            The hyperparameter for the Dirichlet prior on the per-topic word distributions.
        """
        self.num_topics = num_topics
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.verbose = verbose
        self.random_seed = random_seed

        self.topic_word_counts: np.ndarray = None
        self.train_data_shape: Tuple[int, int] = None

    def fit(self, dtm: np.ndarray) -> None:
        """
        Fit the LDA model to the given document-term matrix.

        Parameters:
        - dtm: np.ndarray
            The document-term matrix representing the corpus.

        Returns:
        None
        """
        np.random.seed(self.random_seed)

        self.train_data_shape = dtm.shape

        num_docs, num_words = dtm.shape

        documents = [np.nonzero(doc)[0] for doc in dtm]

        doc_topic_counts = np.zeros((num_docs, self.num_topics)) + self.alpha
        self.topic_word_counts = np.zeros((self.num_topics, num_words)) + self.beta
        topic_counts = np.zeros(self.num_topics) + num_words * self.beta

        assignments = []
        # Randomly assign initial topics
        for d, doc in enumerate(documents):
            current_doc_topics = []
            for word in doc:
                topic = np.random.choice(self.num_topics)
                current_doc_topics.append(topic)
                doc_topic_counts[d, topic] += 1
                self.topic_word_counts[topic, word] += 1
                topic_counts[topic] += 1
            assignments.append(current_doc_topics)

        # Gibbs Sampling
        start_time = time.time()
        for it in range(self.num_iterations):
            if self.verbose:
                elapsed_time = time.time() - start_time
                print(
                    f"\rGibbs Sampling Iteration {it+1}/{self.num_iterations}, Elapsed Time: {int(elapsed_time)} seconds",
                    end="",
                    flush=True,
                )
            for d, doc in enumerate(documents):
                for i, word in enumerate(doc):
                    topic = assignments[d][i]
                    # Decrement counts
                    doc_topic_counts[d, topic] -= 1
                    self.topic_word_counts[topic, word] -= 1
                    topic_counts[topic] -= 1

                    # Calculate probabilities
                    topic_probs = (
                        doc_topic_counts[d] * self.topic_word_counts[:, word]
                    ) / topic_counts
                    topic_probs /= np.sum(topic_probs)

                    # Sample new topic
                    new_topic = np.random.choice(self.num_topics, p=topic_probs)
                    assignments[d][i] = new_topic

                    # Increment counts
                    doc_topic_counts[d, new_topic] += 1
                    self.topic_word_counts[new_topic, word] += 1
                    topic_counts[new_topic] += 1
        if self.verbose:
            print()

    def load_model(self, path: str) -> None:
        """
        Load a pre-trained LDA model from a file.

        Parameters:
        - path: str
            The path to the file containing the model.

        Returns:
        None
        """
        with open(path, "rb") as f:
            self.topic_word_counts = np.load(f)

    def save_model(self, path: str) -> None:
        """
        Save the trained LDA model to a file.

        Parameters:
        - path: str
            The path to save the model.

        Returns:
        None
        """
        with open(path, "wb") as f:
            np.save(f, self.topic_word_counts)

    def get_model_id(self) -> str:
        """
        Get a unique identifier for the LDA model.

        Returns:
        str
            The model identifier.
        """
        return f"lda_topics-{self.num_topics}_it-{self.num_iterations}_aplha-{self.alpha}_beta-{self.beta}"
