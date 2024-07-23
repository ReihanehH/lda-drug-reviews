import hashlib
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer

from src.algorithms.lda import LDA
from src.config.yaml import load_yaml_config


class Simulation:
    """
    A class representing a simulation.

    This class is responsible for running a simulation based on the provided configuration.
    It performs data loading, data cleaning, topic modeling, and result generation.

    Attributes:
        models_dir (str): The directory path for storing models.
        results_dir (str): The directory path for storing results.
        config (dict): The configuration dictionary.
    """

    def __init__(
        self,
        config_path: str = "./config/config.yaml",
    ) -> None:
        """
        Initialize the Simulation object.

        Args:
            config_path (str): The path to the configuration file. Defaults to "./config/config.yaml".
        Returns:
            None
        """

        # Load the configuration
        self.config = load_yaml_config(config_path)

    def clean_text_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the text data in the given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the text data.

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        stop_words = set(stopwords.words("english"))
        porter_stemmer = PorterStemmer()
        # Make texts to lowercase
        result = df.str.lower()
        # Replace the repeating pattern of '&#039;'
        result = result.str.replace("&#039;", "")
        # Remove all the special Characters
        result = result.str.replace(r"[^\w\d\s]", " ")
        # Remove all the non ASCII characters
        result = result.str.replace(r"[^\x00-\x7F]+", " ")
        # Remove the leading and trailing Whitespaces
        result = result.str.replace(r"^\s+|\s+?$", "")
        # Replace multiple Spaces with Single Space
        result = result.str.replace(r"\s+", " ")
        # Replace Two or more dots with one
        result = result.str.replace(r"\.{2,}", " ")
        # Remove words containing number
        result = result.str.replace(r"\w*\d\w*", "")
        # Remove stopwords
        result = result.apply(
            lambda x: " ".join(word for word in x.split() if word not in stop_words)
        )
        # Apply Porter stemming
        result = result.apply(
            lambda x: " ".join(porter_stemmer.stem(word) for word in x.split())
        )
        return result

    def create_train_df(
        self,
        dataset_path: str,
        num_of_samples: int = -1,
        frac_of_samples: float = -1.0,
        sampling_seed: int = 42,
        cache_enabled: bool = False,
        cache_dir_path: str = "./cache",
    ) -> pd.Series:
        """
        Create a clean training DataFrame.

        Args:
            dataset_path (str): The path to the dataset file.
            num_of_samples (int, optional): The number of samples to include in the training DataFrame. Defaults to -1.
            frac_of_samples (float, optional): The fraction of samples to include in the training DataFrame. Defaults to -1.0.
            sampling_seed (int, optional): The seed for random sampling. Defaults to 42.
            cache_enabled (bool, optional): Whether to enable caching. Defaults to False.
            cache_dir_path (str): The path to the cache directory file. Defaults to "./cache".

        Returns:
            pd.Series: The clean training Series.
        """

        if num_of_samples != -1 and frac_of_samples != -1.0:
            print(
                "ERROR: It's not possible to change both `num_of_samples` and `frac_of_samples`"
            )
            os.exit(1)

        clean_review_col_name = "clean_review"

        # Create a hash for the cache file that includes the dataset path, number of samples, fraction of samples, and sampling seed
        cache_hash = hashlib.sha1(
            f"{dataset_path}-{num_of_samples}-{frac_of_samples}-{sampling_seed}".encode()
        ).hexdigest()
        df_cache_path = os.path.join(cache_dir_path, f"{cache_hash}.pkl")

        # Check if the cache file exists and load it if it does
        if cache_enabled:
            if os.path.isfile(df_cache_path):
                print(f"Loading clean training dataframe from {df_cache_path}")
                df = pd.read_pickle(df_cache_path)
                return df[clean_review_col_name]
            else:
                print(f"Cache file named {df_cache_path} is not present")

        print("Loading and cleaning data from scratch (it may take a while)")
        # Load DataFrame from the dataset path and clean the review column and sample the data based on the given parameters
        df = pd.read_csv(dataset_path)
        df[clean_review_col_name] = self.clean_text_df(df["review"])
        if num_of_samples != -1:
            df = df.sample(n=num_of_samples, random_state=sampling_seed)
        elif frac_of_samples != -1.0:
            df = df.sample(frac=frac_of_samples, random_state=sampling_seed)

        # Group the reviews by drugName and concatenate them to reduce the number of words
        df = df.groupby("drugName")[clean_review_col_name].apply(" ".join).reset_index()

        # Save the clean DataFrame to the cache file
        if cache_enabled:
            print(f"Persisting train dataframe to {df_cache_path}")
            df.to_pickle(df_cache_path)

        return df[clean_review_col_name]

    def create_train_dtm(
        self, df: pd.DataFrame, min_df_threshold: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a Document-Term Matrix (DTM) from the given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the text data.
            min_df_threshold (int, optional): The minimum document frequency threshold for the CountVectorizer. Defaults to 0.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The Document-Term Matrix (DTM) and the vocabulary.
        """
        vectorizer = CountVectorizer(min_df=min_df_threshold)
        dtm = vectorizer.fit_transform(df).toarray()
        vocab = vectorizer.get_feature_names_out()
        return dtm, vocab

    def generate_results(
        self,
        topic_word_counts: np.ndarray,
        vocab: np.ndarray,
        image_path: str,
        image_text: str = "",
        num_of_top_words: int = 10,
    ) -> None:
        """
        Generate results based on the topic-word counts and vocabulary.

        Args:
            topic_word_counts (np.ndarray): The topic-word counts.
            vocab (np.ndarray): The vocabulary.
            image_path (str): The path to save the result image.
            image_text (str, optional): The text to be displayed in the image. Defaults to "".
            num_of_top_words (int, optional): The number of top words to display for each topic. Defaults to 10.

        Returns:
            None
        """
        # Calculate the beta value of each word in its topic
        twc = topic_word_counts / np.outer(
            np.sum(topic_word_counts, axis=1), np.ones(topic_word_counts.shape[1])
        )
        num_of_topics = twc.shape[0]

        topics = []

        # Create a DataFrame for each topic containing the top words and their beta values
        for topic_idx in range(num_of_topics):
            word_idxs = np.argsort(twc[topic_idx])[::-1][:num_of_top_words]
            words = []
            beta_values = []
            for word_idx in word_idxs:
                words.append(vocab[word_idx])
                beta_values.append(twc[topic_idx, word_idx])
            df = pd.DataFrame({"word": words, "beta": beta_values})
            df.set_index("word", inplace=True)
            topics.append(df)

        for df in topics:
            df["beta"] = df["beta"].map("{:.6f}".format)

        # Create a figure with subplots for each topic
        fig, axs = plt.subplots(1, num_of_topics, figsize=(4 * num_of_topics, 6))

        # If there's only one topic, axs is not a list, so we make it a list
        if num_of_topics == 1:
            axs = [axs]

        # Iterate over each topic DataFrame and corresponding subplot axis
        for i, (df, ax) in enumerate(zip(topics, axs)):
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set_frame_on(False)
            table = ax.table(
                cellText=df.values,
                colLabels=df.columns,
                rowLabels=df.index,
                cellLoc="center",
                loc="center",
            )

            ax.text(
                0,
                0.85,
                f"Topic {i + 1}",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                horizontalalignment="left",
            )

        # Add the simulation identifire text to the figure
        fig.text(0, 0, image_text)

        plt.tight_layout()

        # Save the figure and display it
        plt.savefig(image_path, bbox_inches="tight")
        img = Image.open(image_path)
        img.show()

    def run(self) -> None:
        """
        Run the simulation.

        Returns:
            None
        """

        # Make sure to have stopwords
        download("stopwords")

        # Make sure that cache, models and results directory exist.
        os.makedirs(self.config.data.train.cache_dir_path, exist_ok=True)
        os.makedirs(self.config.lda.models_dir_path, exist_ok=True)
        os.makedirs(self.config.result.results_dir_path, exist_ok=True)

        # Create a clean dataframe of reviews
        reviews_df = self.create_train_df(
            dataset_path=self.config.data.train.path,
            num_of_samples=self.config.data.train.sampling.number,
            sampling_seed=self.config.data.train.sampling.seed,
            cache_enabled=self.config.data.train.enable_caching,
            cache_dir_path=self.config.data.train.cache_dir_path,
        )

        # Create DTM and vocab
        dtm, vocab = self.create_train_dtm(
            df=reviews_df, min_df_threshold=self.config.dtm.min_df_threshold
        )
        print(f"DTM shape: {dtm.shape}")

        # Trainin LDA Model based on DTM
        lda = LDA(
            num_topics=self.config.lda.topics,
            num_iterations=self.config.lda.iterations,
            alpha=self.config.lda.alpha,
            beta=self.config.lda.beta,
        )
        lda_model_path = os.path.join(
            self.config.lda.models_dir_path, f"{lda.get_model_id()}.pkl"
        )
        if self.config.lda.enable_traning:
            lda.fit(dtm)
            lda.save_model(lda_model_path)
        else:
            lda.load_model(lda_model_path)

        # Set the path for the result image
        result_image_path = os.path.join(
            self.config.result.results_dir_path, f"{lda.get_model_id()}.png"
        )

        # Generate the results using the __generate_results method
        self.generate_results(
            topic_word_counts=lda.topic_word_counts,
            vocab=vocab,
            image_path=result_image_path,
            image_text=lda.get_model_id(),
            num_of_top_words=self.config.result.top_words,
        )
