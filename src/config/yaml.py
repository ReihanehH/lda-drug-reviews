import yaml
from pydantic import BaseModel, Field


class SamplingConfig(BaseModel):
    """
    Configuration class for sampling.

    Attributes:
        number (int): Number of samples. Default is -1.
        seed (int): Random seed. Default is 42.
    """

    number: int = Field(default=-1, description="Number of samples")
    seed: int = Field(default=42, description="Random seed")


class TrainConfig(BaseModel):
    """
    Configuration class for training.

    Attributes:
        path (str): Training data path.
        cache_dir_path (str): Cache directory path.
        enable_caching (bool): Flag to enable caching of cleaned train data.
        sampling (SamplingConfig): Sampling configuration.
    """

    cache_dir_path: str = Field(
        default="./cache", alias="cache-dir-path", description="Cache directory path"
    )
    path: str = Field(..., description="Training data path")
    enable_caching: bool = Field(
        default=True, alias="enable-caching", description="Cache cleaned train data"
    )
    sampling: SamplingConfig = Field(..., description="Sampling configuration")


class DataConfig(BaseModel):
    """
    Configuration class for data.

    Attributes:
        train (TrainConfig): Training configuration.
    """

    train: TrainConfig = Field(..., description="Training configuration")


class DTMConfig(BaseModel):
    """
    Configuration class for Document-Term Matrix (DTM).

    Attributes:
        min_df_threshold (float): Minimum document frequency threshold.

    """

    min_df_threshold: float = Field(
        default=0.005,
        alias="min-df-threshold",
        description="Minimum document frequency threshold",
    )


class LDAConfig(BaseModel):
    """
    Configuration for LDA (Latent Dirichlet Allocation) model.

    This class represents the configuration parameters for the LDA model. It provides
    attributes to control various aspects of the model, such as enabling training,
    specifying the number of topics, iterations, alpha value, and beta value.

    Attributes:
        enable_traning (bool): Flag to enable LDA training.
            Set this to True to enable training the LDA model, or False to disable training.
        models_dir_path (str): Models directory path.
        topics (int): Number of topics.
            Specifies the desired number of topics for the LDA model.
        iterations (int): Number of iterations.
            Specifies the number of iterations to perform during training.
        alpha (float): Alpha value.
            Specifies the alpha value for the LDA model.
        beta (float): Beta value.
            Specifies the beta value for the LDA model.
    """

    enable_traning: bool = Field(
        default=True, alias="enable-traning", description="Enable LDA training"
    )
    models_dir_path: str = Field(
        default="./models", alias="models-dir-path", description="Models directory path"
    )
    topics: int = Field(default=3, description="Number of topics")
    iterations: int = Field(default=5, description="Number of iterations")
    alpha: float = Field(default=0.1, description="Alpha value")
    beta: float = Field(default=0.01, description="Beta value")


class ResultConfig(BaseModel):
    """
    Configuration class for storing result settings.

    Attributes:
        results_dir_path (str): Results directory path.
        top_words (int): Number of top words to display. Default is 10.

    """

    results_dir_path: str = Field(
        default="./results",
        alias="results-dir-path",
        description="Results directory path",
    )
    top_words: int = Field(
        default=10, alias="top-words", description="Number of top words to display"
    )


class Config(BaseModel):
    """
    Configuration class for the application.

    Attributes:
        data (DataConfig): Data configuration.
        dtm (DTMConfig): DTM configuration.
        lda (LDAConfig): LDA configuration.
        result (ResultConfig): Result configuration.
    """

    data: DataConfig = Field(..., description="Data configuration")
    dtm: DTMConfig = Field(..., description="DTM configuration")
    lda: LDAConfig = Field(..., description="LDA configuration")
    result: ResultConfig = Field(..., description="Result configuration")

    class Config:
        env_file = ".env"


def load_yaml_config(file_path: str) -> Config:
    """
    Load YAML configuration from a file.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        Config: The configuration object.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    with open(file_path, "r") as file:
        _yaml = yaml.safe_load(file)
        return Config(**_yaml)
