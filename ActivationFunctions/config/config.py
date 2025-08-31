from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Configuration Dictionary
    model_config = SettingsConfigDict(env_file='config/.env', env_file_encoding='utf-8')

    model_dir: str
    data_dir: str
    input_dimension: int
    hidden_layers: list[int]
    num_classes: int
    batchsize: int
    train_val_split: float
    learning_rate: float
    momentum: float
    max_epochs: int
    patience: int
    seed: int

settings = Settings()


