from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='config/.env', env_file_encoding='utf-8')
    data_dir: str
    input_dim: int
    hidden_layers: list[int]
    num_classes: int
    batchsize: int
    train_val_split: float
    learning_rate: float
    momentum: float
    max_epochs: int
    patience: int
    seed: int
    constant_init: float
    constant_var: float


settings = Settings()
