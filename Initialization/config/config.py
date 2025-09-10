from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')
    input_dim: int
    hidden_layers: list
    num_classes: int

settings = Settings()
