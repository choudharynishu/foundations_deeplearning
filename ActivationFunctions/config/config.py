from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Configuration Dictionary
    model_config = SettingsConfigDict(env_file='config/.env', env_file_encoding='utf-8')

    model_elu: str
    model_relu: str
    model_sigmoid: str
    model_lrelu: str
    model_tanh:str
    model_swish: str


settings = Settings()


