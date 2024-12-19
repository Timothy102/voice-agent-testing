from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Dict, Any

class Settings(BaseSettings):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Store all environment variables in a dictionary
        self.settings = {key: value for key, value in self.__dict__.items()}

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Allow arbitrary fields to be set from environment variables
        extra = "allow"

@lru_cache()
def get_settings() -> Settings:
    return Settings()