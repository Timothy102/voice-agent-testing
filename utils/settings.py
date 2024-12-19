from functools import lru_cache
from typing import Any, Dict

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Settings class for managing environment variables and configuration.

    This class extends BaseSettings from pydantic_settings to handle environment variables
    and configuration settings. It automatically loads values from environment variables
    and .env files.

    Attributes:
        settings (Dict[str, Any]): Dictionary containing all environment variables and settings
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Store all environment variables in a dictionary
        self.settings = {key: value for key, value in self.__dict__.items()}

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


@lru_cache()
def get_settings() -> Settings:
    """Get cached Settings instance.

    Returns:
        Settings: Cached Settings instance containing environment variables and configuration
    """
    return Settings()
