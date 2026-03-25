from pathlib import Path
from typing import ClassVar
from pydantic_settings import BaseSettings, SettingsConfigDict

# Define paths to environment files
CURRENT_FILE = Path(__file__).resolve()
PACKAGE_ROOT = CURRENT_FILE.parent.parent.parent.parent
PROJECT_ROOT = PACKAGE_ROOT
ENV_FILES = [
    PROJECT_ROOT / ".env",
]


class GlobalSettings(BaseSettings):
    """A global settings singleton for managing project paths and configurations."""

    project_dir: Path = PROJECT_ROOT
    config_dir: Path = PROJECT_ROOT / "config"
    data_dir: Path = PROJECT_ROOT / "data"
    output_dir: Path = PROJECT_ROOT / "output"

    # Load environment variables from .env files
    model_config: ClassVar = SettingsConfigDict(
        env_file=ENV_FILES, env_ignore_empty=True, extra="ignore"
    )


# Instantiate a global singleton
settings = GlobalSettings()
