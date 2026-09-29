"""
Application Configuration and Settings Management
Loads parameters from environment and YAML configuration.
"""

import os
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Dict, Any, Optional

class Settings(BaseSettings):
    app_name: str = "CycloVision"
    app_version: str = "1.0.0"
    app_env: str = "development"
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000"
    ]
    demo_mode: bool = True
    device: str = "auto"
    config_path: str = "ml/configs/default.yaml"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

settings = Settings()

def load_yaml_config(path: Optional[str] = None) -> Dict[str, Any]:
    target_path = path or settings.config_path
    if not os.path.exists(target_path):
        return {}
    with open(target_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)