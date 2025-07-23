import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class AppConfig:
    """Application configuration"""
    openai_api_key: str = ""
    database_path: str = "equipment.db"
    max_image_size_mb: int = 10
    max_sensor_history_days: int = 30
    default_equipment_types: list = None
    
    def __post_init__(self):
        if self.default_equipment_types is None:
            self.default_equipment_types = [
                "Centrifugal Pump",
                "Electric Motor", 
                "Air Compressor",
                "Heat Exchanger",
                "Valve",
                "Bearing",
                "Gearbox"
            ]

# Load from environment variables
def load_config() -> AppConfig:
    return AppConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        database_path=os.getenv("DATABASE_PATH", "equipment.db"),
        max_image_size_mb=int(os.getenv("MAX_IMAGE_SIZE_MB", "10")),
        max_sensor_history_days=int(os.getenv("MAX_SENSOR_HISTORY_DAYS", "30"))
    )