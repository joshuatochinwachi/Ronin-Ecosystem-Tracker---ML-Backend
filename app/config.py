"""
Configuration management for Ronin ML Backend
"""
from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Server Configuration
    PORT: int = 8000
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "https://ronin-ecosystem.vercel.app"
    ]
    
    # Ronin API Configuration (Base URL for the 13 endpoints)
    RONIN_API_BASE_URL: str = os.getenv(
        "RONIN_API_BASE_URL", 
        "https://web-production-4fae.up.railway.app"
    )
    
    # Caching Configuration
    CACHE_TTL_SECONDS: int = 300  # 5 minutes
    CACHE_DIR: str = "raw_data_cache"
    ENABLE_CACHE: bool = True
    
    # ML Models Configuration
    ML_MODELS_DIR: str = "ml_models"
    MODEL_RETRAIN_HOURS: int = 24
    
    # Feature Flags
    ENABLE_VOLUME_FORECAST: bool = True
    ENABLE_WHALE_ANALYSIS: bool = True
    ENABLE_CHURN_PREDICTION: bool = True
    ENABLE_ANOMALY_DETECTION: bool = True
    
    # API Timeouts
    HTTP_TIMEOUT_SECONDS: int = 30
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = False
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()

# Create necessary directories
os.makedirs(settings.CACHE_DIR, exist_ok=True)
os.makedirs(settings.ML_MODELS_DIR, exist_ok=True)