import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "NexusML Serving Pipeline"
    API_V1_STR: str = "/api"
    API_PREFIX: str = "/api"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1")
    CORS_ORIGINS: list = ["*"]  # Allow all origins for simplicity

    # Redis configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))

    # MinIO configuration
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_BUCKET: str = os.getenv("MINIO_BUCKET", "nexusml")

    # Model management configuration
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    MODEL_BASE_PATH: str = os.getenv("MODEL_BASE_PATH", "./models_cache") # Local cache for models
    MAX_CACHED_MODELS: int = int(os.getenv("MAX_CACHED_MODELS", 3)) # Max number of models to keep in cache

    # Performance optimization
    WORKERS_PER_MODEL: int = int(os.getenv("WORKERS_PER_MODEL", 2))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", 4))
    MODEL_TIMEOUT: int = int(os.getenv("MODEL_TIMEOUT", 30)) # seconds

    # API Security
    API_KEY_NAME: str = "X-API-Key"
    API_KEY: str = os.getenv("API_KEY", "your-super-secret-api-key") # Change in production

    class Config:
        case_sensitive = True
        env_file = ".env" # Load environment variables from .env file

settings = Settings()
