from app.schemas.models import HealthResponse
from fastapi import Response
from app.core.monitoring import (
    REDIS_CONNECTED,
    MINIO_CONNECTED
)
from app.core.config import settings
from app.core.minio_client import minio_client
import redis
import logging

logger = logging.getLogger(__name__)

class HealthController:
    @classmethod
    async def check_health(cls):
        """Health check endpoint cho Kubernetes/Docker"""
        # Khởi tạo Redis client
        redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            socket_connect_timeout=5,
            socket_timeout=5,
            decode_responses=True
        )
        
        # Kiểm tra Redis connection
        redis_status = 0
        try:
            redis_client.ping()
            redis_status = 1
            logger.debug("Redis connection successful")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
        
        # Kiểm tra MinIO connection
        minio_status = 0
        try:
            # Kiểm tra bucket tồn tại
            minio_client.client.bucket_exists(settings.MINIO_BUCKET)
            minio_status = 1
            logger.debug("MinIO connection successful")
        except Exception as e:
            logger.error(f"MinIO connection failed: {e}")
        
        # Update Prometheus metrics
        REDIS_CONNECTED.set(redis_status)
        MINIO_CONNECTED.set(minio_status)
        
        # Return health status
        if redis_status and minio_status:
            return {
                "status": "healthy", 
                "components": {
                    "redis": "ok", 
                    "minio": "ok"
                },
                "timestamp": "2024-01-01T00:00:00Z"
            }
        else:
            components_status = {}
            components_status["redis"] = "ok" if redis_status else "failed"
            components_status["minio"] = "ok" if minio_status else "failed"
            
            return Response(
                status_code=503, 
                content={
                    "status": "unhealthy", 
                    "components": components_status,
                    "timestamp": "2024-01-01T00:00:00Z"
                }
            )
