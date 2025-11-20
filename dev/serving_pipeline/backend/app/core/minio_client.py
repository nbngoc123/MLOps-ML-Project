import os
from minio import Minio
from minio.error import S3Error
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class MinIOClient:
    def __init__(self):
        self.client = Minio(
            settings.MINIO_ENDPOINT.replace("http://", "").replace("https://", ""),
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=False # Use True for HTTPS
        )
        self.bucket_name = settings.MINIO_BUCKET

    def download_file(self, object_name: str, file_path: str):
        """
        Tải một file từ MinIO về đường dẫn cục bộ.
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            self.client.fget_object(self.bucket_name, object_name, file_path)
            logger.info(f"Successfully downloaded '{object_name}' from MinIO to '{file_path}'")
            return True
        except S3Error as err:
            logger.error(f"Error downloading '{object_name}' from MinIO: {err}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while downloading '{object_name}': {e}")
            return False

    def list_objects(self, prefix: str = "", recursive: bool = False):
        """
        Liệt kê các đối tượng trong một bucket với tiền tố nhất định.
        """
        try:
            objects = self.client.list_objects(self.bucket_name, prefix=prefix, recursive=recursive)
            return [obj.object_name for obj in objects]
        except S3Error as err:
            logger.error(f"Error listing objects in MinIO with prefix '{prefix}': {err}")
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred while listing objects: {e}")
            return []

minio_client = MinIOClient()
