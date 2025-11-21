import os
import threading
from typing import Dict, Type
from collections import OrderedDict
from app.core.config import settings
from app.models._base import BaseModel
import logging
import mlflow

mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Quản lý việc tải và lưu trữ các model trong bộ nhớ đệm.
    Sử dụng chiến lược LRU (Least Recently Used) để quản lý bộ nhớ đệm.
    """
    def __init__(self):
        self.cache: Dict[str, BaseModel] = OrderedDict()
        self.lock = threading.Lock()
        self.model_classes: Dict[str, Type[BaseModel]] = {} # Đăng ký các lớp model

    def register_model_class(self, model_type: str, model_class: Type[BaseModel]):
        """
        Đăng ký một lớp model với một loại model cụ thể.
        """
        self.model_classes[model_type] = model_class
        logger.info(f"Registered model class for type '{model_type}'")

    def _get_model_key(self, model_name: str, model_version: str) -> str:
        return f"{model_name}-{model_version}"

    async def get_model(self, model_type: str, model_name: str, model_version: str) -> BaseModel:
        """
        Lấy một model từ bộ nhớ đệm hoặc tải nó nếu chưa có.
        """
        model_key = self._get_model_key(model_name, model_version)

        with self.lock:
            if model_key in self.cache:
                # Di chuyển model lên đầu danh sách (LRU)
                model = self.cache.pop(model_key)
                self.cache[model_key] = model
                logger.debug(f"Model '{model_key}' found in cache.")
                return model

            logger.info(f"Model '{model_key}' not in cache. Attempting to load...")

            # Kiểm tra xem loại model đã được đăng ký chưa
            model_class = self.model_classes.get(model_type)
            if not model_class:
                raise ValueError(f"Model type '{model_type}' not registered.")

            # Tạo một instance mới của model
            model_instance = model_class(model_name, model_version)

            # Gọi phương thức load() của model instance
            try:
                await model_instance.load()
                logger.info(f"Model '{model_name}' version '{model_version}' loaded successfully through model.load().")
            except Exception as e:
                logger.error(f"Error loading model '{model_name}' version '{model_version}': {e}")
                raise

            # ModelManager sẽ không tự load model nữa mà để cho model instance tự load thông qua phương thức load() của chúng.

            # Thêm model vào bộ nhớ đệm và quản lý giới hạn kích thước
            if len(self.cache) >= settings.MAX_CACHED_MODELS:
                # Loại bỏ model ít được sử dụng nhất (LRU)
                lru_key = next(iter(self.cache))
                del self.cache[lru_key]
                logger.info(f"Evicted model '{lru_key}' from cache (LRU).")
            self.cache[model_key] = model_instance
            logger.info(f"Model '{model_key}' loaded and added to cache.")
            return model_instance

model_manager = ModelManager()

# Logging setup (can be moved to a more central place if needed)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
