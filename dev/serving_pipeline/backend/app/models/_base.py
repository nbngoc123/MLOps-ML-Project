from abc import ABC, abstractmethod
from typing import Any

class BaseModel(ABC):
    """
    Giao diện cơ bản cho tất cả các model.
    Mỗi model cụ thể sẽ kế thừa từ lớp này và triển khai các phương thức trừu tượng.
    """
    def __init__(self, model_name: str, model_version: str):
        self.model_name = model_name
        self.model_version = model_version
        self.model = None

    @abstractmethod
    def load(self, model_path: str):
        """
        Tải model từ đường dẫn đã cho.
        """
        pass

    @abstractmethod
    def predict(self, data: Any) -> Any:
        """
        Thực hiện dự đoán với dữ liệu đầu vào.
        """
        pass

    def is_loaded(self) -> bool:
        """
        Kiểm tra xem model đã được tải chưa.
        """
        return self.model is not None
