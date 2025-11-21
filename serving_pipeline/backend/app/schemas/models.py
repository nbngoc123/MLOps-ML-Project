from pydantic import BaseModel, Field
from typing import List, Optional
import datetime

class HealthResponse(BaseModel):
    status: str
    models_ready: bool
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)

# Các schema liên quan đến YOLO và Stable Diffusion đã được xóa theo yêu cầu của người dùng.
