import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes import api_router
from app.services.model_manager import model_manager
from app.models.inference import SentimentModel, TopicModel, EmailModel, RecSysModel, TrendModel
from app.core.monitoring import (
    PrometheusMiddleware, 
    update_system_metrics,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ml-server")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML models...")
    # Đăng ký các model classes với ModelManager
    model_manager.register_model_class("sentiment", SentimentModel)
    model_manager.register_model_class("topic", TopicModel)
    model_manager.register_model_class("email", EmailModel)
    model_manager.register_model_class("recsys", RecSysModel)
    model_manager.register_model_class("trend", TrendModel)
    logger.info("Model classes registered successfully!")
    yield
    logger.info("Shutting down, releasing resources...")
    logger.info("Tắt ứng dụng, giải phóng tài nguyên...")
    # ModelManager sẽ tự động cleanup khi ứng dụng dừng

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for serving ML models (Sentiment, Topic, Email, RecSys, Trend)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(PrometheusMiddleware())

@app.get("/metrics")
async def metrics():
    """Endpoint để Prometheus scrape dữ liệu"""
    update_system_metrics()
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
app.include_router(api_router, prefix=settings.API_PREFIX)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)
