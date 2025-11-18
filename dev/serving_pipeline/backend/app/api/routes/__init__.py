from fastapi import APIRouter
# from .health import router as health_routes
from .sentiment import router as sentiment_routes

api_router = APIRouter()

# api_router.include_router(health_routes, tags=["system"])
api_router.include_router(sentiment_routes, prefix="/sentiment", tags=["sentiment-analysis"])
