from fastapi import APIRouter
from .health import router as health_routes
from .sentiment import router as sentiment_routes
from .email import router as email_routes
from .recsys import router as recsys_routes
from .topic import router as topic_routes
from .trend import router as trend_routes

api_router = APIRouter()

api_router.include_router(health_routes, tags=["system"])
api_router.include_router(sentiment_routes, prefix="/sentiment", tags=["sentiment-analysis"])
api_router.include_router(email_routes, prefix="/email", tags=["email-analysis"])
api_router.include_router(recsys_routes, prefix="/recsys", tags=["recsys-analysis"])
api_router.include_router(topic_routes, prefix="/topic", tags=["topic-analysis"])
api_router.include_router(trend_routes, prefix="/trend", tags=["trend-analysis"])
