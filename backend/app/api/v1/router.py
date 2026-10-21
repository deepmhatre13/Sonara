from fastapi import APIRouter

from app.api.v1 import health, maintenance, rag, recordings, search

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(recordings.router)
api_router.include_router(search.router)
api_router.include_router(rag.router)
api_router.include_router(maintenance.router)


