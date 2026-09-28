from fastapi import APIRouter
from app.api.system import router as system_router
from app.api.generation import router as generation_router
from app.api.models import router as models_router

api_router = APIRouter()

# Mount system, generation, and models routers
api_router.include_router(system_router)
api_router.include_router(generation_router)
api_router.include_router(models_router)


