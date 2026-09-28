from fastapi import APIRouter, Depends, Request
from app.ai.manager.model_manager import ModelManager

router = APIRouter(prefix="/models", tags=["Models"])


def get_model_manager(request: Request) -> ModelManager:
    """
    Dependency injection helper to retrieve the stateful ModelManager from application state.
    """
    return request.app.state.model_manager


@router.get("")
async def get_models(
    model_manager: ModelManager = Depends(get_model_manager)
) -> dict:
    """
    Returns operational status, device mapping, memory consumption,
    and availability for the Tencent Hunyuan3D-2 model provider.
    """
    return model_manager.get_status("hunyuan3d")


@router.get("/status")
async def get_models_status(
    model_manager: ModelManager = Depends(get_model_manager)
) -> dict:
    """
    Retrieves system memory consumption, running hardware device, and 
    individual model health statistics.
    """
    return {
        "device": model_manager.device,
        "memory": model_manager.get_memory_usage(),
        "status": model_manager.get_status("hunyuan3d")
    }
