import uvicorn
from app.core.config import get_settings

if __name__ == "__main__":
    # Fetch environment configurations
    settings = get_settings()
    
    # Run the Uvicorn ASGI server
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        reload_dirs=["app", "backend/src"] if settings.DEBUG else None,
        reload_excludes=["temp/*", "outputs/*", "uploads/*", "debug/*", "*.glb", "*.blend", "*.png", "*.jpg"] if settings.DEBUG else None,
    )
