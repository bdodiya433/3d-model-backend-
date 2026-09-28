import asyncio
import logging
from pathlib import Path
from typing import List, Optional
import uuid
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect
)
from fastapi.responses import FileResponse

from app.config.settings import Settings, get_settings
from app.schemas.responses import BaseResponse
from app.schemas.jobs import (
    UploadResponse,
    UploadMultiResponse,
    GenerateRequest,
    GenerateResponse,
    JobStatusResponse
)
from app.services.jobs import JobManager

logger = logging.getLogger("app.generation")
router = APIRouter(tags=["Generation"])


def get_job_manager(request: Request) -> JobManager:
    return request.app.state.job_manager


def get_job_manager_ws(websocket: WebSocket) -> JobManager:
    return websocket.app.state.job_manager


@router.post("/upload", response_model=BaseResponse[UploadResponse])
async def upload_image(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings)
) -> BaseResponse[UploadResponse]:
    """
    Accepts a single raw image, performs size/extension checks, and saves to uploads folder.
    """
    MAX_SIZE = 10 * 1024 * 1024
    content = await file.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds maximum 10MB limit")

    extension = Path(file.filename or "").suffix.lower()
    if extension not in [".png", ".jpg", ".jpeg", ".webp"]:
        raise HTTPException(status_code=400, detail="Invalid image type. Supported: png, jpg, jpeg, webp")

    image_id = str(uuid.uuid4())
    filename = f"{image_id}{extension}"
    target_path = settings.upload_dir / filename

    try:
        target_path.write_bytes(content)
        logger.info(f"Image uploaded successfully: {filename}")
        return BaseResponse(data=UploadResponse(image_id=image_id, filename=filename))
    except Exception as e:
        logger.error(f"Failed to write uploaded image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error saving image upload")


@router.post("/upload-multi", response_model=BaseResponse[UploadMultiResponse])
async def upload_multiple_images(
    files: List[UploadFile] = File(...),
    settings: Settings = Depends(get_settings)
) -> BaseResponse[UploadMultiResponse]:
    """
    Accepts up to 6 multi-view images (front, back, left, right, top, bottom),
    performs validation, and returns the list of image IDs.
    """
    if len(files) < 1 or len(files) > 6:
        raise HTTPException(status_code=400, detail="Multi-view upload requires between 1 and 6 image files.")

    MAX_SIZE = 10 * 1024 * 1024
    image_ids: List[str] = []
    filenames: List[str] = []

    for file in files:
        content = await file.read()
        if len(content) > MAX_SIZE:
            raise HTTPException(status_code=400, detail=f"File '{file.filename}' exceeds 10MB limit.")

        extension = Path(file.filename or "").suffix.lower()
        if extension not in [".png", ".jpg", ".jpeg", ".webp"]:
            raise HTTPException(status_code=400, detail=f"File '{file.filename}' has unsupported format.")

        img_id = str(uuid.uuid4())
        fname = f"{img_id}{extension}"
        target_path = settings.upload_dir / fname

        target_path.write_bytes(content)
        image_ids.append(img_id)
        filenames.append(fname)

    logger.info(f"Successfully uploaded {len(image_ids)} multi-view image(s).")
    return BaseResponse(data=UploadMultiResponse(image_ids=image_ids, filenames=filenames))


@router.post("/generate", response_model=BaseResponse[GenerateResponse])
async def generate_model(
    payload: GenerateRequest,
    request: Request,
    job_manager: JobManager = Depends(get_job_manager),
    settings: Settings = Depends(get_settings)
) -> BaseResponse[GenerateResponse]:
    """
    Triggers 3D reconstruction asynchronously. Supports single image (image_id)
    or multi-view images (image_ids: up to 6 pictures).
    """
    target_ids: List[str] = []
    if payload.image_ids and len(payload.image_ids) > 0:
        target_ids = payload.image_ids
    elif payload.image_id:
        target_ids = [payload.image_id]

    if not target_ids:
        raise HTTPException(status_code=400, detail="Provide at least one valid image_id or image_ids list.")

    # Verify files exist on disk
    for img_id in target_ids:
        upload_files = list(settings.upload_dir.glob(f"{img_id}.*"))
        if not upload_files:
            raise HTTPException(status_code=404, detail=f"Uploaded image source for ID '{img_id}' not found.")

    # Create job entry
    job_id = job_manager.create_job(
        image_ids=target_ids,
        view_mapping=payload.view_mapping,
        model_name=payload.model_name,
        num_inference_steps=payload.num_inference_steps or 30,
        octree_resolution=payload.octree_resolution or 384,
        enable_color=payload.enable_color if payload.enable_color is not None else True,
    )
    model_manager = request.app.state.model_manager
    asyncio.create_task(job_manager.start_generation_pipeline(job_id, model_manager))

    job_info = job_manager.get_job(job_id)
    if not job_info:
        raise HTTPException(status_code=500, detail="Failed to initialize job entry")

    return BaseResponse(
        data=GenerateResponse(
            job_id=job_id,
            status=job_info["status"],
            created_at=job_info["created_at"]
        )
    )


@router.get("/download/{job_id}")
async def download_model(
    job_id: str,
    format: str = Query("glb", description="Asset export format: 'glb' or 'blend'"),
    settings: Settings = Depends(get_settings),
    job_manager: JobManager = Depends(get_job_manager)
) -> FileResponse:
    """
    Serves the generated 3D output asset.
    Supports format='glb' (default) or format='blend' (Blender project file).
    """
    job_info = job_manager.get_job(job_id)
    if not job_info or job_info["status"] != "COMPLETED":
        raise HTTPException(status_code=400, detail="Job is not completed or does not exist")

    fmt = format.lower().strip()
    if fmt == "blend":
        output_path = settings.output_dir / f"{job_id}.blend"
        media_type = "application/x-blender"
        filename = f"model_{job_id}.blend"
    else:
        output_path = settings.output_dir / f"{job_id}.glb"
        media_type = "model/gltf-binary"
        filename = f"model_{job_id}.glb"

    if not output_path.exists():
        raise HTTPException(status_code=404, detail=f"Generated {fmt.upper()} asset file not found")

    return FileResponse(
        path=output_path,
        filename=filename,
        media_type=media_type
    )


@router.websocket("/progress/{job_id}")
async def progress_websocket(
    websocket: WebSocket,
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager_ws)
) -> None:
    """
    WebSocket endpoint streaming real-time progress telemetry from 5% to 100%.
    """
    await websocket.accept()
    logger.info(f"WebSocket client connected for progress updates on job: {job_id}")

    try:
        while True:
            job_info = job_manager.get_job(job_id)
            if not job_info:
                await websocket.send_json({"error": f"Job {job_id} not found"})
                break

            await websocket.send_json({
                "job_id": job_id,
                "status": job_info["status"],
                "progress": job_info["progress"],
                "stage": job_info["stage"],
                "download_url": job_info.get("download_url"),
                "download_blend_url": job_info.get("download_blend_url"),
                "error": job_info.get("error")
            })

            if job_info["status"] in ["COMPLETED", "FAILED"]:
                break

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job: {job_id}")
    except Exception as e:
        logger.error(f"Error streaming progress updates: {e}")
        try:
            await websocket.send_json({"error": "Stream processing error"})
        except Exception:
            pass
    finally:
        logger.info(f"WebSocket stream closed for job: {job_id}")
