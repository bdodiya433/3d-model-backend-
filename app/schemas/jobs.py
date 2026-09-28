from typing import List, Optional, Dict
from pydantic import BaseModel, Field, AliasChoices, ConfigDict


class UploadResponse(BaseModel):
    """
    Response schema returned after a single image upload.
    """
    image_id: str = Field(..., description="Unique UUID identifier for the uploaded image")
    filename: str = Field(..., description="Standardized filename of the stored image")


class UploadMultiResponse(BaseModel):
    """
    Response schema returned after uploading multiple multi-view images (up to 6).
    """
    image_ids: List[str] = Field(..., description="List of unique UUID identifiers for uploaded images")
    filenames: List[str] = Field(..., description="List of filenames stored on server")


class GenerateRequest(BaseModel):
    """
    Request schema to initiate the 3D generation pipeline.
    Supports single-image (image_id), multi-view images (image_ids), and view_mapping ({'front': id, ...}),
    accepting both camelCase (from Flutter) and snake_case.
    """
    model_config = ConfigDict(populate_by_name=True)

    image_id: Optional[str] = Field(default=None, validation_alias=AliasChoices("image_id", "imageId"), description="Single image UUID identifier")
    image_ids: Optional[List[str]] = Field(default=None, validation_alias=AliasChoices("image_ids", "imageIds"), description="List of multi-view image UUIDs (front, back, left, right, top, bottom)")
    view_mapping: Optional[Dict[str, str]] = Field(default=None, validation_alias=AliasChoices("view_mapping", "viewMapping"), description="Dictionary mapping view names to image IDs")
    model_name: str = Field(default="hunyuan3d", validation_alias=AliasChoices("model_name", "modelName"), description="The AI Model Provider to run (e.g. hunyuan3d)")
    num_inference_steps: Optional[int] = Field(default=25, validation_alias=AliasChoices("num_inference_steps", "numInferenceSteps"), description="Number of diffusion inference steps (20-30 for high fidelity)")
    octree_resolution: Optional[int] = Field(default=256, validation_alias=AliasChoices("octree_resolution", "octreeResolution"), description="Octree grid resolution for mesh extraction (default 256)")
    enable_color: Optional[bool] = Field(default=True, validation_alias=AliasChoices("enable_color", "enableColor", "color"), description="Whether to extract and map colors from images. If False, generates classic Blender clay/neutral base model.")



class GenerateResponse(BaseModel):
    """
    Response schema containing details of the triggered generation job.
    """
    job_id: str = Field(..., description="Unique UUID identifier for the generation job")
    status: str = Field(..., description="Current state of the job (e.g. PENDING, PROCESSING)")
    created_at: str = Field(..., description="ISO 8601 timestamp of when the job was enqueued")


class JobStatusResponse(BaseModel):
    """
    Schema representing the granular status and progress of a background generation task.
    """
    job_id: str = Field(..., description="Unique UUID identifier for the generation job")
    status: str = Field(..., description="General job status (PENDING, PROCESSING, COMPLETED, FAILED)")
    progress: int = Field(..., description="Integer representation of progress (0 to 100)")
    stage: str = Field(..., description="Description of the current active pipeline stage")
    download_url: Optional[str] = Field(default=None, description="URL to download the completed 3D GLB asset")
    download_blend_url: Optional[str] = Field(default=None, description="URL to download the native Blender .blend project file")
    error: Optional[str] = Field(default=None, description="Detail message if the job failed")
