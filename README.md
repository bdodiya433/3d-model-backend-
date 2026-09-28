# Production-Ready Image-to-3D AI Backend

This is a production-grade FastAPI backend for an Image-to-3D AI reconstruction platform (similar to Tripo AI), optimized for Apple Silicon (Mac M4, 16 GB RAM) and supporting standard Linux (CUDA) or CPU runtimes.

---

## Technical Highlights
- **App Factory Pattern**: Bootstraps the application via `create_app()` factory in `app/main.py`.
- **Abstract AI Model Contract**: Implemented using python `abc.ABC` to allow hot-swappable providers.
- **Model Lifecycle Orchestration**: `ModelManager` manages loading, caching, memory footprints, and graceful unloading of weights on shutdown.
- **Hardware Acceleration Autodetection**: Automatically detects and configures PyTorch hardware environments:
  - **Apple MPS (Metal Performance Shaders)** on Apple Silicon ARM64 macOS platforms.
  - **Nvidia CUDA** on standard graphics environments.
  - **CPU fallback** if no graphics accelerators are present.
- **Unified Memory (UMA) Diagnostics**: Gathers real-time RAM statistics using process `rss` and system memory footprints.
- **Structured Telemetry & Correlation**: Logs request timings, model loading times, and traces request pathways using a unique request ID.

---

## Directory Structure
```
3d_model_ai_Backend/
├── app/
│   ├── main.py              # Application entrypoint factory
│   ├── core/                # Core operations (logging, exceptions)
│   ├── config/              # Configuration (Pydantic settings)
│   ├── middleware/          # Request tracing middleware
│   ├── api/                 # Endpoint routers & handlers
│   ├── schemas/             # Pydantic response/validation models
│   ├── repositories/        # Storage data abstraction (DAL)
│   ├── services/            # Core business workflows
│   │   ├── image/           # Image enhancements/removals
│   │   ├── queue/           # Background tasks processing
│   │   ├── blender/         # Blender automated scripts
│   │   └── ai/              # AI subsystem
│   │       ├── base/        # Base abstract AIModel class
│   │       ├── loader/      # Models loader helper
│   │       ├── manager/     # ModelManager (lifecycle & hardware settings)
│   │       └── providers/   # Swappable model weight providers
│   │           ├── hunyuan/ # Hunyuan 3D provider
│   │           ├── triposr/ # TripoSR provider
│   │           └── instantmesh/# InstantMesh provider
│   ├── models/              # Schema definitions (DB/ORM)
│   ├── workers/             # Background queue workers
│   └── utils/               # Common helper utilities
├── uploads/                 # Storage for uploads
├── outputs/                 # Storage for exports
├── weights/                 # Model weight directories
├── logs/                    # System logs
├── temp/                    # Temp storage
├── tests/                   # Testing directories
├── requirements.txt         # Package requirements
├── .env                     # Local environment settings
└── README.md                # System documentation
```

---

## Environment Configuration
Configs are mapped in `.env`. Key settings include:
- `ENABLE_HUNYUAN`: Enables Hunyuan model provider (`true`/`false`).
- `ENABLE_TRIPOSR`: Enables TripoSR model provider (`true`/`false`).
- `ENABLE_INSTANTMESH`: Enables InstantMesh model provider (`true`/`false`).

---

## System Endpoints

### 1. Root Greeting
`GET /`

### 2. Operational Health Check
`GET /health` (Checks directory write accessibility and uptime)

### 3. Application Version
`GET /version` (Reads app name and debug mode)

### 4. Active Models Overview
`GET /models`
- **Output:** List of enabled and loaded model names.
  ```json
  {
    "success": true,
    "data": {
      "loaded_models": ["hunyuan", "triposr"],
      "enabled_models": ["hunyuan", "triposr"]
    },
    "error": null
  }
  ```

### 5. Detailed Models Status
`GET /models/status`
- **Output:** Individual model load durations, hardware device type, platform architecture type, and memory statistics.
  ```json
  {
    "success": true,
    "data": {
      "models": {
        "hunyuan": {
          "enabled": true,
          "loaded": true,
          "load_time_seconds": 0.4058,
          "health": true
        },
        "triposr": {
          "enabled": true,
          "loaded": true,
          "load_time_seconds": 0.2055,
          "health": true
        },
        "instantmesh": {
          "enabled": false,
          "loaded": false,
          "load_time_seconds": null,
          "health": false
        }
      },
      "memory_info": {
        "system_total_gb": 16.0,
        "system_available_gb": 3.12,
        "system_used_gb": 5.26,
        "system_percent": 80.5,
        "process_rss_gb": 0.05,
        "device": "mps",
        "architecture": "Unified Memory Architecture (UMA)"
      },
      "device": "mps"
    },
    "error": null
  }
  ```
