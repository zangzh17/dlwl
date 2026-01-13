"""
DOE Optimizer Web Frontend - FastAPI Application.

This is the main entry point for the web backend.
Run with: uvicorn web_frontend.backend.app:app --reload

# v3.1 - Added validation error logging
"""

# Must be done before importing torch anywhere
import os
if os.environ.get('FORCE_CPU', ''):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pathlib import Path

from .config import config
from .routes import wizard, validate, preview, optimize, export


# Create FastAPI app
app = FastAPI(
    title="DOE Optimizer",
    description="Web interface for Diffractive Optical Element optimization",
    version="3.0.0",
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Validation error handler for debugging
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Log validation errors with details for debugging."""
    errors = exc.errors()
    print(f"[VALIDATION ERROR] {request.url.path}")
    for error in errors:
        print(f"  - {error['loc']}: {error['msg']} (type: {error['type']})")
    return JSONResponse(
        status_code=422,
        content={"detail": errors}
    )


# Include API routers
app.include_router(wizard.router)
app.include_router(validate.router)
app.include_router(preview.router)
app.include_router(optimize.router)
app.include_router(export.router)

# Get frontend directory path
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


# Mount static files
if FRONTEND_DIR.exists():
    app.mount("/css", StaticFiles(directory=FRONTEND_DIR / "css"), name="css")
    app.mount("/js", StaticFiles(directory=FRONTEND_DIR / "js"), name="js")


@app.get("/")
async def serve_index():
    """Serve the main frontend page with auto-versioned static files.

    Automatically appends version query params based on file modification times.
    This ensures browser cache is invalidated when files change.
    """
    import re
    from fastapi.responses import HTMLResponse

    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return {"message": "DOE Optimizer API", "docs": "/docs"}

    html_content = index_path.read_text(encoding='utf-8')

    # Calculate version from max mtime of JS and CSS files
    max_mtime = 0
    for subdir in ["js", "css"]:
        dir_path = FRONTEND_DIR / subdir
        if dir_path.exists():
            for f in dir_path.glob("*"):
                if f.is_file():
                    max_mtime = max(max_mtime, f.stat().st_mtime)

    if max_mtime > 0:
        version = f"v={int(max_mtime)}"
        # Replace existing version strings (e.g., ?v=4.5 or ?v=1234567890)
        html_content = re.sub(r'\?v=[\d.]+', f'?{version}', html_content)
        # Also add version to CSS without existing version
        html_content = re.sub(
            r'(/css/styles\.css)(?!\?)',
            rf'\1?{version}',
            html_content
        )

    return HTMLResponse(content=html_content)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "3.0.0"}


@app.get("/api/device")
async def get_device_info():
    """Get current compute device information."""
    import torch

    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": "cpu",
        "device_name": "CPU",
    }

    if torch.cuda.is_available():
        try:
            # Test if CUDA is actually working
            test = torch.zeros(1, device='cuda')
            _ = test + 1
            torch.cuda.synchronize()
            del test

            info["current_device"] = "cuda"
            info["device_name"] = torch.cuda.get_device_name(0)
            info["cuda_memory_allocated"] = f"{torch.cuda.memory_allocated(0) / 1024**2:.1f} MB"
            info["cuda_memory_reserved"] = f"{torch.cuda.memory_reserved(0) / 1024**2:.1f} MB"
        except Exception as e:
            info["cuda_error"] = str(e)
            info["current_device"] = "cpu"
            info["device_name"] = "CPU (CUDA fallback)"

    return info


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    print(f"DOE Optimizer Web Frontend starting...")
    print(f"Max resolution: {config.max_resolution}")
    print(f"Max concurrent tasks: {config.max_concurrent_tasks}")
    print(f"Frontend directory: {FRONTEND_DIR}")

    # Test CUDA availability with robust error handling
    import torch
    import gc

    try:
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False

    if cuda_available:
        try:
            gc.collect()

            # Each operation wrapped individually
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            try:
                torch.cuda.init()
            except Exception:
                pass

            try:
                torch.cuda.synchronize()
            except Exception:
                print("CUDA sync failed, using CPU")
                return

            # Test actual CUDA usage
            test = torch.zeros(1, device='cuda')
            _ = test + 1
            torch.cuda.synchronize()
            device_name = torch.cuda.get_device_name(0)
            del test

            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            print(f"CUDA initialized: {device_name}")
        except Exception as e:
            print(f"CUDA initialization failed: {e}")
            print("Falling back to CPU mode")
    else:
        print("CUDA not available, using CPU")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("DOE Optimizer Web Frontend shutting down...")


# Development runner
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "web_frontend.backend.app:app",
        host=config.host,
        port=config.port,
        reload=True
    )
