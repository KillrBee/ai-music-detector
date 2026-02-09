"""FastAPI application entry point for AI Audio Detector."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import APP_VERSION, settings
from .routers import analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

app = FastAPI(
    title="AI Audio Detector",
    description="Detect AI-generated music using signal processing and spectral analysis",
    version=APP_VERSION,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(analysis.router)


@app.on_event("startup")
async def startup():
    """Ensure upload directory exists."""
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    logging.getLogger(__name__).info(
        "AI Audio Detector started. Upload dir: %s", settings.upload_dir
    )
