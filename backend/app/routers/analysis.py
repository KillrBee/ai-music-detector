"""Audio analysis API endpoints."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import aiofiles
from fastapi import APIRouter, HTTPException, File, UploadFile, Query, BackgroundTasks

from ..config import APP_VERSION
from ..models.schemas import (
    AnalysisDepth,
    AnalysisResponse,
    AIArtifactIndicator,
    DomainAnalysisResult,
)
from ..services.audio_analyzer import audio_analyzer

router = APIRouter(prefix="/api", tags=["analysis"])

SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}


def _cleanup_temp_file(file_path: Path):
    """Background task to clean up temporary file."""
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception:
        pass


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(
    file: UploadFile = File(...),
    depth: AnalysisDepth = Query(default=AnalysisDepth.STANDARD),
    background_tasks: BackgroundTasks = None,
):
    """Upload an audio file and analyze it for AI generation indicators.

    **Depth levels:**
    - `quick` (~3-5s): Spectral + spatial + production (basic)
    - `standard` (~10-15s): + temporal + production (reverb)
    - `deep` (~30-45s): + structural + vocal + watermark

    Supports: WAV, MP3, FLAC, OGG, M4A, AAC, WMA
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename required")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}",
        )

    temp_dir = Path(tempfile.gettempdir()) / "ai-audio-detector"
    temp_dir.mkdir(exist_ok=True)

    temp_filename = f"{uuid4()}_{file.filename}"
    temp_path = temp_dir / temp_filename

    try:
        async with aiofiles.open(temp_path, "wb") as f:
            content = await file.read()
            await f.write(content)

        result = audio_analyzer.analyze_file(
            audio_path=temp_path,
            depth=depth.value,
        )

        return AnalysisResponse(
            scan_id=str(uuid4()),
            analyzed_at=datetime.now(timezone.utc).isoformat(),
            tool_version=APP_VERSION,
            filename=result["filename"],
            duration_seconds=result["duration_seconds"],
            sample_rate=result["sample_rate"],
            channels=result["channels"],
            peak_db=result["peak_db"],
            rms_db=result["rms_db"],
            overall_score=result["overall_score"],
            confidence=result["confidence"],
            confidence_value=result["confidence_value"],
            depth_used=result["depth_used"],
            domain_results=[
                DomainAnalysisResult(
                    domain=d["domain"],
                    display_name=d["display_name"],
                    score=d["score"],
                    active=d["active"],
                    weight=d["weight"],
                    artifacts=[
                        AIArtifactIndicator(**a) for a in d.get("artifacts", [])
                    ],
                )
                for d in result.get("domain_results", [])
            ],
            ai_artifacts=[
                AIArtifactIndicator(**a) for a in result.get("ai_artifacts", [])
            ],
            overall_ai_likelihood=result.get("overall_ai_likelihood", "unknown"),
            high_freq_cutoff_hz=result.get("high_freq_cutoff_hz"),
            stereo_correlation=result.get("stereo_correlation"),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
    finally:
        if background_tasks:
            background_tasks.add_task(_cleanup_temp_file, temp_path)
        else:
            _cleanup_temp_file(temp_path)


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "ai-audio-detector"}
