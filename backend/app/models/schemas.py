"""Pydantic models for API request/response schemas."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class AnalysisDepth(str, Enum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class AIArtifactIndicator(BaseModel):
    """Individual artifact detection result."""
    name: str
    detected: bool
    severity: str  # "none", "low", "medium", "high"
    value: Optional[float] = None
    description: str
    probability: float = 0.0
    domain: str = ""
    weight: float = 1.0
    tier: int = 3  # 1=definitive, 2=strong, 3=moderate, 4=weak


class DomainAnalysisResult(BaseModel):
    """Results for one analysis domain."""
    domain: str
    display_name: str
    score: float  # 0.0-1.0
    active: bool = True
    weight: float = 0.0
    artifacts: list[AIArtifactIndicator] = []


class AnalysisRequest(BaseModel):
    """Request parameters for analysis."""
    depth: AnalysisDepth = AnalysisDepth.STANDARD


class AnalysisResponse(BaseModel):
    """Full analysis response with domain-grouped results."""
    # Scan metadata
    scan_id: str = ""                       # UUID for this scan
    analyzed_at: str = ""                   # ISO 8601 timestamp
    tool_version: str = "0.1.0"             # Software version

    # Audio properties
    filename: str
    duration_seconds: float
    sample_rate: int
    channels: int
    peak_db: float
    rms_db: float

    # AI detection scoring
    overall_score: float = 0.0              # 0-100 percentage
    confidence: str = "low"                 # "low", "medium", "high"
    confidence_value: float = 0.0           # 0.0-1.0
    depth_used: str = "standard"

    # Domain-grouped results
    domain_results: list[DomainAnalysisResult] = []

    # Flat artifact list
    ai_artifacts: list[AIArtifactIndicator] = []
    overall_ai_likelihood: str = "unknown"  # "unlikely", "possible", "likely"

    # Legacy quick-access fields
    high_freq_cutoff_hz: Optional[float] = None
    stereo_correlation: Optional[float] = None
