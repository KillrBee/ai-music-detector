"""Domain-specific audio analyzers for AI detection."""

from .base import AnalysisDepth, ArtifactResult, DomainResult, BaseAnalyzer
from .scoring import WeightedScoringEngine, ScoringResult
from .spectral import SpectralAnalyzer
from .temporal import TemporalAnalyzer
from .spatial import SpatialAnalyzer
from .structural import StructuralAnalyzer
from .production import ProductionAnalyzer
from .vocal import VocalAnalyzer
from .watermark import WatermarkAnalyzer
from .weights import (
    ArtifactTier,
    ArtifactWeightConfig,
    DomainWeightConfig,
    ARTIFACT_WEIGHTS,
    DOMAIN_WEIGHTS,
    TIER_LABELS,
    get_artifact_weight,
    get_artifact_tier,
    get_domain_base_weight,
)

__all__ = [
    "AnalysisDepth",
    "ArtifactResult",
    "DomainResult",
    "BaseAnalyzer",
    "WeightedScoringEngine",
    "ScoringResult",
    "SpectralAnalyzer",
    "TemporalAnalyzer",
    "SpatialAnalyzer",
    "StructuralAnalyzer",
    "ProductionAnalyzer",
    "VocalAnalyzer",
    "WatermarkAnalyzer",
    "ArtifactTier",
    "ArtifactWeightConfig",
    "DomainWeightConfig",
    "ARTIFACT_WEIGHTS",
    "DOMAIN_WEIGHTS",
    "TIER_LABELS",
    "get_artifact_weight",
    "get_artifact_tier",
    "get_domain_base_weight",
]
