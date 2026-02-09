"""Base classes for domain-specific audio analyzers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np

from .weights import get_artifact_tier, get_artifact_weight, get_domain_base_weight


class AnalysisDepth(IntEnum):
    """Analysis depth levels controlling which analyzers run."""
    QUICK = 0       # ~3-5 seconds: spectral + spatial + production (basic)
    STANDARD = 1    # ~10-15 seconds: + temporal + production (reverb)
    DEEP = 2        # ~30-45 seconds: + structural + vocal + watermark


@dataclass
class ArtifactResult:
    """Result from a single artifact detection check."""
    name: str
    detected: bool
    severity: str               # "none", "low", "medium", "high"
    value: Optional[float]
    description: str
    probability: float = 0.0    # 0.0-1.0, probability this indicates AI generation
    weight: float = 1.0         # relative importance within its domain
    domain: str = ""            # set by the parent analyzer
    tier: int = 3               # 1=definitive, 2=strong, 3=moderate, 4=weak


@dataclass
class DomainResult:
    """Aggregated result for one analysis domain."""
    domain: str                           # e.g., "spectral"
    display_name: str                     # e.g., "Spectral Analysis"
    score: float                          # 0.0-1.0 aggregate AI probability
    artifacts: list[ArtifactResult] = field(default_factory=list)
    weight: float = 0.0                   # domain weight for overall scoring
    active: bool = True                   # whether this domain ran / was applicable


class BaseAnalyzer(ABC):
    """Abstract base class for all domain analyzers."""

    domain: str = ""
    display_name: str = ""
    base_weight: float = 0.0
    min_depth: AnalysisDepth = AnalysisDepth.STANDARD

    @abstractmethod
    def analyze(
        self,
        y_mono: np.ndarray,
        sr: int,
        y_stereo: Optional[np.ndarray] = None,
        depth: AnalysisDepth = AnalysisDepth.STANDARD,
    ) -> DomainResult:
        """Run all checks in this domain.

        Args:
            y_mono: Mono audio signal as numpy array.
            sr: Sample rate.
            y_stereo: Optional stereo audio (shape: [2, samples]).
            depth: Analysis depth level.

        Returns:
            Aggregated DomainResult with all artifact checks.
        """
        ...

    def _severity_to_probability(self, severity: str) -> float:
        """Convert a severity string to a probability score."""
        return {
            "none": 0.0,
            "low": 0.25,
            "medium": 0.6,
            "high": 0.9,
        }.get(severity, 0.0)

    def _make_domain_result(self, artifacts: list[ArtifactResult]) -> DomainResult:
        """Aggregate individual artifact results into a DomainResult.

        Applies centralized weights and tiers from weights.py, then computes
        a blended domain score: 70% weighted mean + 30% max individual
        probability. This ensures that a single strong detection (e.g.,
        checkerboard at 0.90) isn't diluted by co-domain checks returning 0.0.
        """
        # Apply centralized weights and tiers (overrides any per-artifact value)
        for a in artifacts:
            a.domain = self.domain
            a.weight = get_artifact_weight(a.name)
            a.tier = get_artifact_tier(a.name)

        if not artifacts:
            return DomainResult(
                domain=self.domain,
                display_name=self.display_name,
                score=0.0,
                artifacts=[],
                weight=self._get_domain_weight(),
                active=False,
            )

        total_weight = sum(a.weight for a in artifacts)
        if total_weight > 0:
            weighted_mean = sum(a.probability * a.weight for a in artifacts) / total_weight
        else:
            weighted_mean = 0.0

        max_prob = max(a.probability for a in artifacts)

        # Blend: weighted mean provides robustness against single false positives,
        # max probability ensures strong detections aren't diluted away
        score = 0.7 * weighted_mean + 0.3 * max_prob

        return DomainResult(
            domain=self.domain,
            display_name=self.display_name,
            score=score,
            artifacts=artifacts,
            weight=self._get_domain_weight(),
            active=True,
        )

    def _inactive_result(self, reason: str = "") -> DomainResult:
        """Return an inactive result when this domain cannot run."""
        return DomainResult(
            domain=self.domain,
            display_name=self.display_name,
            score=0.0,
            artifacts=[],
            weight=self._get_domain_weight(),
            active=False,
        )

    def _get_domain_weight(self) -> float:
        """Get domain weight from centralized config, falling back to class attribute."""
        return get_domain_base_weight(self.domain) or self.base_weight
