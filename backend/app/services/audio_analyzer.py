"""Audio analysis orchestrator for AI-generated music detection.

Coordinates all domain-specific analyzers and produces a unified
detection result with weighted probabilistic scoring.
"""

import dataclasses
import logging
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

from .analyzers import (
    AnalysisDepth,
    WeightedScoringEngine,
    SpectralAnalyzer,
    TemporalAnalyzer,
    SpatialAnalyzer,
    StructuralAnalyzer,
    ProductionAnalyzer,
    VocalAnalyzer,
    WatermarkAnalyzer,
)
from .analyzers.base import BaseAnalyzer, DomainResult
from .analyzers.scoring import ScoringResult

logger = logging.getLogger(__name__)


class AudioAnalyzer:
    """Orchestrates AI detection analysis across all domains."""

    def __init__(self):
        self._analyzers: list[BaseAnalyzer] = [
            SpectralAnalyzer(),
            SpatialAnalyzer(),
            ProductionAnalyzer(),
            TemporalAnalyzer(),
            StructuralAnalyzer(),
            VocalAnalyzer(),
            WatermarkAnalyzer(),
        ]
        self._scoring_engine = WeightedScoringEngine()

    def analyze_file(
        self,
        audio_path: Path,
        depth: str = "standard",
    ) -> dict:
        """Analyze an audio file for AI generation indicators.

        Args:
            audio_path: Path to the audio file.
            depth: Analysis depth ("quick", "standard", "deep").

        Returns:
            Dict with full analysis results including domain breakdowns,
            overall score, confidence, and individual artifact details.
        """
        depth_enum = AnalysisDepth[depth.upper()]

        # Load audio
        y, sr = librosa.load(str(audio_path), sr=None, mono=False)

        # Handle stereo/mono
        if y.ndim == 1:
            y_mono = y
            channels = 1
            y_stereo = None
        else:
            y_mono = np.mean(y, axis=0)
            channels = y.shape[0]
            y_stereo = y

        duration = len(y_mono) / sr

        # Basic audio metrics
        peak = float(np.max(np.abs(y_mono)))
        peak_db = 20.0 * np.log10(peak + 1e-10)
        rms = float(np.sqrt(np.mean(y_mono ** 2)))
        rms_db = 20.0 * np.log10(rms + 1e-10)

        # Run domain analyzers
        scoring_result = self._run_analyzers(y_mono, y_stereo, sr, depth_enum)

        # Build result dict
        result = {
            "filename": audio_path.name,
            "duration_seconds": round(duration, 2),
            "sample_rate": sr,
            "channels": channels,
            "peak_db": round(peak_db, 1),
            "rms_db": round(rms_db, 1),
            # New scoring
            "overall_score": scoring_result.overall_score,
            "confidence": scoring_result.confidence,
            "confidence_value": scoring_result.confidence_value,
            "depth_used": depth,
            # Domain results
            "domain_results": [
                dataclasses.asdict(d) for d in scoring_result.domain_results
            ],
            # Flat artifact list
            "ai_artifacts": [
                dataclasses.asdict(a) for a in scoring_result.artifacts_flat
            ],
            "overall_ai_likelihood": scoring_result.overall_likelihood,
        }

        # Extract legacy quick-access values
        for artifact in scoring_result.artifacts_flat:
            if artifact.name == "spectral_rolloff":
                result["high_freq_cutoff_hz"] = artifact.value
            elif artifact.name == "phase_correlation":
                result["stereo_correlation"] = artifact.value

        return result

    def _run_analyzers(
        self,
        y_mono: np.ndarray,
        y_stereo: Optional[np.ndarray],
        sr: int,
        depth: AnalysisDepth,
    ) -> ScoringResult:
        """Run all applicable domain analyzers and score results.

        Analyzers are skipped if the requested depth is below their
        minimum depth requirement. Skipped analyzers return inactive
        DomainResults so their weight is redistributed.
        """
        domain_results: list[DomainResult] = []

        for analyzer in self._analyzers:
            if depth < analyzer.min_depth:
                # Depth too shallow for this analyzer
                domain_results.append(DomainResult(
                    domain=analyzer.domain,
                    display_name=analyzer.display_name,
                    score=0.0,
                    artifacts=[],
                    weight=analyzer.base_weight,
                    active=False,
                ))
                logger.debug(
                    "Skipping %s (requires %s, got %s)",
                    analyzer.domain, analyzer.min_depth.name, depth.name,
                )
                continue

            try:
                logger.info("Running %s analyzer...", analyzer.domain)
                result = analyzer.analyze(y_mono, sr, y_stereo, depth)
                domain_results.append(result)
                logger.info(
                    "%s: score=%.2f, active=%s, artifacts=%d",
                    analyzer.domain, result.score, result.active,
                    len(result.artifacts),
                )
            except Exception as e:
                logger.error("Analyzer %s failed: %s", analyzer.domain, e)
                domain_results.append(DomainResult(
                    domain=analyzer.domain,
                    display_name=analyzer.display_name,
                    score=0.0,
                    artifacts=[],
                    weight=analyzer.base_weight,
                    active=False,
                ))

        return self._scoring_engine.calculate(domain_results)


# Singleton instance
audio_analyzer = AudioAnalyzer()
