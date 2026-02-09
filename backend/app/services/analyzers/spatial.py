"""Spatial domain analyzer for AI-generated audio detection.

Checks:
1. Phase correlation (L/R relationship, mono compatibility)
2. Bass stereo width (low-frequency stereo placement)
"""

import logging
from typing import Optional

import numpy as np
from scipy import signal as scipy_signal

from .base import AnalysisDepth, ArtifactResult, BaseAnalyzer, DomainResult

logger = logging.getLogger(__name__)


class SpatialAnalyzer(BaseAnalyzer):
    """Analyzes stereo imaging and phase characteristics for AI detection."""

    domain = "spatial"
    display_name = "Spatial Analysis"
    base_weight = 0.10
    min_depth = AnalysisDepth.QUICK

    def analyze(
        self,
        y_mono: np.ndarray,
        sr: int,
        y_stereo: Optional[np.ndarray] = None,
        depth: AnalysisDepth = AnalysisDepth.STANDARD,
    ) -> DomainResult:
        if y_stereo is None or y_stereo.shape[0] != 2:
            return self._inactive_result("Mono audio — spatial analysis requires stereo.")

        artifacts = []
        try:
            artifacts.append(self._check_phase_correlation(y_stereo, sr))
        except Exception as e:
            logger.warning("Phase correlation check failed: %s", e)

        try:
            artifacts.append(self._check_bass_width(y_stereo, sr))
        except Exception as e:
            logger.warning("Bass width check failed: %s", e)

        return self._make_domain_result(artifacts)

    def _check_phase_correlation(
        self, y_stereo: np.ndarray, sr: int
    ) -> ArtifactResult:
        """Check stereo phase correlation for AI anomalies.

        AI audio often exhibits:
        - Negative phase correlation (anti-correlated channels)
        - Catastrophic mono cancellation (signal disappears when summed)
        - Extremely high correlation (near-mono, lazy stereo generation)

        Professional mixes maintain +0.5 to +1.0 correlation with
        good mono compatibility.
        """
        left = y_stereo[0]
        right = y_stereo[1]

        # Overall Pearson correlation
        correlation = float(np.corrcoef(left, right)[0, 1])

        # Mono summation test
        mid = (left + right) / 2.0
        rms_left = float(np.sqrt(np.mean(left ** 2)))
        rms_mid = float(np.sqrt(np.mean(mid ** 2)))

        mono_cancel_ratio = 0.0
        if rms_left > 1e-10:
            mono_cancel_ratio = 1.0 - (rms_mid / rms_left)

        # Per-band correlation analysis
        band_issues = 0
        for low, high, name in [(20, 200, "low"), (200, 4000, "mid"), (4000, min(sr // 2 - 100, 20000), "high")]:
            if high <= low:
                continue
            try:
                sos = scipy_signal.butter(3, [low, high], btype="band", fs=sr, output="sos")
                l_band = scipy_signal.sosfilt(sos, left)
                r_band = scipy_signal.sosfilt(sos, right)
                band_corr = float(np.corrcoef(l_band, r_band)[0, 1])
                if band_corr < -0.1:
                    band_issues += 1
            except Exception:
                pass

        # Map to probability (multiple sub-indicators, take max)
        probabilities = []

        # Negative correlation
        if correlation < -0.2:
            probabilities.append(0.7)
        elif correlation < 0.0:
            probabilities.append(0.5)

        # Catastrophic mono cancellation
        if mono_cancel_ratio > 0.5:
            probabilities.append(0.8)
        elif mono_cancel_ratio > 0.3:
            probabilities.append(0.5)

        # Extremely high correlation (near-mono stereo)
        if correlation > 0.99:
            probabilities.append(0.9)
        elif correlation > 0.97:
            probabilities.append(0.6)
        elif correlation > 0.95:
            probabilities.append(0.3)

        # Per-band issues
        if band_issues >= 2:
            probabilities.append(0.6)

        probability = max(probabilities) if probabilities else 0.0

        if probability >= 0.7:
            severity = "high"
        elif probability >= 0.4:
            severity = "medium"
        elif probability >= 0.2:
            severity = "low"
        else:
            severity = "none"

        detected = probability >= 0.3

        desc_parts = [f"Stereo correlation: {correlation:.3f}"]
        if mono_cancel_ratio > 0.3:
            desc_parts.append(
                f"mono cancellation: {mono_cancel_ratio * 100:.0f}%"
            )
        if correlation < 0.0:
            desc_parts.append("negative phase relationship detected")
        elif correlation > 0.97:
            desc_parts.append("near-mono stereo field")

        return ArtifactResult(
            name="phase_correlation",
            detected=detected,
            severity=severity,
            value=round(correlation, 4),
            description=". ".join(desc_parts) + ".",
            probability=probability,
            weight=1.0,
        )

    def _check_bass_width(self, y_stereo: np.ndarray, sr: int) -> ArtifactResult:
        """Check for improperly wide bass stereo image.

        Professional mixes center bass frequencies (<200Hz) in mono.
        AI models sometimes place significant bass energy in the side channel,
        creating a disorienting listening experience and speaker muddiness.
        """
        left = y_stereo[0]
        right = y_stereo[1]

        # Lowpass filter at 200Hz
        sos = scipy_signal.butter(4, 200, btype="low", fs=sr, output="sos")
        left_bass = scipy_signal.sosfilt(sos, left)
        right_bass = scipy_signal.sosfilt(sos, right)

        # Mid-side decomposition of bass
        mid_bass = (left_bass + right_bass) / 2.0
        side_bass = (left_bass - right_bass) / 2.0

        rms_mid = float(np.sqrt(np.mean(mid_bass ** 2)))
        rms_side = float(np.sqrt(np.mean(side_bass ** 2)))

        if rms_mid < 1e-10:
            return ArtifactResult(
                name="bass_stereo_width",
                detected=False,
                severity="none",
                value=None,
                description="Insufficient bass content for width analysis.",
                probability=0.0,
                weight=1.0,
            )

        bass_side_ratio = rms_side / rms_mid

        # Data-driven thresholds from 22-track analysis:
        # Human mean 0.21 > AI mean 0.08 — inversely correlated.
        # Wide bass is a professional mixing choice, not an AI artifact.
        # Only flag extreme values.
        if bass_side_ratio > 0.75:
            probability = 0.3
            severity = "low"
        else:
            probability = 0.0
            severity = "none"

        detected = bass_side_ratio > 0.75

        return ArtifactResult(
            name="bass_stereo_width",
            detected=detected,
            severity=severity,
            value=round(bass_side_ratio, 3),
            description=(
                f"Bass side/mid ratio: {bass_side_ratio:.3f}. "
                + (
                    "Significant stereo width in bass frequencies (<200Hz). "
                    "Professional mixes center bass; wide bass is typical of AI generation."
                    if detected
                    else "Bass properly centered in mono, consistent with professional mixing."
                )
            ),
            probability=probability,
            weight=1.0,
        )
