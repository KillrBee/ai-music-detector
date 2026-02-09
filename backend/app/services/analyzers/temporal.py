"""Temporal domain analyzer for AI-generated audio detection.

Checks:
1. Log-attack time (transient sharpness)
2. Rhythmic jitter (beat tracking confidence, micro-timing)
"""

import logging
from typing import Optional

import librosa
import numpy as np
from scipy import signal as scipy_signal

from .base import AnalysisDepth, ArtifactResult, BaseAnalyzer, DomainResult

logger = logging.getLogger(__name__)


class TemporalAnalyzer(BaseAnalyzer):
    """Analyzes temporal dynamics and rhythmic integrity for AI detection."""

    domain = "temporal"
    display_name = "Temporal Analysis"
    base_weight = 0.12
    min_depth = AnalysisDepth.STANDARD

    def analyze(
        self,
        y_mono: np.ndarray,
        sr: int,
        y_stereo: Optional[np.ndarray] = None,
        depth: AnalysisDepth = AnalysisDepth.STANDARD,
    ) -> DomainResult:
        artifacts = []

        try:
            artifacts.append(self._check_log_attack_time(y_mono, sr))
        except Exception as e:
            logger.warning("Log-attack time check failed: %s", e)

        try:
            artifacts.append(self._check_rhythmic_jitter(y_mono, sr))
        except Exception as e:
            logger.warning("Rhythmic jitter check failed: %s", e)

        return self._make_domain_result(artifacts)

    def _check_log_attack_time(self, y: np.ndarray, sr: int) -> ArtifactResult:
        """Measure transient attack sharpness via Log-Attack Time.

        AI models (especially diffusion-based) smear transients due to
        phase reconstruction errors. Real drums have sub-millisecond attacks;
        AI drums may have 10-20ms attacks.

        LAT = log10(attack_time_in_seconds)
        Real drums: LAT < -2.0 (< 10ms)
        AI drums: LAT > -1.5 (> 30ms)
        """
        # Detect onsets
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr, hop_length=512, backtrack=True
        )

        if len(onset_frames) < 5:
            return ArtifactResult(
                name="log_attack_time",
                detected=False,
                severity="none",
                value=None,
                description="Insufficient onsets for attack time analysis.",
                probability=0.0,
                weight=1.0,
            )

        onset_samples = librosa.frames_to_samples(onset_frames, hop_length=512)

        # Measure attack time for each onset
        attack_times = []
        search_window = int(sr * 0.05)  # 50ms search window for peak

        for onset_sample in onset_samples:
            peak_end = min(onset_sample + search_window, len(y))
            if peak_end <= onset_sample:
                continue

            segment = np.abs(y[onset_sample:peak_end])
            if len(segment) < 2:
                continue

            peak_idx = np.argmax(segment)
            if peak_idx == 0:
                continue

            # Attack time in seconds
            attack_time = peak_idx / sr

            # Only consider meaningful attacks (not noise)
            if segment[peak_idx] > np.mean(np.abs(y)) * 0.5:
                attack_times.append(attack_time)

        if len(attack_times) < 3:
            return ArtifactResult(
                name="log_attack_time",
                detected=False,
                severity="none",
                value=None,
                description="Insufficient significant transients for LAT analysis.",
                probability=0.0,
                weight=1.0,
            )

        # Compute median LAT
        attack_times = np.array(attack_times)
        # Clamp to avoid log(0)
        attack_times = np.maximum(attack_times, 1e-6)
        lat_values = np.log10(attack_times)
        median_lat = float(np.median(lat_values))

        # Data-driven thresholds from 22-track analysis:
        # All 22 tracks have LAT between -1.45 and -1.87. Zero discrimination.
        # Narrowed to only flag very slow attacks (>-1.2).
        if median_lat > -0.8:
            probability = 0.85
            severity = "high"
        elif median_lat > -1.2:
            probability = 0.5
            severity = "medium"
        else:
            probability = 0.0
            severity = "none"

        detected = median_lat > -1.2

        median_attack_ms = 10 ** median_lat * 1000

        return ArtifactResult(
            name="log_attack_time",
            detected=detected,
            severity=severity,
            value=round(median_lat, 2),
            description=(
                f"Median log-attack time: {median_lat:.2f} "
                f"(~{median_attack_ms:.1f}ms). "
                + (
                    "Slow transient attacks suggest phase smearing from "
                    "AI diffusion models or neural codec artifacts."
                    if detected
                    else "Sharp transient attacks consistent with real acoustic sources."
                )
            ),
            probability=probability,
            weight=1.0,
        )

    def _check_rhythmic_jitter(self, y: np.ndarray, sr: int) -> ArtifactResult:
        """Analyze beat tracking confidence and micro-timing characteristics.

        AI models produce uncorrelated random jitter in beat timing,
        unlike human groove (correlated, stylistic micro-timing) or
        DAW quantization (perfectly rigid grid).
        """
        # Beat tracking
        tempo, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr, hop_length=512
        )

        if len(beat_frames) < 8:
            return ArtifactResult(
                name="rhythmic_jitter",
                detected=False,
                severity="none",
                value=None,
                description="Insufficient beats detected for rhythmic analysis.",
                probability=0.0,
                weight=1.0,
            )

        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)

        # Inter-beat intervals
        ibis = np.diff(beat_times)
        if len(ibis) < 4:
            return ArtifactResult(
                name="rhythmic_jitter",
                detected=False,
                severity="none",
                value=None,
                description="Too few beat intervals for jitter analysis.",
                probability=0.0,
                weight=1.0,
            )

        mean_ibi = float(np.mean(ibis))
        if mean_ibi < 1e-6:
            return ArtifactResult(
                name="rhythmic_jitter",
                detected=False,
                severity="none",
                value=None,
                description="Invalid beat intervals detected.",
                probability=0.0,
                weight=1.0,
            )

        ibi_cv = float(np.std(ibis) / mean_ibi)  # Coefficient of variation

        # Compute autocorrelation of IBI deviations
        # Real music has correlated micro-timing (positive autocorrelation)
        # AI has uncorrelated jitter (near-zero autocorrelation)
        ibi_deviations = ibis - mean_ibi
        if len(ibi_deviations) > 4:
            autocorr_lag1 = float(
                np.corrcoef(ibi_deviations[:-1], ibi_deviations[1:])[0, 1]
            )
        else:
            autocorr_lag1 = 0.0

        # Handle NaN from corrcoef
        if np.isnan(autocorr_lag1):
            autocorr_lag1 = 0.0

        # Beat tracking confidence from onset strength at beat positions
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        beat_strengths = onset_env[beat_frames[beat_frames < len(onset_env)]]
        if len(beat_strengths) > 0 and np.max(onset_env) > 0:
            beat_confidence = float(
                np.mean(beat_strengths) / np.max(onset_env)
            )
        else:
            beat_confidence = 0.5

        # Map to probability — multiple sub-indicators
        probabilities = []

        # Low beat tracking confidence
        if beat_confidence < 0.3:
            probabilities.append(0.7)
        elif beat_confidence < 0.5:
            probabilities.append(0.5)

        # High jitter with uncorrelated timing (AI signature)
        if ibi_cv > 0.15 and abs(autocorr_lag1) < 0.2:
            probabilities.append(0.7)
        elif ibi_cv > 0.1 and abs(autocorr_lag1) < 0.3:
            probabilities.append(0.4)

        # Perfectly rigid timing (suspicious but less so)
        if ibi_cv < 0.005:
            probabilities.append(0.3)

        # Near-zero autocorrelation with moderate jitter (AI uncorrelated noise)
        if 0.02 < ibi_cv < 0.15 and abs(autocorr_lag1) < 0.15:
            probabilities.append(0.5)

        probability = max(probabilities) if probabilities else 0.0

        if probability >= 0.6:
            severity = "high"
        elif probability >= 0.4:
            severity = "medium"
        elif probability >= 0.2:
            severity = "low"
        else:
            severity = "none"

        detected = probability >= 0.4

        bpm = 60.0 / mean_ibi if mean_ibi > 0 else 0

        return ArtifactResult(
            name="rhythmic_jitter",
            detected=detected,
            severity=severity,
            value=round(ibi_cv, 4),
            description=(
                f"Tempo: {bpm:.1f} BPM, IBI variation: {ibi_cv:.4f}, "
                f"beat confidence: {beat_confidence:.2f}, "
                f"timing autocorrelation: {autocorr_lag1:.2f}. "
                + (
                    "Uncorrelated rhythmic jitter or low beat confidence "
                    "suggests AI-generated timing rather than human groove."
                    if detected
                    else "Rhythmic timing consistent with human performance or DAW quantization."
                )
            ),
            probability=probability,
            weight=1.0,
        )
