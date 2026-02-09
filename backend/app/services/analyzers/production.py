"""Production quality analyzer for AI-generated audio detection.

Checks:
1. Crest factor (peak/RMS dynamics and waveform shape)
2. Loudness range (EBU R128 dynamic variation)
3. Reverb tail analysis (decay characteristics)
"""

import logging
from typing import Optional

import librosa
import numpy as np
import pyloudnorm as pyln
from scipy import signal as scipy_signal
from scipy.optimize import curve_fit

from .base import AnalysisDepth, ArtifactResult, BaseAnalyzer, DomainResult

logger = logging.getLogger(__name__)


class ProductionAnalyzer(BaseAnalyzer):
    """Analyzes production quality metrics for AI detection."""

    domain = "production"
    display_name = "Production Analysis"
    base_weight = 0.15
    min_depth = AnalysisDepth.QUICK

    def analyze(
        self,
        y_mono: np.ndarray,
        sr: int,
        y_stereo: Optional[np.ndarray] = None,
        depth: AnalysisDepth = AnalysisDepth.STANDARD,
    ) -> DomainResult:
        artifacts = []

        try:
            artifacts.append(self._check_crest_factor(y_mono, sr))
        except Exception as e:
            logger.warning("Crest factor check failed: %s", e)

        try:
            artifacts.append(self._check_loudness_range(y_mono, sr))
        except Exception as e:
            logger.warning("Loudness range check failed: %s", e)

        # Reverb tail analysis only at STANDARD depth and above
        if depth >= AnalysisDepth.STANDARD:
            try:
                artifacts.append(self._check_reverb_tail(y_mono, sr))
            except Exception as e:
                logger.warning("Reverb tail check failed: %s", e)

        return self._make_domain_result(artifacts)

    def _check_crest_factor(self, y: np.ndarray, sr: int) -> ArtifactResult:
        """Check crest factor and waveform saturation characteristics.

        AI tracks often have low crest factor (4-6dB) with soft tanh-shaped
        saturation, while professional masters use precise look-ahead limiting.
        """
        peak = float(np.max(np.abs(y)))
        rms = float(np.sqrt(np.mean(y ** 2)))

        if rms < 1e-10 or peak < 1e-10:
            return ArtifactResult(
                name="crest_factor",
                detected=False,
                severity="none",
                value=None,
                description="Insufficient signal level for crest factor analysis.",
                probability=0.0,
                weight=1.0,
            )

        crest_factor_db = 20.0 * np.log10(peak / rms)

        # Check for tanh-shaped saturation
        # Compute histogram of sample values to detect soft-clipping shape
        has_tanh_saturation = False
        hist, bin_edges = np.histogram(np.abs(y), bins=100, range=(0, peak))
        # tanh saturation: heavy density around 0.6-0.8 of peak, sharp rolloff after
        if peak > 0.1:
            norm_bins = (bin_edges[:-1] + bin_edges[1:]) / 2.0 / peak
            upper_region = hist[(norm_bins > 0.6) & (norm_bins < 0.85)]
            peak_region = hist[norm_bins > 0.9]
            if len(upper_region) > 0 and len(peak_region) > 0:
                if np.mean(upper_region) > np.mean(peak_region) * 3:
                    has_tanh_saturation = True

        # Map to probability
        if crest_factor_db < 4.0:
            probability = 0.85
            severity = "high"
        elif crest_factor_db < 6.0:
            probability = 0.6
            severity = "medium"
        elif crest_factor_db < 8.0:
            probability = 0.3
            severity = "low"
        else:
            probability = 0.0
            severity = "none"

        # Tanh saturation bonus
        if has_tanh_saturation and probability > 0.1:
            probability = min(1.0, probability + 0.2)
            if severity == "none":
                severity = "low"

        detected = probability >= 0.3

        desc = f"Crest factor: {crest_factor_db:.1f} dB"
        if has_tanh_saturation:
            desc += " with tanh-shaped soft saturation"
        if detected:
            desc += ". Low dynamic range with soft waveform typical of AI generation."
        else:
            desc += ". Dynamic range consistent with professional production."

        return ArtifactResult(
            name="crest_factor",
            detected=detected,
            severity=severity,
            value=round(crest_factor_db, 1),
            description=desc,
            probability=probability,
            weight=1.0,
        )

    def _check_loudness_range(self, y: np.ndarray, sr: int) -> ArtifactResult:
        """Check EBU R128 loudness range for flat dynamics.

        AI tracks often have extremely low loudness range (<3 LU),
        meaning verse and chorus are equally loud with no dynamic arc.
        """
        # pyloudnorm requires float64
        y_64 = y.astype(np.float64)

        meter = pyln.Meter(sr)

        # Compute short-term loudness in overlapping windows
        block_size = int(3.0 * sr)  # 3-second blocks
        hop_size = int(1.0 * sr)    # 1-second hop

        if len(y_64) < block_size:
            return ArtifactResult(
                name="loudness_range",
                detected=False,
                severity="none",
                value=None,
                description="Audio too short for loudness range analysis (need >3 seconds).",
                probability=0.0,
                weight=1.0,
            )

        short_term_loudness = []
        for start in range(0, len(y_64) - block_size, hop_size):
            block = y_64[start : start + block_size]
            try:
                loudness = meter.integrated_loudness(block)
                if loudness > -70.0:  # Gate out silence
                    short_term_loudness.append(loudness)
            except Exception:
                pass

        if len(short_term_loudness) < 5:
            return ArtifactResult(
                name="loudness_range",
                detected=False,
                severity="none",
                value=None,
                description="Insufficient non-silent blocks for loudness range analysis.",
                probability=0.0,
                weight=1.0,
            )

        stl = np.array(short_term_loudness)
        # LRA approximation: difference between 95th and 10th percentile
        lra = float(np.percentile(stl, 95) - np.percentile(stl, 10))

        # Data-driven thresholds from 22-track analysis:
        # 8/12 AI tracks have LRA < 5.0 LU; 8/10 human tracks >= 5.0 LU.
        if lra < 3.5:
            probability = 0.75
            severity = "high"
        elif lra < 5.0:
            probability = 0.50
            severity = "medium"
        elif lra < 7.0:
            probability = 0.15
            severity = "low"
        else:
            probability = 0.0
            severity = "none"

        detected = lra < 5.0

        return ArtifactResult(
            name="loudness_range",
            detected=detected,
            severity=severity,
            value=round(lra, 1),
            description=(
                f"Loudness Range (LRA): {lra:.1f} LU. "
                + (
                    "Extremely flat dynamics — verse and chorus are equally loud, "
                    "typical of AI-generated audio."
                    if detected
                    else "Healthy dynamic variation consistent with professional production."
                )
            ),
            probability=probability,
            weight=1.0,
        )

    def _check_reverb_tail(self, y: np.ndarray, sr: int) -> ArtifactResult:
        """Analyze reverb tail decay characteristics.

        Natural reverb follows exponential decay (RT60 model). AI-generated
        reverb often morphs, gates abruptly, or has non-monotonic decay.
        """
        # Find decay regions: detect onsets, then analyze the decay after each
        onsets = librosa.onset.onset_detect(
            y=y, sr=sr, units="samples", hop_length=512
        )

        if len(onsets) < 3:
            return ArtifactResult(
                name="reverb_tail_analysis",
                detected=False,
                severity="none",
                value=None,
                description="Insufficient onsets for reverb tail analysis.",
                probability=0.0,
                weight=1.0,
            )

        # Analyze decay after each onset
        r_squared_values = []
        abrupt_cutoffs = 0
        window_ms = 500  # Analyze 500ms of decay
        window_samples = int(sr * window_ms / 1000)
        frame_ms = 10  # 10ms analysis frames
        frame_samples = int(sr * frame_ms / 1000)

        for onset_sample in onsets:
            # Find the peak after onset
            peak_search_end = min(onset_sample + int(sr * 0.05), len(y))
            if peak_search_end <= onset_sample:
                continue

            peak_idx = onset_sample + np.argmax(
                np.abs(y[onset_sample:peak_search_end])
            )

            # Extract decay region
            decay_start = peak_idx
            decay_end = min(decay_start + window_samples, len(y))

            if decay_end - decay_start < frame_samples * 5:
                continue

            # Compute envelope in frames
            envelope = []
            for i in range(decay_start, decay_end - frame_samples, frame_samples):
                frame_rms = np.sqrt(np.mean(y[i : i + frame_samples] ** 2))
                envelope.append(frame_rms)

            if len(envelope) < 5:
                continue

            envelope = np.array(envelope)
            if envelope[0] < 1e-10:
                continue

            # Normalize envelope
            envelope_norm = envelope / envelope[0]

            # Only analyze if energy actually decays significantly
            # (not sustained by other instruments in a dense mix)
            peak_to_end_db = 20 * np.log10(
                envelope_norm[-1] / (envelope_norm[0] + 1e-10) + 1e-10
            )
            if peak_to_end_db > -18:
                continue  # Energy didn't decay enough — not an isolated reverb tail

            # Check for abrupt cutoff (>20dB drop in one frame)
            for j in range(1, len(envelope_norm)):
                if envelope_norm[j - 1] > 1e-5:
                    ratio = envelope_norm[j] / envelope_norm[j - 1]
                    if ratio < 0.1:  # >20dB drop
                        abrupt_cutoffs += 1
                        break

            # Fit exponential decay: y = exp(-t/tau)
            t = np.arange(len(envelope_norm), dtype=float)
            try:
                # Log-linear fit for exponential
                valid = envelope_norm > 1e-6
                if np.sum(valid) < 3:
                    continue
                log_env = np.log(envelope_norm[valid] + 1e-10)
                t_valid = t[valid]
                coeffs = np.polyfit(t_valid, log_env, 1)
                # R-squared
                predicted = coeffs[0] * t_valid + coeffs[1]
                ss_res = np.sum((log_env - predicted) ** 2)
                ss_tot = np.sum((log_env - np.mean(log_env)) ** 2)
                if ss_tot > 1e-10:
                    r2 = 1.0 - ss_res / ss_tot
                    r_squared_values.append(max(0.0, r2))
            except Exception:
                pass

        if not r_squared_values:
            return ArtifactResult(
                name="reverb_tail_analysis",
                detected=False,
                severity="none",
                value=None,
                description=(
                    "Dense arrangement — no isolated reverb tails for analysis. "
                    "All onsets had sustained energy from other instruments."
                ),
                probability=0.0,
                weight=1.0,
            )

        median_r2 = float(np.median(r_squared_values))
        cutoff_ratio = abrupt_cutoffs / max(len(onsets), 1)

        # Map R² to probability
        # Real studio reverb with compression/room characteristics typically
        # has R² of 0.3-0.7; only very low R² indicates AI morphing/gating
        if median_r2 < 0.2:
            probability = 0.8
            severity = "high"
        elif median_r2 < 0.4:
            probability = 0.5
            severity = "medium"
        elif median_r2 < 0.6:
            probability = 0.2
            severity = "low"
        else:
            probability = 0.0
            severity = "none"

        # Abrupt cutoff override
        if cutoff_ratio > 0.3:
            probability = max(probability, 0.7)
            severity = "high" if severity != "high" else severity

        detected = probability >= 0.3

        desc = f"Reverb decay fit R²: {median_r2:.2f}"
        if cutoff_ratio > 0.1:
            desc += f", abrupt cutoffs: {cutoff_ratio * 100:.0f}% of onsets"
        if detected:
            desc += ". Non-exponential reverb decay suggests AI-generated effects."
        else:
            desc += ". Natural exponential reverb decay consistent with real acoustics."

        return ArtifactResult(
            name="reverb_tail_analysis",
            detected=detected,
            severity=severity,
            value=round(median_r2, 3),
            description=desc,
            probability=probability,
            weight=1.0,  # Overridden by centralized weights.py
        )
