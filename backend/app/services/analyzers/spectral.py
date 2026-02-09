"""Spectral domain analyzer for AI-generated audio detection.

Checks:
1. Checkerboard artifacts (CNN upsampling grid patterns)
2. High-frequency harmonic-to-noise ratio (metallic HF harmonics)
3. Spectral rolloff (brick-wall cutoffs, aliasing)
4. Spectral flux variance (static/smeared mix transitions)
"""

import logging
from typing import Optional

import librosa
import numpy as np
from scipy import signal as scipy_signal

from .base import AnalysisDepth, ArtifactResult, BaseAnalyzer, DomainResult

logger = logging.getLogger(__name__)


class SpectralAnalyzer(BaseAnalyzer):
    """Analyzes frequency-domain characteristics for AI detection."""

    domain = "spectral"
    display_name = "Spectral Analysis"
    base_weight = 0.25
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
            artifacts.append(self._check_checkerboard(y_mono, sr))
        except Exception as e:
            logger.warning("Checkerboard check failed: %s", e)

        try:
            artifacts.append(self._check_hf_hnr(y_mono, sr))
        except Exception as e:
            logger.warning("HF-HNR check failed: %s", e)

        try:
            artifacts.append(self._check_spectral_rolloff(y_mono, sr))
        except Exception as e:
            logger.warning("Spectral rolloff check failed: %s", e)

        try:
            artifacts.append(self._check_spectral_flux(y_mono, sr))
        except Exception as e:
            logger.warning("Spectral flux check failed: %s", e)

        return self._make_domain_result(artifacts)

    def _check_checkerboard(self, y: np.ndarray, sr: int) -> ArtifactResult:
        """Detect periodic grid-like spectral artifacts from CNN upsampling.

        CNN checkerboard artifacts create alternating strong/weak energy
        across consecutive STFT frames. We detect this by analyzing the
        temporal periodicity of HF energy: compute mean HF energy per frame,
        take frame-to-frame differences, then FFT the difference signal
        to detect periodic alternation at the Nyquist rate.
        """
        S = np.abs(librosa.stft(y, n_fft=4096, hop_length=1024))

        freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
        # Focus on 4kHz-20kHz band
        band_mask = (freqs >= 4000) & (freqs <= min(20000, sr // 2))
        S_band = S[band_mask, :]

        if S_band.shape[0] < 16:
            return ArtifactResult(
                name="checkerboard_artifacts",
                detected=False,
                severity="none",
                value=None,
                description="Insufficient high-frequency content for checkerboard analysis.",
                probability=0.0,
                weight=1.0,
            )

        # Compute mean HF energy per frame (temporal profile)
        hf_energy_per_frame = np.mean(S_band, axis=0)

        # Frame-to-frame differences reveal alternation patterns
        frame_diffs = np.diff(hf_energy_per_frame)

        if len(frame_diffs) < 16:
            return ArtifactResult(
                name="checkerboard_artifacts",
                detected=False,
                severity="none",
                value=None,
                description="Audio too short for temporal checkerboard analysis.",
                probability=0.0,
                weight=1.0,
            )

        # FFT of the difference signal to find periodic alternation
        temporal_fft = np.abs(np.fft.rfft(frame_diffs))
        temporal_fft_no_dc = temporal_fft[1:]

        if len(temporal_fft_no_dc) < 4:
            return ArtifactResult(
                name="checkerboard_artifacts",
                detected=False,
                severity="none",
                value=None,
                description="Could not analyze temporal periodicity.",
                probability=0.0,
                weight=1.0,
            )

        # Checkerboard = strong energy at high temporal frequencies
        # (every-other-frame alternation appears near Nyquist)
        n = len(temporal_fft_no_dc)
        upper_quarter = temporal_fft_no_dc[3 * n // 4:]  # high-frequency periodicity
        lower_half = temporal_fft_no_dc[:n // 2]          # low-frequency periodicity

        mean_lower = float(np.mean(lower_half))
        if mean_lower < 1e-10:
            return ArtifactResult(
                name="checkerboard_artifacts",
                detected=False,
                severity="none",
                value=None,
                description="No significant temporal periodicity baseline.",
                probability=0.0,
                weight=1.0,
            )

        periodicity_ratio = float(np.max(upper_quarter) / mean_lower)

        # Data-driven thresholds from 22-track analysis:
        # Human ratios range 1.3-9.3; 5/10 humans were at 5.8-9.3 (false positives).
        # Only 1 AI track (11.8x) clearly exceeds human range.
        # Raised thresholds to reduce false positives.
        if periodicity_ratio > 7.0:
            probability = 0.9
            severity = "high"
        elif periodicity_ratio > 5.0:
            probability = 0.6
            severity = "medium"
        elif periodicity_ratio > 3.0:
            probability = 0.15
            severity = "low"
        else:
            probability = 0.0
            severity = "none"

        detected = probability > 0.0

        return ArtifactResult(
            name="checkerboard_artifacts",
            detected=detected,
            severity=severity,
            value=round(periodicity_ratio, 2),
            description=(
                f"Temporal periodicity ratio: {periodicity_ratio:.1f}x. "
                + (
                    "Periodic frame-to-frame alternation detected in high frequencies, "
                    "suggesting convolutional upsampling artifacts."
                    if detected
                    else "No significant checkerboard grid patterns found."
                )
            ),
            probability=probability,
            weight=1.0,  # Overridden by centralized weights.py
        )

    def _check_hf_hnr(self, y: np.ndarray, sr: int) -> ArtifactResult:
        """Check high-frequency harmonic-to-noise ratio.

        AI models often force harmonic structure into the HF band (>12kHz),
        creating a metallic/buzzy texture. Real audio has noise-dominated
        (airy) high frequencies.

        Uses HPSS to separate harmonics and noise in the 12-20kHz band,
        then computes HNR = 10 * log10(harmonic_energy / noise_energy).
        """
        nyquist = sr // 2
        if nyquist < 12000:
            return ArtifactResult(
                name="hf_harmonic_noise_ratio",
                detected=False,
                severity="none",
                value=None,
                description="Sample rate too low for HF analysis (need >24kHz).",
                probability=0.0,
                weight=1.0,
            )

        # Bandpass filter: 12kHz to Nyquist
        low_freq = 12000
        high_freq = min(20000, nyquist - 100)
        if high_freq <= low_freq:
            return ArtifactResult(
                name="hf_harmonic_noise_ratio",
                detected=False,
                severity="none",
                value=None,
                description="Insufficient bandwidth for HF-HNR analysis.",
                probability=0.0,
                weight=1.0,
            )

        sos = scipy_signal.butter(
            4, [low_freq, high_freq], btype="band", fs=sr, output="sos"
        )
        y_hf = scipy_signal.sosfilt(sos, y)

        # HPSS on the HF band
        S_hf = librosa.stft(y_hf, n_fft=2048, hop_length=512)
        S_harm, S_perc = librosa.decompose.hpss(S_hf)

        harmonic_energy = float(np.sum(np.abs(S_harm) ** 2))
        noise_energy = float(np.sum(np.abs(S_perc) ** 2))

        if noise_energy < 1e-20:
            hf_hnr = 30.0  # Effectively all harmonic
        else:
            hf_hnr = 10.0 * np.log10(harmonic_energy / noise_energy + 1e-20)

        # Data-driven thresholds from 22-track analysis:
        # AI HNR clusters at -0.0 to -7.0 dB; most humans below -7 dB.
        # One AI track at -0.0 dB (near boundary). Extended range to catch
        # the AI cluster while minimizing human false positives.
        if hf_hnr > 5.0:
            probability = 0.9
            severity = "high"
        elif hf_hnr > 0.0:
            probability = 0.7
            severity = "high"
        elif hf_hnr > -4.0:
            probability = 0.4
            severity = "medium"
        elif hf_hnr > -7.0:
            probability = 0.15
            severity = "low"
        else:
            probability = 0.0
            severity = "none"

        detected = hf_hnr > -7.0

        return ArtifactResult(
            name="hf_harmonic_noise_ratio",
            detected=detected,
            severity=severity,
            value=round(hf_hnr, 1),
            description=(
                f"HF Harmonic-to-Noise Ratio: {hf_hnr:.1f} dB (12-20kHz). "
                + (
                    "High harmonicity in upper frequencies indicates metallic/buzzy AI synthesis artifacts."
                    if detected
                    else "Natural noise-dominated high frequencies (airy texture)."
                )
            ),
            probability=probability,
            weight=1.0,
        )

    def _check_spectral_rolloff(self, y: np.ndarray, sr: int) -> ArtifactResult:
        """Detect unnatural spectral cutoffs and brick-wall rolloff.

        Uses librosa spectral_rolloff (85% energy concentration point) for
        the primary cutoff measurement, plus brick-wall detection by comparing
        energy in the top frequency band vs the band just below it.

        AI models often have hard bandwidth limits (e.g., 16kHz or 24kHz)
        caused by training data sample rate or neural codec limits.
        """
        # Use 95% energy rolloff point — more robust to genre differences
        # than 85%. Jazz/acoustic naturally concentrates energy lower, but
        # 95% rolloff still stays high for full-bandwidth recordings.
        # AI codec bandwidth limits show up clearly at 95%.
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=8192, hop_length=2048, roll_percent=0.95
        )
        median_rolloff = float(np.median(rolloff))

        if median_rolloff < 1.0:
            return ArtifactResult(
                name="spectral_rolloff",
                detected=False,
                severity="none",
                value=None,
                description="No significant spectral content detected.",
                probability=0.0,
                weight=1.0,
            )

        # Brick-wall detection: sharp energy drop near Nyquist
        nyquist = sr // 2
        S = np.abs(librosa.stft(y, n_fft=8192, hop_length=2048))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=8192)
        avg_spectrum = np.mean(S, axis=1)

        is_brick_wall = False
        if nyquist > 4000:
            top_mask = (freqs >= nyquist - 2000) & (freqs < nyquist)
            next_mask = (freqs >= nyquist - 4000) & (freqs < nyquist - 2000)
            top_energy = float(np.mean(avg_spectrum[top_mask])) if np.any(top_mask) else 0
            next_energy = float(np.mean(avg_spectrum[next_mask])) if np.any(next_mask) else 1e-20
            if next_energy > 1e-20:
                drop_db = 10 * np.log10(top_energy / next_energy + 1e-20)
                is_brick_wall = drop_db < -30  # >30dB drop = brick wall

        # Map to probability — brick-wall detection is the primary indicator.
        # Rolloff frequency alone is genre-dependent (jazz/acoustic concentrate
        # energy lower than pop/rock), so non-brick-wall only flags extreme cases.
        if is_brick_wall:
            # Hard bandwidth cutoff is a strong AI indicator
            if median_rolloff < 8000:
                probability = 0.95
                severity = "high"
            elif median_rolloff < 12000:
                probability = 0.7
                severity = "high"
            elif median_rolloff < 16000:
                probability = 0.4
                severity = "medium"
            else:
                probability = 0.0
                severity = "none"
        else:
            # Without brick-wall, rolloff frequency is genre-dependent
            # (jazz/acoustic naturally concentrates energy lower than rock/pop).
            # Only flag telephone-quality bandwidth (<2kHz at 95% energy).
            if median_rolloff < 2000:
                probability = 0.5
                severity = "medium"
            else:
                probability = 0.0
                severity = "none"

        detected = probability >= 0.3

        desc = f"Spectral rolloff (95% energy): {median_rolloff:.0f} Hz"
        if is_brick_wall and detected:
            desc += " (brick-wall rolloff). Abrupt cutoff suggests AI model bandwidth limitation."
        elif detected:
            desc += ". Frequency content concentrated below expected range for professional audio."
        else:
            desc += ". Normal frequency extension."

        return ArtifactResult(
            name="spectral_rolloff",
            detected=detected,
            severity=severity,
            value=round(median_rolloff, 0),
            description=desc,
            probability=probability,
            weight=1.0,
        )

    def _check_spectral_flux(self, y: np.ndarray, sr: int) -> ArtifactResult:
        """Measure spectral flux variance to detect static/smeared mixes.

        Professional mixes have high spectral flux variance due to clean
        instrument transitions and EQ carving. AI-generated audio tends
        to produce a "wall of sound" with lower flux variance.
        """
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))

        # Compute spectral flux: L2 norm of frame-to-frame differences
        diff = np.diff(S, axis=1)
        flux = np.sqrt(np.sum(diff ** 2, axis=0))

        if len(flux) < 10:
            return ArtifactResult(
                name="spectral_flux_variance",
                detected=False,
                severity="none",
                value=None,
                description="Audio too short for spectral flux analysis.",
                probability=0.0,
                weight=1.0,
            )

        # Normalize flux by mean to get coefficient of variation
        mean_flux = np.mean(flux)
        if mean_flux < 1e-10:
            return ArtifactResult(
                name="spectral_flux_variance",
                detected=False,
                severity="none",
                value=None,
                description="No significant spectral variation detected.",
                probability=0.0,
                weight=1.0,
            )

        flux_cv = float(np.std(flux) / mean_flux)

        # Map to probability
        if flux_cv < 0.3:
            probability = 0.8
            severity = "high"
        elif flux_cv < 0.5:
            probability = 0.5
            severity = "medium"
        elif flux_cv < 0.8:
            probability = 0.2
            severity = "low"
        else:
            probability = 0.0
            severity = "none"

        detected = flux_cv < 0.5

        return ArtifactResult(
            name="spectral_flux_variance",
            detected=detected,
            severity=severity,
            value=round(flux_cv, 3),
            description=(
                f"Spectral flux coefficient of variation: {flux_cv:.3f}. "
                + (
                    "Low spectral flux variance indicates static, smeared mix transitions "
                    "typical of AI-generated audio."
                    if detected
                    else "Healthy spectral variation consistent with professional mixing."
                )
            ),
            probability=probability,
            weight=1.0,
        )
