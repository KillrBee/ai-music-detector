"""Vocal domain analyzer for AI-generated audio detection.

Checks:
1. Formant stability (vocal tract consistency across pitch)
2. Breath logic (physiologically plausible breathing)
3. Phoneme clarity (sibilant/plosive sharpness)

Requires vocals to be present in the audio. If no vocals detected,
this domain returns inactive.
"""

import logging
from typing import Optional

import librosa
import numpy as np
from scipy import signal as scipy_signal

from .base import AnalysisDepth, ArtifactResult, BaseAnalyzer, DomainResult

logger = logging.getLogger(__name__)

# Optional parselmouth import
try:
    import parselmouth
    HAS_PARSELMOUTH = True
except ImportError:
    HAS_PARSELMOUTH = False
    logger.info("parselmouth not installed — formant stability check disabled. "
                "Install with: uv pip install praat-parselmouth")


class VocalAnalyzer(BaseAnalyzer):
    """Analyzes vocal synthesis artifacts for AI detection."""

    domain = "vocal"
    display_name = "Vocal Analysis"
    base_weight = 0.13
    min_depth = AnalysisDepth.DEEP

    def analyze(
        self,
        y_mono: np.ndarray,
        sr: int,
        y_stereo: Optional[np.ndarray] = None,
        depth: AnalysisDepth = AnalysisDepth.STANDARD,
    ) -> DomainResult:
        # First check if vocals are present
        if not self._has_vocals(y_mono, sr):
            return self._inactive_result("No vocals detected in audio.")

        artifacts = []

        if HAS_PARSELMOUTH:
            try:
                artifacts.append(self._check_formant_stability(y_mono, sr))
            except Exception as e:
                logger.warning("Formant stability check failed: %s", e)
        else:
            logger.info("Skipping formant stability (parselmouth not installed)")

        try:
            artifacts.append(self._check_breath_logic(y_mono, sr))
        except Exception as e:
            logger.warning("Breath logic check failed: %s", e)

        try:
            artifacts.append(self._check_phoneme_clarity(y_mono, sr))
        except Exception as e:
            logger.warning("Phoneme clarity check failed: %s", e)

        if not artifacts:
            return self._inactive_result("All vocal checks failed or were skipped.")

        return self._make_domain_result(artifacts)

    def _has_vocals(self, y: np.ndarray, sr: int) -> bool:
        """Heuristic check for vocal presence.

        Vocals are characterized by significant energy in the 300-3500Hz
        formant range with pitched content. Uses pyin pitch detection to
        verify actual voiced (pitched) content is present, not just
        instrumental energy in the vocal frequency range.
        """
        S = np.abs(librosa.stft(y, n_fft=4096, hop_length=1024))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

        # Energy in vocal formant range (300-3500Hz)
        vocal_mask = (freqs >= 300) & (freqs <= 3500)
        vocal_energy = np.mean(S[vocal_mask, :])

        # Total energy
        total_energy = np.mean(S)

        if total_energy < 1e-10:
            return False

        vocal_ratio = vocal_energy / total_energy

        # Also check for pitched content using zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)[0]
        # Voiced audio has lower ZCR than noise
        mean_zcr = float(np.mean(zcr))

        # Pitch detection: verify actual voiced (pitched) content
        # Use a short segment (first 30s) for efficiency
        y_segment = y[:min(len(y), int(30 * sr))]
        try:
            _f0, voiced_flag, _voiced_probs = librosa.pyin(
                y_segment, fmin=80, fmax=1000, sr=sr, frame_length=2048
            )
            voiced_ratio = float(np.sum(voiced_flag) / max(len(voiced_flag), 1))
        except Exception:
            voiced_ratio = 0.0

        # Vocals require: energy in vocal range, moderate ZCR, AND
        # significant pitched (voiced) content (>20% of frames)
        return vocal_ratio > 1.2 and mean_zcr < 0.12 and voiced_ratio > 0.20

    def _check_formant_stability(self, y: np.ndarray, sr: int) -> ArtifactResult:
        """Check formant consistency across pitch changes.

        Real singers have a fixed vocal tract size. Formants (F1, F2, F3)
        are relatively stable regardless of pitch. AI vocals often shift
        formants linearly with pitch, creating a "variable head size" effect.
        """
        snd = parselmouth.Sound(y, sampling_frequency=sr)

        # Extract pitch
        pitch = snd.to_pitch(time_step=0.01)
        pitch_values = pitch.selected_array["frequency"]

        # Extract formants
        formant = snd.to_formant_burg(time_step=0.01, max_number_of_formants=3)

        # Collect (pitch, F1, F2, F3) for voiced frames
        voiced_data = []
        for i in range(len(pitch_values)):
            f0 = pitch_values[i]
            if f0 == 0:  # Unvoiced
                continue
            t = pitch.get_time_from_frame_number(i + 1)
            try:
                f1 = formant.get_value_at_time(1, t)
                f2 = formant.get_value_at_time(2, t)
                f3 = formant.get_value_at_time(3, t)
                if all(not np.isnan(v) for v in [f1, f2, f3]):
                    voiced_data.append((f0, f1, f2, f3))
            except Exception:
                pass

        if len(voiced_data) < 20:
            return ArtifactResult(
                name="formant_stability",
                detected=False,
                severity="none",
                value=None,
                description="Insufficient voiced frames for formant stability analysis.",
                probability=0.0,
                weight=1.0,
            )

        data = np.array(voiced_data)
        pitches = data[:, 0]
        formants = data[:, 1:4]  # F1, F2, F3

        # Compute correlation between pitch changes and formant changes
        pitch_diffs = np.diff(pitches)
        formant_diffs = np.diff(formants, axis=0)

        correlations = []
        for i in range(3):  # F1, F2, F3
            if np.std(pitch_diffs) > 1e-6 and np.std(formant_diffs[:, i]) > 1e-6:
                corr = abs(float(np.corrcoef(pitch_diffs, formant_diffs[:, i])[0, 1]))
                if not np.isnan(corr):
                    correlations.append(corr)

        if not correlations:
            return ArtifactResult(
                name="formant_stability",
                detected=False,
                severity="none",
                value=None,
                description="Could not compute pitch-formant correlation.",
                probability=0.0,
                weight=1.0,
            )

        max_corr = max(correlations)

        # Map to probability
        if max_corr > 0.5:
            probability = 0.8
            severity = "high"
        elif max_corr > 0.3:
            probability = 0.5
            severity = "medium"
        elif max_corr > 0.1:
            probability = 0.2
            severity = "low"
        else:
            probability = 0.0
            severity = "none"

        detected = max_corr > 0.3

        return ArtifactResult(
            name="formant_stability",
            detected=detected,
            severity=severity,
            value=round(max_corr, 3),
            description=(
                f"Pitch-formant correlation: {max_corr:.3f}. "
                + (
                    "Formants shift with pitch, suggesting AI vocal synthesis "
                    "with 'variable head size' artifacts."
                    if detected
                    else "Stable formants across pitch changes, consistent with real singing."
                )
            ),
            probability=probability,
            weight=1.0,
        )

    def _check_breath_logic(self, y: np.ndarray, sr: int) -> ArtifactResult:
        """Check for physiologically impossible breathing patterns.

        Flags:
        - Phrases >15 seconds without a breath (impossible for real singing)
        - Breaths during sustained vocal activity (impossible)
        - Double breaths (<200ms apart)
        """
        # Compute energy envelope
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

        # Thresholds
        mean_rms = float(np.mean(rms))
        vocal_threshold = mean_rms * 0.5
        breath_threshold = mean_rms * 0.15

        # Identify vocal (active) and breath (quiet) regions
        is_vocal = rms > vocal_threshold
        is_quiet = rms < breath_threshold

        # Find vocal phrases (continuous active regions)
        anomalies = 0
        phrase_duration = 0.0
        in_phrase = False
        last_breath_time = -10.0

        frame_duration = float(times[1] - times[0]) if len(times) > 1 else hop_length / sr

        for i in range(len(rms)):
            t = float(times[i])

            if is_vocal[i]:
                if not in_phrase:
                    in_phrase = True
                    phrase_duration = 0.0
                phrase_duration += frame_duration

                # Check for infinite phrase (>30 seconds without breath)
                # Professional singers can sustain 15-20s phrases with technique
                if phrase_duration > 30.0:
                    anomalies += 1
                    phrase_duration = 0.0  # Reset to avoid counting repeatedly

            elif is_quiet[i]:
                if in_phrase:
                    in_phrase = False

                    # Check for double breath
                    if t - last_breath_time < 0.2 and last_breath_time > 0:
                        anomalies += 1

                    last_breath_time = t
                    phrase_duration = 0.0

        # Normalize anomaly count by duration (anomalies per minute)
        duration_minutes = len(rms) * frame_duration / 60.0
        anomaly_rate = anomalies / max(duration_minutes, 0.5)

        # Data-driven dual detection from 22-track analysis:
        # AI tracks at 0.3, 0.8/min are below ALL human vocal tracks (min 1.2/min).
        # Some AI tracks have extreme excess (11.4/min).
        # Normal human range is ~1.0-3.0/min.
        if anomaly_rate < 0.5:
            # Almost no breathing: strong AI indicator
            probability = 0.6
            severity = "medium"
        elif anomaly_rate < 1.0:
            # Very few breath events: moderate AI indicator
            probability = 0.3
            severity = "low"
        elif anomaly_rate > 5.0:
            # Impossibly many anomalies: also suspicious
            probability = 0.6
            severity = "medium"
        elif anomaly_rate > 3.0:
            # High anomalies: mildly suspicious
            probability = 0.3
            severity = "low"
        else:
            # Normal human range (1.0-3.0/min)
            probability = 0.0
            severity = "none"

        detected = anomaly_rate < 1.0 or anomaly_rate > 3.0

        return ArtifactResult(
            name="breath_logic",
            detected=detected,
            severity=severity,
            value=round(anomaly_rate, 2),
            description=(
                f"Breathing anomalies: {anomalies} ({anomaly_rate:.1f}/min). "
                + (
                    "Absence of natural breathing patterns suggests AI vocal synthesis."
                    if anomaly_rate < 1.0
                    else (
                        "Physiologically impossible breathing patterns suggest AI vocal synthesis."
                        if anomaly_rate > 3.0
                        else "Breathing patterns consistent with real vocal performance."
                    )
                )
            ),
            probability=probability,
            weight=1.0,
        )

    def _check_phoneme_clarity(self, y: np.ndarray, sr: int) -> ArtifactResult:
        """Analyze sibilant and plosive clarity.

        AI vocals often have smeared consonants due to neural codec HF issues.
        Real sibilants (S, Sh) have sharp, varied HF energy (5-10kHz).
        Real plosives (P, T, K) have brief gaps and sharp transients.
        """
        # Detect sibilant-like events: short HF energy bursts
        nyquist = sr // 2
        if nyquist < 5000:
            return ArtifactResult(
                name="phoneme_clarity",
                detected=False,
                severity="none",
                value=None,
                description="Sample rate too low for phoneme clarity analysis.",
                probability=0.0,
                weight=1.0,
            )

        # Bandpass 5-10kHz for sibilants
        high_limit = min(10000, nyquist - 100)
        if high_limit <= 5000:
            return ArtifactResult(
                name="phoneme_clarity",
                detected=False,
                severity="none",
                value=None,
                description="Insufficient bandwidth for sibilant analysis.",
                probability=0.0,
                weight=1.0,
            )

        sos = scipy_signal.butter(4, [5000, high_limit], btype="band", fs=sr, output="sos")
        y_sib = scipy_signal.sosfilt(sos, y)

        # Compute energy envelope of sibilant band
        hop_length = 256
        rms_sib = librosa.feature.rms(y=y_sib, frame_length=1024, hop_length=hop_length)[0]

        # Detect sibilant events (peaks above threshold)
        sib_threshold = float(np.mean(rms_sib)) * 2.0
        sib_events = []

        i = 0
        while i < len(rms_sib):
            if rms_sib[i] > sib_threshold:
                start = i
                while i < len(rms_sib) and rms_sib[i] > sib_threshold * 0.5:
                    i += 1
                duration_ms = (i - start) * hop_length / sr * 1000
                if 30 < duration_ms < 200:  # Sibilant-like duration
                    # Get peak frequency of this event
                    seg_start = start * hop_length
                    seg_end = min(i * hop_length, len(y))
                    if seg_end - seg_start > 256:
                        seg_fft = np.abs(np.fft.rfft(y[seg_start:seg_end]))
                        seg_freqs = np.fft.rfftfreq(seg_end - seg_start, 1.0 / sr)
                        hf_mask = seg_freqs >= 5000
                        if np.any(hf_mask) and np.max(seg_fft[hf_mask]) > 0:
                            peak_freq = seg_freqs[hf_mask][np.argmax(seg_fft[hf_mask])]
                            sib_events.append({"peak_freq": peak_freq, "duration_ms": duration_ms})
            i += 1

        if len(sib_events) < 3:
            return ArtifactResult(
                name="phoneme_clarity",
                detected=False,
                severity="none",
                value=None,
                description="Insufficient sibilant events for phoneme clarity analysis.",
                probability=0.0,
                weight=1.0,
            )

        # Analyze sibilant frequency variation
        peak_freqs = np.array([e["peak_freq"] for e in sib_events])
        freq_std = float(np.std(peak_freqs))

        # Map to probability
        probabilities = []

        # Uniform sibilant frequency (AI tends to produce identical sibilants)
        if freq_std < 200:
            probabilities.append(0.6)
        elif freq_std < 400:
            probabilities.append(0.3)

        # Check sibilant-to-surrounding energy ratio
        sib_energy = float(np.mean(rms_sib[rms_sib > sib_threshold]))
        surrounding_energy = float(np.mean(rms_sib[rms_sib <= sib_threshold]))
        if surrounding_energy > 1e-10:
            ratio_db = 20 * np.log10(sib_energy / surrounding_energy + 1e-10)
            if ratio_db < 6:  # Weak sibilants
                probabilities.append(0.5)
        else:
            probabilities.append(0.3)

        probability = max(probabilities) if probabilities else 0.0

        if probability >= 0.5:
            severity = "medium"
        elif probability >= 0.3:
            severity = "low"
        else:
            severity = "none"

        detected = probability >= 0.3

        return ArtifactResult(
            name="phoneme_clarity",
            detected=detected,
            severity=severity,
            value=round(freq_std, 0),
            description=(
                f"Sibilant frequency std: {freq_std:.0f} Hz "
                f"({len(sib_events)} events detected). "
                + (
                    "Uniform or weak consonant articulation suggests "
                    "neural codec smearing of high-frequency phonemes."
                    if detected
                    else "Natural variation in consonant articulation."
                )
            ),
            probability=probability,
            weight=1.0,
        )
