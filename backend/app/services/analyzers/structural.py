"""Structural domain analyzer for AI-generated audio detection.

Checks:
1. Self-similarity matrix (repetition structure)
2. Structural entropy (harmonic predictability)
"""

import logging
from typing import Optional

import librosa
import numpy as np
from scipy.stats import entropy as shannon_entropy

from .base import AnalysisDepth, ArtifactResult, BaseAnalyzer, DomainResult

logger = logging.getLogger(__name__)


class StructuralAnalyzer(BaseAnalyzer):
    """Analyzes musical structure and semantic coherence for AI detection."""

    domain = "structural"
    display_name = "Structural Analysis"
    base_weight = 0.15
    min_depth = AnalysisDepth.DEEP

    def analyze(
        self,
        y_mono: np.ndarray,
        sr: int,
        y_stereo: Optional[np.ndarray] = None,
        depth: AnalysisDepth = AnalysisDepth.STANDARD,
    ) -> DomainResult:
        artifacts = []

        try:
            artifacts.append(self._check_self_similarity(y_mono, sr))
        except Exception as e:
            logger.warning("Self-similarity check failed: %s", e)

        try:
            artifacts.append(self._check_structural_entropy(y_mono, sr))
        except Exception as e:
            logger.warning("Structural entropy check failed: %s", e)

        return self._make_domain_result(artifacts)

    def _check_self_similarity(self, y: np.ndarray, sr: int) -> ArtifactResult:
        """Analyze self-similarity matrix for structural repetition.

        Real songs have clear repetition (verse-chorus structure visible as
        off-diagonal stripes in SSM). AI may produce:
        - No repetition (hallucinated choruses)
        - Perfect rigid repetition (repetition trap / looping)
        """
        # Limit analysis to first 5 minutes to keep computation tractable
        max_samples = int(5 * 60 * sr)
        y_trimmed = y[:max_samples]

        # Compute chroma features with large hop for efficiency
        hop_length = 4096
        chroma = librosa.feature.chroma_cqt(
            y=y_trimmed, sr=sr, hop_length=hop_length
        )

        n_frames = chroma.shape[1]
        if n_frames < 20:
            return ArtifactResult(
                name="self_similarity_matrix",
                detected=False,
                severity="none",
                value=None,
                description="Audio too short for structural similarity analysis.",
                probability=0.0,
                weight=1.0,
            )

        # Normalize chroma columns
        norms = np.linalg.norm(chroma, axis=0, keepdims=True)
        norms[norms < 1e-10] = 1.0
        chroma_norm = chroma / norms

        # Compute self-similarity matrix (cosine similarity)
        ssm = chroma_norm.T @ chroma_norm

        # Analyze off-diagonal similarity
        # Look for structural repetition at expected intervals
        # Estimate bar length from tempo
        tempo, _ = librosa.beat.beat_track(y=y_trimmed, sr=sr, hop_length=hop_length)
        if hasattr(tempo, '__len__'):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            tempo = float(tempo)
        if tempo <= 0:
            tempo = 120.0

        beats_per_bar = 4
        seconds_per_bar = 60.0 / tempo * beats_per_bar
        frames_per_bar = max(1, int(seconds_per_bar * sr / hop_length))

        # Check similarity at multiples of 4, 8, 16 bars (typical section lengths)
        off_diag_similarities = []
        for multiplier in [4, 8, 16, 32]:
            offset = frames_per_bar * multiplier
            if offset >= n_frames:
                continue

            # Extract diagonal stripe at this offset
            diag_length = n_frames - offset
            diag_values = np.array([
                ssm[i, i + offset] for i in range(diag_length)
            ])
            off_diag_similarities.append(float(np.mean(diag_values)))

        if not off_diag_similarities:
            return ArtifactResult(
                name="self_similarity_matrix",
                detected=False,
                severity="none",
                value=None,
                description="Could not compute structural similarity at expected intervals.",
                probability=0.0,
                weight=1.0,
            )

        max_off_diag = max(off_diag_similarities)

        # Check for repetition trap (perfect grid)
        # Autocorrelation of the first row of SSM
        first_row = ssm[0, :]
        acf = np.correlate(first_row - np.mean(first_row), first_row - np.mean(first_row), mode="full")
        acf = acf[len(acf) // 2:]
        if acf[0] > 0:
            acf = acf / acf[0]

        # Find peaks in ACF (indicating rigid repetition)
        rigid_repetition = False
        if len(acf) > frames_per_bar * 2:
            acf_peaks = acf[frames_per_bar:frames_per_bar * 8]
            if len(acf_peaks) > 0 and np.max(acf_peaks) > 0.95:
                rigid_repetition = True

        # Map to probability
        if max_off_diag < 0.3:
            # No structural repetition — hallucinated sections
            probability = 0.7
            severity = "high"
        elif max_off_diag < 0.5:
            probability = 0.4
            severity = "medium"
        elif rigid_repetition:
            # Perfect repetition trap
            probability = 0.6
            severity = "medium"
        else:
            probability = 0.0
            severity = "none"

        detected = probability >= 0.3

        desc = f"Max off-diagonal similarity: {max_off_diag:.2f}"
        if rigid_repetition:
            desc += " (rigid repetition pattern detected)"
        if max_off_diag < 0.3:
            desc += ". No structural repetition found — suggests AI-hallucinated sections."
        elif rigid_repetition:
            desc += ". Perfect looping pattern suggests AI repetition trap."
        else:
            desc += ". Healthy structural repetition consistent with real song form."

        return ArtifactResult(
            name="self_similarity_matrix",
            detected=detected,
            severity=severity,
            value=round(max_off_diag, 3),
            description=desc,
            probability=probability,
            weight=1.0,
        )

    def _check_structural_entropy(self, y: np.ndarray, sr: int) -> ArtifactResult:
        """Analyze harmonic predictability and section-level dynamics.

        Real music has structured entropy changes (quiet verses, loud choruses).
        AI tends to have either monotonous entropy or erratic, meaningless changes.
        Also checks for non-resolving cadences (harmonic tension without resolution).
        """
        # Limit to 5 minutes
        max_samples = int(5 * 60 * sr)
        y_trimmed = y[:max_samples]

        hop_length = 2048

        # Get tempo for bar estimation
        tempo, _ = librosa.beat.beat_track(y=y_trimmed, sr=sr, hop_length=hop_length)
        if hasattr(tempo, '__len__'):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            tempo = float(tempo)
        if tempo <= 0:
            tempo = 120.0

        # Compute chroma
        chroma = librosa.feature.chroma_cqt(
            y=y_trimmed, sr=sr, hop_length=hop_length
        )

        # Estimate bar length in frames
        seconds_per_bar = 60.0 / tempo * 4
        frames_per_bar = max(1, int(seconds_per_bar * sr / hop_length))

        # Compute entropy per bar
        n_bars = chroma.shape[1] // frames_per_bar
        if n_bars < 4:
            return ArtifactResult(
                name="structural_entropy",
                detected=False,
                severity="none",
                value=None,
                description="Audio too short for structural entropy analysis.",
                probability=0.0,
                weight=1.0,
            )

        bar_entropies = []
        bar_dominant_chroma = []

        for i in range(n_bars):
            start = i * frames_per_bar
            end = start + frames_per_bar
            bar_chroma = np.mean(chroma[:, start:end], axis=1)

            # Normalize to probability distribution
            bar_sum = np.sum(bar_chroma)
            if bar_sum > 1e-10:
                bar_dist = bar_chroma / bar_sum
            else:
                bar_dist = np.ones(12) / 12.0

            bar_entropies.append(shannon_entropy(bar_dist))
            bar_dominant_chroma.append(int(np.argmax(bar_dist)))

        bar_entropies = np.array(bar_entropies)
        mean_entropy = float(np.mean(bar_entropies))

        # Use coefficient of variation (CV) instead of raw variance
        # for better normalization across genres
        if mean_entropy > 1e-6:
            entropy_cv = float(np.std(bar_entropies) / mean_entropy)
        else:
            entropy_cv = 0.0

        # Check for non-resolving cadences
        # At section boundaries (every 4-8 bars), check if dominant pitch
        # resolves to a tonic-like relationship
        non_resolving = 0
        section_length = 8  # bars per section
        n_sections = n_bars // section_length
        for s in range(n_sections - 1):
            end_bar = (s + 1) * section_length - 1
            next_bar = (s + 1) * section_length
            if end_bar < len(bar_dominant_chroma) and next_bar < len(bar_dominant_chroma):
                end_pitch = bar_dominant_chroma[end_bar]
                next_pitch = bar_dominant_chroma[next_bar]
                # Common resolutions: perfect fifth (7 semitones), perfect fourth (5),
                # or same pitch (0). Non-resolution: other intervals
                interval = (next_pitch - end_pitch) % 12
                if interval not in {0, 5, 7}:
                    non_resolving += 1

        # Data-driven compound condition from 22-track analysis:
        # Low entropy CV alone is genre-dependent (pop/electronic naturally low).
        # Require BOTH low CV AND high non-resolving cadence ratio together.
        cadence_ratio = 0.0
        if n_sections > 1:
            cadence_ratio = non_resolving / max(n_sections - 1, 1)

        if entropy_cv < 0.02 and cadence_ratio > 0.40:
            probability = 0.7
            severity = "high"
        elif entropy_cv < 0.03 and cadence_ratio > 0.30:
            probability = 0.5
            severity = "medium"
        elif entropy_cv < 0.04 and cadence_ratio > 0.40:
            probability = 0.3
            severity = "low"
        elif cadence_ratio > 0.55:
            # Very high non-resolving cadences alone
            probability = 0.25
            severity = "low"
        else:
            probability = 0.0
            severity = "none"

        detected = probability >= 0.3

        desc = f"Entropy CV: {entropy_cv:.4f}"
        if n_sections > 1:
            desc += f", non-resolving cadences: {non_resolving}/{max(n_sections - 1, 1)}"
        if detected:
            desc += ". Low harmonic variation or weak tonal resolution suggests AI-generated structure."
        else:
            desc += ". Healthy harmonic dynamics consistent with composed music."

        return ArtifactResult(
            name="structural_entropy",
            detected=detected,
            severity=severity,
            value=round(entropy_cv, 4),
            description=desc,
            probability=probability,
            weight=1.0,  # Overridden by centralized weights.py
        )
