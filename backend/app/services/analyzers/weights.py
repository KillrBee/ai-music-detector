"""Centralized artifact and domain weight configuration.

All detection weights are defined here rather than scattered across
individual analyzer files. This makes the weighting system transparent,
auditable, and easy to tune.

Artifact Tier System:
    Tier 1 (Definitive): Almost exclusively appear in AI audio.
    Tier 2 (Strong): Strongly suggest AI, rare in legitimate audio.
    Tier 3 (Moderate): Useful supporting evidence.
    Tier 4 (Weak/Contextual): Commonly triggered by legitimate audio.
"""

from dataclasses import dataclass
from enum import IntEnum


class ArtifactTier(IntEnum):
    """Reliability tier for an artifact as an AI indicator."""

    DEFINITIVE = 1  # Near-certain AI indicator when detected
    STRONG = 2  # Strongly suggests AI
    MODERATE = 3  # Useful supporting evidence
    WEAK = 4  # Commonly triggered by legitimate audio


TIER_LABELS: dict[int, str] = {
    1: "Definitive",
    2: "Strong",
    3: "Moderate",
    4: "Weak",
}


@dataclass(frozen=True)
class ArtifactWeightConfig:
    """Weight configuration for a single artifact."""

    weight: float  # Relative importance within its domain (>0)
    tier: ArtifactTier  # Reliability classification
    notes: str = ""  # Human-readable explanation


@dataclass(frozen=True)
class DomainWeightConfig:
    """Weight configuration for a domain."""

    base_weight: float  # Domain's share of overall score (all sum to 1.0)
    display_name: str


# ---- Domain Weights --------------------------------------------------------
# Must sum to 1.0. These define how much each domain contributes
# to the overall AI probability score.

DOMAIN_WEIGHTS: dict[str, DomainWeightConfig] = {
    "spectral": DomainWeightConfig(
        base_weight=0.4, display_name="Spectral Analysis"
    ),
    "temporal": DomainWeightConfig(
        base_weight=0.1, display_name="Temporal Analysis"
    ),
    "spatial": DomainWeightConfig(
        base_weight=0.05, display_name="Spatial Analysis"
    ),
    "structural": DomainWeightConfig(
        base_weight=0.2, display_name="Structural Analysis"
    ),
    "production": DomainWeightConfig(
        base_weight=0.05, display_name="Production Analysis"
    ),
    "vocal": DomainWeightConfig(
        base_weight=0.2, display_name="Vocal Analysis"
    ),
    #"watermark": DomainWeightConfig(
    #    base_weight=0.2, display_name="Watermark Detection"
    #),
}


# ---- Artifact Weights -------------------------------------------------------
# Keys are artifact names as they appear in ArtifactResult.name.
# Weight is relative importance *within the artifact's domain*.
# Tier classifies the artifact's reliability as an AI indicator.

ARTIFACT_WEIGHTS: dict[str, ArtifactWeightConfig] = {
    # ---- Tier 1: Definitive ----
    "audioseal_watermark": ArtifactWeightConfig(
        weight=10,
        tier=ArtifactTier.DEFINITIVE,
        notes="AI watermark is near-definitive proof of AI generation.",
    ),
    "loudness_range": ArtifactWeightConfig(
        weight=4.5,
        tier=ArtifactTier.DEFINITIVE,
        notes="Best empirical discriminator: 8/12 AI tracks < 5 LU vs 8/10 humans ≥ 5 LU.",
    ),
    # ---- Tier 2: Strong ----
    "reverb_tail_analysis": ArtifactWeightConfig(
        weight=2.0,
        tier=ArtifactTier.STRONG,
        notes="2nd best discriminator: AI clusters at R²=0.18–0.50.",
    ),
    "hf_harmonic_noise_ratio": ArtifactWeightConfig(
        weight=3.5,
        tier=ArtifactTier.STRONG,
        notes="AI clusters at -0 to -7 dB HF HNR; humans mostly below -7 dB.",
    ),
    "breath_logic": ArtifactWeightConfig(
        weight=1.5,
        tier=ArtifactTier.STRONG,
        notes="Absence of breathing (<0.5/min) or extreme excess (>5/min) indicates AI.",
    ),
    # ---- Tier 3: Moderate ----
    "checkerboard_artifacts": ArtifactWeightConfig(
        weight=5.0,
        tier=ArtifactTier.MODERATE,
        notes="Demoted: 5/10 human tracks false positive at 0.60. Ratio overlap too high.",
    ),
    "spectral_rolloff": ArtifactWeightConfig(
        weight=1.0,
        tier=ArtifactTier.MODERATE,
        notes="Brick-wall rolloff is theoretically sound but rarely triggers in practice.",
    ),
    "structural_entropy": ArtifactWeightConfig(
        weight=1.5,
        tier=ArtifactTier.MODERATE,
        notes="Genre-dependent: pop/electronic naturally has low entropy CV.",
    ),
    "spectral_flux_variance": ArtifactWeightConfig(
        weight=1.0,
        tier=ArtifactTier.MODERATE,
        notes="Low flux variance is common in AI but also in some genres.",
    ),
    # ---- Tier 4: Weak / Contextual ----
    "formant_stability": ArtifactWeightConfig(
        weight=0.5,
        tier=ArtifactTier.WEAK,
        notes="Demoted: never fires on AI tracks in 22-track dataset.",
    ),
    "phase_correlation": ArtifactWeightConfig(
        weight=0.5,
        tier=ArtifactTier.WEAK,
        notes="Demoted: only 1/22 tracks triggered (1 AI at 0.30).",
    ),
    "self_similarity_matrix": ArtifactWeightConfig(
        weight=0.5,
        tier=ArtifactTier.WEAK,
        notes="Demoted: never fires on any track.",
    ),
    "crest_factor": ArtifactWeightConfig(
        weight=0.3,
        tier=ArtifactTier.WEAK,
        notes="Demoted: never fires on any track.",
    ),
    "log_attack_time": ArtifactWeightConfig(
        weight=0.3,
        tier=ArtifactTier.WEAK,
        notes="Demoted: no discrimination; all tracks cluster at same value.",
    ),
    "bass_stereo_width": ArtifactWeightConfig(
        weight=0.3,
        tier=ArtifactTier.WEAK,
        notes="Inversely correlated: human mean 0.21 > AI mean 0.08.",
    ),
    "phoneme_clarity": ArtifactWeightConfig(
        weight=0.3,
        tier=ArtifactTier.WEAK,
        notes="Demoted: never fires on any track.",
    ),
    "rhythmic_jitter": ArtifactWeightConfig(
        weight=0.05,
        tier=ArtifactTier.WEAK,
        notes="Near-disabled: pure noise, human mean 0.62 ≈ AI mean 0.63.",
    ),
}


def get_artifact_weight(artifact_name: str) -> float:
    """Get the weight for an artifact, defaulting to 1.0 if not configured."""
    config = ARTIFACT_WEIGHTS.get(artifact_name)
    return config.weight if config else 1.0


def get_artifact_tier(artifact_name: str) -> int:
    """Get the tier for an artifact, defaulting to MODERATE (3) if unknown."""
    config = ARTIFACT_WEIGHTS.get(artifact_name)
    return int(config.tier) if config else int(ArtifactTier.MODERATE)


def get_domain_base_weight(domain_name: str) -> float:
    """Get the base weight for a domain, defaulting to 0.0 if not configured."""
    config = DOMAIN_WEIGHTS.get(domain_name)
    return config.base_weight if config else 0.0
