"""Weighted probabilistic scoring engine for AI detection.

Uses evidence gating to suppress uninformative domains and a concordance
boost when multiple independent domains agree on AI detection.
"""

from dataclasses import dataclass, field

from .base import ArtifactResult, DomainResult

# --- Scoring constants ---
EVIDENCE_THRESHOLD = 0.10       # Below this, domain weight is linearly ramped down
CONCORDANCE_THRESHOLD = 0.35    # Above this, domain counts as concordant evidence
CONCORDANCE_BOOST_FACTOR = 0.25 # Multiplied with concordant signal for bonus


@dataclass
class ScoringResult:
    """Final aggregated scoring result across all domains."""
    overall_score: float             # 0-100 percentage
    confidence: str                  # "low", "medium", "high"
    confidence_value: float          # 0.0-1.0
    overall_likelihood: str          # "unlikely", "possible", "likely"
    domain_results: list[DomainResult] = field(default_factory=list)
    artifacts_flat: list[ArtifactResult] = field(default_factory=list)


class WeightedScoringEngine:
    """Computes overall AI probability from domain-level results.

    Domain base weights (sum to 1.0):
        spectral=0.25, temporal=0.12, spatial=0.10, structural=0.15,
        production=0.15, vocal=0.13, watermark=0.10

    When a domain is inactive (not applicable or skipped by depth),
    its weight is redistributed proportionally among active domains.

    Two-step scoring:
    1. Evidence-gated weighted mean — domains with near-zero scores get
       their weight scaled down so they don't dilute strong signals.
    2. Concordance boost — when ≥2 domains independently flag AI above
       the concordance threshold, a bonus is added proportional to the
       strength and breadth of agreement.
    """

    def calculate(self, domain_results: list[DomainResult]) -> ScoringResult:
        """Calculate the overall AI detection score.

        Args:
            domain_results: List of DomainResult from each analyzer.

        Returns:
            ScoringResult with overall score, confidence, and breakdown.
        """
        active = [d for d in domain_results if d.active]

        if not active:
            return ScoringResult(
                overall_score=0.0,
                confidence="low",
                confidence_value=0.0,
                overall_likelihood="unknown",
                domain_results=domain_results,
                artifacts_flat=[],
            )

        # --- Step 1: Evidence-gated weighted mean ---
        # Domains scoring below EVIDENCE_THRESHOLD get their weight
        # linearly scaled down so near-zero domains don't consume budget.
        effective_weights = []
        for d in active:
            gate = min(d.score / EVIDENCE_THRESHOLD, 1.0) if EVIDENCE_THRESHOLD > 0 else 1.0
            effective_weights.append(d.weight * gate)

        total_effective = sum(effective_weights)

        if total_effective > 0:
            base_score = sum(
                d.score * ew for d, ew in zip(active, effective_weights)
            ) / total_effective
        else:
            # Fallback: simple mean if all weights gated to zero
            base_score = sum(d.score for d in active) / len(active)

        # --- Step 2: Concordance boost ---
        # When ≥2 domains independently score above the concordance
        # threshold, multiple lines of evidence agree → boost the score.
        concordant = [d for d in active if d.score > CONCORDANCE_THRESHOLD]

        if len(concordant) >= 2:
            concordant_ratio = len(concordant) / len(active)
            mean_concordant = sum(d.score for d in concordant) / len(concordant)
            bonus = concordant_ratio * mean_concordant * CONCORDANCE_BOOST_FACTOR
            final_score = min(base_score + bonus, 1.0)
        else:
            final_score = base_score

        overall_score = round(final_score * 100, 1)

        # Confidence based on how many domains contributed
        # 7 is the total number of possible domains
        total_possible = 7
        active_ratio = len(active) / total_possible
        confidence_value = min(1.0, active_ratio * 1.2)

        if confidence_value >= 0.7:
            confidence = "high"
        elif confidence_value >= 0.4:
            confidence = "medium"
        else:
            confidence = "low"

        # Backward-compatible likelihood
        if overall_score >= 65:
            likelihood = "likely"
        elif overall_score >= 35:
            likelihood = "possible"
        else:
            likelihood = "unlikely"

        # Watermark override: definitive proof trumps everything
        for d in active:
            if d.domain == "watermark" and d.score > 0.9:
                overall_score = max(overall_score, 95.0)
                likelihood = "likely"
                confidence = "high"
                confidence_value = max(confidence_value, 0.95)

        # Flatten all artifacts for backward compatibility
        all_artifacts = []
        for d in domain_results:
            all_artifacts.extend(d.artifacts)

        return ScoringResult(
            overall_score=overall_score,
            confidence=confidence,
            confidence_value=round(confidence_value, 2),
            overall_likelihood=likelihood,
            domain_results=domain_results,
            artifacts_flat=all_artifacts,
        )
