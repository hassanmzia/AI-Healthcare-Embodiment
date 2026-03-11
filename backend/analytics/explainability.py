"""
Explainability module for AI-Healthcare-Embodiment.

Provides SHAP-style feature attribution for MS risk assessments,
model card generation, and counterfactual analysis.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("embodiment.analytics.explainability")


# Feature weights used in phenotyping agent
PHENOTYPING_WEIGHTS = {
    "age": 0.10,
    "sex": 0.05,
    "family_history_ms": 0.15,
    "vitamin_d_deficiency": 0.10,
    "prior_optic_neuritis": 0.20,
    "mri_lesion_count": 0.15,
    "fatigue_score": 0.08,
    "numbness_tingling": 0.07,
    "geographic_latitude": 0.05,
    "smoking_status": 0.05,
}

# Population baselines for feature z-scoring
BASELINES = {
    "age": {"mean": 40.0, "std": 12.0},
    "vitamin_d_level": {"mean": 30.0, "std": 10.0},
    "mri_lesion_count": {"mean": 0.5, "std": 1.5},
    "fatigue_score": {"mean": 3.0, "std": 2.0},
}


def compute_feature_attribution(
    patient_data: Dict[str, Any],
    risk_score: float,
    feature_contributions: Dict[str, float],
) -> Dict[str, Any]:
    """
    Compute SHAP-style feature attribution for a patient's risk assessment.

    Args:
        patient_data: Raw patient features
        risk_score: Final risk score (0-1)
        feature_contributions: Per-feature contribution from phenotyping agent

    Returns:
        Attribution result with contributions, counterfactuals, and explanation
    """
    contributions = []

    for feature, contribution in sorted(
        feature_contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    ):
        baseline = BASELINES.get(feature, {})
        patient_value = patient_data.get(feature)
        direction = "negative" if contribution > 0 else "positive"

        entry = {
            "feature": feature,
            "value": patient_value,
            "contribution": round(contribution, 4),
            "direction": direction,
            "weight": PHENOTYPING_WEIGHTS.get(feature, 0.0),
        }

        if baseline.get("mean") is not None and patient_value is not None:
            try:
                z = (float(patient_value) - baseline["mean"]) / baseline["std"]
                entry["z_score"] = round(z, 2)
                entry["percentile"] = round(_z_to_percentile(z), 1)
            except (TypeError, ZeroDivisionError):
                pass

        contributions.append(entry)

    top_drivers = [c["feature"] for c in contributions[:3]]

    # Counterfactual analysis
    counterfactuals = _compute_counterfactuals(
        feature_contributions, risk_score, PHENOTYPING_WEIGHTS
    )

    # Natural language explanation
    explanation = _generate_explanation(risk_score, contributions[:5], counterfactuals)

    return {
        "risk_score": round(risk_score, 4),
        "risk_level": _risk_level(risk_score),
        "contributions": contributions,
        "top_drivers": top_drivers,
        "counterfactual_analysis": counterfactuals,
        "natural_language_explanation": explanation,
    }


def generate_model_card(
    agent_name: str,
    metrics: Dict[str, float] = None,
    fairness: List[Dict] = None,
) -> Dict[str, Any]:
    """
    Generate a model card for an Embodiment agent.

    Returns a structured model card dict following Google's Model Cards format.
    """
    cards = {
        "phenotyping_agent": {
            "identity": {
                "agent_name": "phenotyping_agent",
                "version": "1.0.0",
                "agent_tier": "risk_assessment",
            },
            "overview": {
                "description": (
                    "Computes weighted MS risk scores (0.0-1.0) using 10 clinical features. "
                    "Assigns autonomy levels based on configurable policy thresholds."
                ),
                "intended_use": (
                    "Population-level MS risk screening for patients with neurological symptoms "
                    "or high-risk demographics. Not a diagnostic tool."
                ),
                "out_of_scope_uses": [
                    "Definitive MS diagnosis (requires McDonald Criteria evaluation)",
                    "Treatment selection or medication dosing",
                    "Pediatric patients (model trained on adults 18+)",
                ],
            },
            "technical": {
                "model_type": "weighted_ensemble",
                "features": list(PHENOTYPING_WEIGHTS.keys()),
                "feature_weights": PHENOTYPING_WEIGHTS,
            },
            "safety": {
                "considerations": [
                    "Risk score is screening-only; false negatives may delay diagnosis",
                    "Geographic latitude feature may introduce demographic bias",
                    "Autonomy levels are policy-dependent — changes affect auto-action rates",
                ],
                "known_limitations": [
                    "Feature weights are static, not learned from data",
                    "No temporal modeling — scores are point-in-time",
                    "Limited to 10 features; does not incorporate genetic markers",
                ],
                "hitl_requirements": (
                    "DRAFT_ORDER and RECOMMEND actions require clinician review. "
                    "AUTO_ORDER executes only with guardrail approval."
                ),
            },
            "clinical": {
                "evidence_level": "B",
                "source_guidelines": [
                    "McDonald Criteria 2017 (MS Diagnosis)",
                    "AAN Practice Guidelines for MS",
                ],
            },
        },
        "safety_agent": {
            "identity": {
                "agent_name": "safety_agent",
                "version": "1.0.0",
                "agent_tier": "governance",
            },
            "overview": {
                "description": (
                    "Validates agent outputs against safety rules, checks for contraindications, "
                    "verifies appropriate autonomy levels, and flags potential errors."
                ),
                "intended_use": "Safety checkpoint for all agent outputs before clinical action execution.",
            },
            "safety": {
                "considerations": [
                    "Must NEVER suppress safety flags",
                    "All flagged assessments require physician review",
                    "Flag count directly affects autonomy level assignment",
                ],
            },
        },
        "coordinator_agent": {
            "identity": {
                "agent_name": "coordinator_agent",
                "version": "1.0.0",
                "agent_tier": "orchestration",
            },
            "overview": {
                "description": (
                    "Orchestrates the multi-agent pipeline: patient intake → phenotyping → "
                    "safety check → action determination. Manages state and error recovery."
                ),
                "intended_use": "Pipeline orchestration for batch and single-patient risk assessment workflows.",
            },
        },
    }

    card = cards.get(agent_name, {
        "identity": {"agent_name": agent_name, "version": "1.0.0"},
        "overview": {"description": f"No model card registered for {agent_name}"},
    })

    # Inject runtime metrics if provided
    if metrics:
        card["performance"] = metrics
    if fairness:
        card["fairness"] = fairness

    return card


def compute_disparity_analysis(
    subgroup_metrics: List[Dict],
) -> Dict[str, Any]:
    """
    Compute disparity ratios and fairness assessment from subgroup metrics.

    Args:
        subgroup_metrics: Output from analytics.services.subgroup_analysis()

    Returns:
        Fairness assessment with disparity ratios and pass/flag/fail determination
    """
    if not subgroup_metrics or len(subgroup_metrics) < 2:
        return {"assessment": "insufficient_data", "subgroups": subgroup_metrics or []}

    flagged_rates = [s.get("flagged_rate", 0) for s in subgroup_metrics if s.get("n", 0) >= 30]
    if len(flagged_rates) < 2:
        return {"assessment": "insufficient_data", "subgroups": subgroup_metrics}

    max_rate = max(flagged_rates)
    min_rate = min(r for r in flagged_rates if r > 0) if any(r > 0 for r in flagged_rates) else 0.001
    disparity_ratio = max_rate / min_rate if min_rate > 0 else float("inf")

    # Assessment thresholds (HHS AI Trustworthiness Framework)
    if disparity_ratio <= 1.25:
        assessment = "pass"
    elif disparity_ratio <= 1.50:
        assessment = "flag"
    else:
        assessment = "fail"

    return {
        "disparity_ratio": round(disparity_ratio, 4),
        "max_flagged_rate": round(max_rate, 4),
        "min_flagged_rate": round(min_rate, 4),
        "demographic_parity_gap": round(max_rate - min_rate, 4),
        "assessment": assessment,
        "subgroups": subgroup_metrics,
    }


# ── Helpers ──────────────────────────────────────────────────────────

def _risk_level(score: float) -> str:
    if score >= 0.90:
        return "CRITICAL"
    elif score >= 0.65:
        return "HIGH"
    elif score >= 0.35:
        return "MEDIUM"
    return "LOW"


def _z_to_percentile(z: float) -> float:
    import math
    return round(50 * (1 + math.erf(z / math.sqrt(2))), 1)


def _compute_counterfactuals(
    contributions: Dict[str, float],
    current_score: float,
    weights: Dict[str, float],
) -> List[Dict]:
    """What feature changes would lower the risk level?"""
    if current_score < 0.35:
        return []  # Already LOW risk

    target = 0.64 if current_score >= 0.65 else 0.34
    needed_reduction = current_score - target

    counterfactuals = []
    for feature, contrib in sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True):
        if contrib <= 0:
            continue
        weight = weights.get(feature, 0.1)
        if weight > 0 and contrib >= needed_reduction * 0.3:
            needed_value_change = needed_reduction / weight
            counterfactuals.append({
                "feature": feature,
                "current_contribution": round(contrib, 4),
                "needed_reduction": round(needed_reduction, 4),
                "result": f"Addressing {feature} could lower risk from {_risk_level(current_score)} to {_risk_level(target)}",
                "actionable": feature not in ("age", "sex", "geographic_latitude"),
            })

    return counterfactuals[:3]


def _generate_explanation(
    score: float,
    top_contribs: List[Dict],
    counterfactuals: List[Dict],
) -> str:
    level = _risk_level(score)
    parts = [f"MS risk score: {score:.2f} ({level})."]

    if top_contribs:
        drivers = [
            f"{c['feature']} (contribution: {c['contribution']:.3f})"
            for c in top_contribs[:3]
        ]
        parts.append(f"Top risk factors: {', '.join(drivers)}.")

    actionable = [cf for cf in counterfactuals if cf.get("actionable")]
    if actionable:
        parts.append(
            f"Modifiable factor: {actionable[0]['feature']} — "
            f"{actionable[0]['result']}."
        )

    return " ".join(parts)
