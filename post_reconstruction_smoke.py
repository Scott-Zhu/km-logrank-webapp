"""
Deterministic post-reconstruction bookkeeping checks (no live API calls).
"""

from __future__ import annotations

import sys
import types

# Allow importing llm_extraction without installed openai runtime.
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))

from llm_extraction import (  # noqa: E402
    build_canonical_reconstruction,
    cap_confidence_for_quality,
    grade_extraction_quality,
)


def run_smoke_checks() -> None:
    # 1) canonical totals consistency + interval delta consistency (terminal known)
    payload_known = {
        "groups": [
            {
                "name": "A",
                "initial_n": 10,
                "terminal_risk_known": True,
                "risk_table_counts": [10, 7, 5],
                "interval_summary": [
                    {
                        "interval": "0-1",
                        "interval_start": 0.0,
                        "interval_end": 1.0,
                        "at_risk_start": 10,
                        "next_known_at_risk": 7,
                        "estimated_events": 2,
                        "estimated_censors": 1,
                    },
                    {
                        "interval": "1-2",
                        "interval_start": 1.0,
                        "interval_end": 2.0,
                        "at_risk_start": 7,
                        "next_known_at_risk": 5,
                        "estimated_events": 1,
                        "estimated_censors": 1,
                    },
                ],
            }
        ]
    }
    known_out, _ = build_canonical_reconstruction(payload_known)
    known_canonical = known_out["canonical_reconstruction"]
    assert known_canonical["accounting_identities_passed"] is True
    assert known_canonical["interval_conservation_passed"] is True

    # 2) terminal unknown accounting
    payload_unknown = {
        "groups": [
            {
                "name": "B",
                "initial_n": 12,
                "terminal_risk_known": False,
                "risk_table_counts": [12, 8, None],
                "interval_summary": [
                    {
                        "interval": "0-1",
                        "interval_start": 0.0,
                        "interval_end": 1.0,
                        "at_risk_start": 12,
                        "next_known_at_risk": 8,
                        "estimated_events": 3,
                        "estimated_censors": 1,
                    },
                    {
                        "interval": "1-2",
                        "interval_start": 1.0,
                        "interval_end": 2.0,
                        "at_risk_start": 8,
                        "next_known_at_risk": None,
                        "estimated_events": 1,
                        "estimated_censors": 1,
                    },
                ],
            }
        ]
    }
    unknown_out, _ = build_canonical_reconstruction(payload_unknown)
    unknown_group = unknown_out["canonical_reconstruction"]["groups"][0]
    assert unknown_group["unresolved_tail_count"] >= 0

    # 3) pairwise suppression when quality low
    quality_payload = {
        "groups": [
            {
                "name": "X",
                "risk_table_counts": [10, None],
                "visible_censor_density": "high",
                "confidence_band_present": True,
                "unresolved_tail_count": 6,
                "terminal_risk_known": False,
            }
        ]
    }
    _, quality_summary = grade_extraction_quality(
        quality_payload,
        suspicious_rules=["group-order contradiction between mid-follow-up and right-tail ordering"],
        repair_notes=["repair1", "repair2", "repair3", "repair4"],
    )
    assert quality_summary["pairwise_recommendation"] in {"exploratory", "suppress"}

    # 4) confidence cap based on quality
    high_conf, _ = cap_confidence_for_quality(0.92, "high")
    medium_conf, _ = cap_confidence_for_quality(0.92, "medium")
    low_conf, _ = cap_confidence_for_quality(0.92, "low")
    assert high_conf >= medium_conf >= low_conf
    assert medium_conf <= 0.75
    assert low_conf <= 0.55


if __name__ == "__main__":
    run_smoke_checks()
    print("post_reconstruction_smoke: all checks passed")
