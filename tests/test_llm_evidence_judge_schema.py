from __future__ import annotations

import pytest

from climate_pipeline.llm_evidence_judge import (
    JudgeSchemaError,
    deterministic_arbiter,
    validate_arbiter_judgment,
    validate_primary_judgment,
    validate_skeptic_judgment,
)


def _valid_primary() -> dict:
    return {
        "candidate_id": "rawce_a94472747c9c47e7",
        "source_url": "https://example.org",
        "source_title": "Example",
        "page_type": "official_response_page",
        "status": "accepted",
        "component_supported": "wet",
        "hazard_supported": True,
        "impact_supported": True,
        "impact_types": ["transportation", "evacuation_or_rescue"],
        "location_match": "same_county",
        "time_match": "exact_window",
        "source_role": "primary_impact",
        "explicit_transition_support": False,
        "quoted_supporting_spans": ["01/10/2023 evacuation order for Salinas River flooding."],
        "reason": "Span names date, river, flooding, and evacuation order.",
        "failure_reason_if_rejected": "none",
    }


def test_primary_judgment_schema_accepts_valid_payload() -> None:
    payload = _valid_primary()

    normalized = validate_primary_judgment(payload)

    assert normalized["status"] == "accepted"
    assert normalized["component_supported"] == "wet"


def test_primary_judgment_rejects_accepted_without_quote() -> None:
    payload = _valid_primary()
    payload["quoted_supporting_spans"] = []

    with pytest.raises(JudgeSchemaError):
        validate_primary_judgment(payload)


def test_primary_judgment_rejects_year_only_accepted_wet_impact() -> None:
    payload = _valid_primary()
    payload["time_match"] = "year_only"

    with pytest.raises(JudgeSchemaError):
        validate_primary_judgment(payload)


def test_skeptic_schema_and_deterministic_arbiter_downgrade() -> None:
    primary = _valid_primary()
    skeptic = validate_skeptic_judgment(
        {
            "skeptic_decision": "reject",
            "skeptic_reason": "The page is a duplicate URL.",
            "identified_failure_modes": ["duplicate_url"],
            "minimum_safe_status": "rejected",
        }
    )

    final = deterministic_arbiter(primary, skeptic)

    assert final["final_status"] == "rejected"
    assert final["final_impact_supported"] is False


def test_arbiter_schema_accepts_context_only_payload() -> None:
    payload = {
        "final_status": "context_only",
        "final_component_supported": "none",
        "final_impact_supported": False,
        "final_impact_types": [],
        "final_source_role": "context",
        "final_reason": "Broad resource page only.",
        "final_quoted_spans": [],
        "disagreement_resolution": "skeptic_context_only",
    }

    normalized = validate_arbiter_judgment(payload)

    assert normalized["final_status"] == "context_only"



def test_primary_schema_normalizes_common_aliases_and_missing_failure_reason() -> None:
    payload = _valid_primary()
    payload["impact_types"] = ["evacuation", "power outage", "road closure", "damage", "shelter"]
    payload.pop("failure_reason_if_rejected")

    normalized = validate_primary_judgment(payload)

    assert normalized["failure_reason_if_rejected"] == "none"
    assert normalized["impact_types"] == [
        "evacuation_or_rescue",
        "public_services",
        "transportation",
        "property_damage",
    ]
    assert "original_impact_types" in normalized
    assert any("failure_reason_if_rejected:<missing>->none" in note for note in normalized["schema_normalization_notes"])


def test_primary_schema_normalizes_weak_time_and_location_failure_reasons() -> None:
    payload = _valid_primary()
    payload["status"] = "context_only"
    payload["impact_supported"] = False
    payload["impact_types"] = []
    payload["time_match"] = "year_only"
    payload["failure_reason_if_rejected"] = "year_only_or_weak_time"

    normalized = validate_primary_judgment(payload)

    assert normalized["failure_reason_if_rejected"] == "weak_time_grounding"

    payload = _valid_primary()
    payload["status"] = "context_only"
    payload["impact_supported"] = False
    payload["impact_types"] = []
    payload["location_match"] = "regional_only"
    payload["failure_reason_if_rejected"] = "regional_only"

    normalized = validate_primary_judgment(payload)

    assert normalized["failure_reason_if_rejected"] == "weak_location_grounding"


def test_arbiter_schema_normalizes_final_impact_type_aliases() -> None:
    payload = {
        "final_status": "needs_review",
        "final_component_supported": "none",
        "final_impact_supported": False,
        "final_impact_types": ["rescue", "power_outage", "shelter"],
        "final_source_role": "reject",
        "final_reason": "Needs review.",
        "final_quoted_spans": [],
        "disagreement_resolution": "test",
    }

    normalized = validate_arbiter_judgment(payload)

    assert normalized["final_impact_types"] == ["evacuation_or_rescue", "public_services"]
    assert "original_final_impact_types" in normalized
