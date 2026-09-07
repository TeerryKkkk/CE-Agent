from __future__ import annotations

from pathlib import Path

import pytest

from climate_pipeline.evidence_span_extractor import CandidateContext, extract_candidate_spans
from climate_pipeline.llm_evidence_judge import apply_safety_gates
from climate_pipeline.llm_evidence_validation import (
    LLMEvidenceJudgeConfig,
    LLMEvidenceValidationUnavailable,
    pre_gate_page,
    validate_fetched_pages_with_llm_judge,
)


def _accepted_final() -> dict:
    return {
        "final_status": "accepted",
        "final_component_supported": "wet",
        "final_impact_supported": True,
        "final_impact_types": ["transportation"],
        "final_source_role": "primary_impact",
        "final_reason": "accepted",
        "final_quoted_spans": ["01/10/2023 Monterey County road closure from flooding."],
        "disagreement_resolution": "test",
    }


def _primary(**overrides: object) -> dict:
    payload = {
        "page_type": "official_response_page",
        "time_match": "exact_window",
        "location_match": "same_county",
        "component_supported": "wet",
        "impact_supported": True,
        "impact_types": ["transportation"],
    }
    payload.update(overrides)
    return payload


def test_no_quoted_span_cannot_remain_accepted() -> None:
    final = _accepted_final()
    final["final_quoted_spans"] = []

    guarded = apply_safety_gates(final=final, primary=_primary(), skeptic={}, source_metadata={})

    assert guarded["final_status"] == "needs_review"
    assert guarded["final_impact_supported"] is False


def test_accepted_quote_not_found_in_body_cannot_remain_accepted() -> None:
    final = _accepted_final()
    final["final_quoted_spans"] = ["01/10/2023 Monterey County road closure from flooding."]

    guarded = apply_safety_gates(
        final=final,
        primary=_primary(),
        skeptic={},
        source_metadata={},
        body_text="The fetched body only discusses a county board meeting and contains no storm damage details.",
        body_spans=["The fetched body only discusses a county board meeting."],
    )

    assert guarded["final_status"] == "needs_review"
    assert guarded["final_impact_supported"] is False
    assert guarded["failure_reason_if_rejected"] == "weak_impact_grounding"
    assert "quote_not_body_grounded" in guarded["disagreement_resolution"]


def test_accepted_quote_found_in_body_remains_accepted() -> None:
    quote = "01/10/2023 Monterey County road closure from flooding."
    final = _accepted_final()
    final["final_quoted_spans"] = [quote]

    guarded = apply_safety_gates(
        final=final,
        primary=_primary(),
        skeptic={},
        source_metadata={},
        body_text=f"Officials reported that {quote} Crews reopened the route later that day.",
        body_spans=[],
    )

    assert guarded["final_status"] == "accepted"


def test_year_only_time_match_cannot_support_wet_impact() -> None:
    guarded = apply_safety_gates(
        final=_accepted_final(),
        primary=_primary(time_match="year_only"),
        skeptic={},
        source_metadata={},
    )

    assert guarded["final_status"] == "rejected"
    assert guarded["failure_reason_if_rejected"] == "weak_time_grounding"


def test_generic_planning_page_rejected_after_llm() -> None:
    guarded = apply_safety_gates(
        final=_accepted_final(),
        primary=_primary(page_type="generic_planning_page"),
        skeptic={},
        source_metadata={},
    )

    assert guarded["final_status"] == "rejected"
    assert guarded["failure_reason_if_rejected"] == "generic_planning_page"


def test_generic_planning_context_only_is_still_rejected() -> None:
    final = _accepted_final()
    final["final_status"] = "context_only"
    final["final_impact_supported"] = False
    final["final_impact_types"] = []
    final["final_source_role"] = "context"

    guarded = apply_safety_gates(
        final=final,
        primary=_primary(page_type="generic_planning_page"),
        skeptic={},
        source_metadata={},
    )

    assert guarded["final_status"] == "rejected"
    assert guarded["failure_reason_if_rejected"] == "generic_planning_page"


def test_broad_congressional_resource_page_context_only() -> None:
    guarded = apply_safety_gates(
        final=_accepted_final(),
        primary=_primary(page_type="broad_resource_page"),
        skeptic={},
        source_metadata={"source_url": "https://lofgren.house.gov/2023-winter-storm"},
    )

    assert guarded["final_status"] == "context_only"
    assert guarded["final_impact_supported"] is False


def test_duplicate_url_not_counted_as_accepted_evidence() -> None:
    guarded = apply_safety_gates(
        final=_accepted_final(),
        primary=_primary(),
        skeptic={},
        source_metadata={"duplicate_for_candidate": True},
    )

    assert guarded["final_status"] == "rejected"
    assert guarded["failure_reason_if_rejected"] == "duplicate_url"


def test_component_impact_does_not_imply_transition_support() -> None:
    primary = _primary(explicit_transition_support=False)
    guarded = apply_safety_gates(final=_accepted_final(), primary=primary, skeptic={}, source_metadata={})

    assert guarded["final_status"] == "accepted"
    assert guarded["final_component_supported"] == "wet"
    assert primary["explicit_transition_support"] is False


def test_missing_llm_fails_closed_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("climate_pipeline.llm_evidence_validation.read_openai_api_key", lambda path=None: "")
    with pytest.raises(LLMEvidenceValidationUnavailable):
        validate_fetched_pages_with_llm_judge(
            pages=[],
            judge_config=LLMEvidenceJudgeConfig(
                use_llm_evidence_judge=True,
                llm_judge_fail_closed=True,
                api_key_path=tmp_path / "missing_key.txt",
            ),
        )


def test_broad_resource_pre_gate_marks_context_only_without_local_impact_span() -> None:
    context = CandidateContext(
        candidate_id="rawce_9d5649929a20ea52",
        county="Santa Clara County",
        county_fips="06085",
        state="California",
        wet_window="2022-12-31 to 2023-01-02",
        locality_hints=["San Jose"],
    )
    spans = extract_candidate_spans(
        candidate=context,
        source_url="https://lofgren.house.gov/2023-winter-storm",
        body_text="Congresswoman resources and assistance resources for California winter storm recovery.",
    )

    gate = pre_gate_page(
        page={"accepted_from_snippet": False, "source_title": "2023 Winter Storm Resources"},
        context=context,
        source_url="https://lofgren.house.gov/2023-winter-storm",
        source_family="other_official",
        duplicate_for_candidate=False,
        spans=spans,
    )

    assert gate["final_status"] == "context_only"
    assert gate["failure_reason"] == "broad_resource_page"
