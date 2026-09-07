from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

from climate_pipeline import full_body_direct_judge as direct
from climate_pipeline import llm_evidence_validation as validation


BODY = (
    "Officials in Example County reported that heavy rain caused flooding on January 10, 2023. "
    "The flood damaged two homes. A warning issued January 8 predicted additional rain. "
    "The county later approved recovery assistance after the event."
)
CANDIDATE = {
    "candidate_id": "candidate-fixture",
    "county": "Example County",
    "locality_hints": [],
    "wet_window_start": "2023-01-09",
    "wet_window_end": "2023-01-11",
    "target_hazards": ["rainfall", "flood"],
}


def semantic(**overrides: object) -> dict:
    payload = {
        "source_readability": "readable",
        "readability_reason": "Readable article body.",
        "event_identity": "one_uniquely_relevant_event",
        "event_actuality": "observed",
        "date_mentions": [
            {
                "text": "January 10, 2023",
                "semantic_role": "event_day",
                "normalized_start": "2023-01-10",
                "normalized_end": "2023-01-10",
                "quote": "heavy rain caused flooding on January 10, 2023",
            }
        ],
        "event_date_relation": "aligned",
        "event_locations": ["Example County"],
        "location_quotes": ["Officials in Example County reported"],
        "event_location_relation": "aligned",
        "observed_hazards": ["heavy rain", "flooding"],
        "prospective_hazards": [],
        "hazard_quotes": ["heavy rain caused flooding"],
        "target_hazard_status": "observed",
        "realized_impacts": [{"claim": "two homes damaged", "quote": "The flood damaged two homes."}],
        "potential_impacts_or_risks": [],
        "asset_or_economic_values": [],
        "aid_or_administrative_actions": [],
        "target_hazard_observed_impact": "yes",
        "explicit_target_hazard_to_impact_attribution": "yes",
        "attribution_quotes": ["The flood damaged two homes."],
        "explicit_drought_to_wet_transition": "no",
        "transition_quotes": [],
        "candidate_support_recommendation": "supports",
        "supporting_quotes": ["Officials in Example County reported that heavy rain caused flooding on January 10, 2023."],
        "uncertainty": "",
    }
    payload.update(overrides)
    return payload


def semantic_v2(**event_overrides: object) -> dict:
    event = {
        "event_summary": "Observed Example County flood on January 10, 2023.",
        "binding_status": "explicit_same_event",
        "actuality": "observed",
        "date_relation": "aligned",
        "date_evidence": [
            {
                "semantic_role": "event_day",
                "precision": "exact_day",
                "normalized_start": "2023-01-10",
                "normalized_end": "2023-01-10",
                "continuous_event_period": False,
                "relative_anchor": None,
                "quote": "heavy rain caused flooding on January 10, 2023",
            }
        ],
        "observed_status_as_of": None,
        "location_relation": "aligned",
        "location_evidence": [
            {"location": "Example County", "kind": "event_location", "quote": "Officials in Example County reported"}
        ],
        "hazard_status": "observed",
        "hazard_evidence": [
            {"hazard": "heavy rain", "actuality": "observed", "quote": "heavy rain caused flooding"},
            {"hazard": "flooding", "actuality": "observed", "quote": "heavy rain caused flooding"},
        ],
        "realized_impacts": [{"claim": "two homes damaged", "quote": "The flood damaged two homes."}],
        "potential_impacts_or_risks": [],
        "asset_or_economic_values": [],
        "aid_or_administrative_actions": [],
        "observed_impact": "yes",
        "attribution": "yes",
        "attribution_quotes": ["The flood damaged two homes."],
        "transition": "no",
        "transition_quotes": [],
        "supporting_quotes": [
            "Officials in Example County reported that heavy rain caused flooding on January 10, 2023."
        ],
    }
    event.update(event_overrides)
    return {
        "source_readability": "readable",
        "readability_reason": "Readable article body.",
        "event_identity": "one_uniquely_relevant_event",
        "candidate_event": event,
        "candidate_support_recommendation": "supports",
        "uncertainty": "",
    }


def result(payload: dict, *, body: str = BODY, candidate: dict | None = None) -> dict:
    return direct.derive_local_page_result(
        semantic=payload,
        candidate=candidate or CANDIDATE,
        body_text=body,
    )


def test_direct_candidate_metadata_targets_drought_window_for_drought_gap() -> None:
    context = validation.CandidateContext(
        candidate_id="candidate-drought",
        county="Example County",
        county_fips="00001",
        state="California",
        drought_window="2022-05 to 2022-12",
        wet_window="2023-01-09 to 2023-01-11",
        candidate_stratum="test",
        locality_hints=[],
    )

    metadata = validation.direct_candidate_metadata(
        context,
        page={"target_gap": "drought_hazard_public_corroboration"},
    )

    assert metadata["target_hazard_axis"] == "drought"
    assert metadata["target_window_start"] == "2022-05-01"
    assert metadata["target_window_end"] == "2022-12-31"
    assert metadata["wet_window_start"] == "2023-01-09"
    assert metadata["wet_window_end"] == "2023-01-11"
    assert "drought" in metadata["target_hazards"]


@pytest.mark.parametrize(
    ("page", "expected_axis"),
    [
        (
            {
                "target_gap": "impact_enrichment",
                "expansion_trigger": "wet_impact_exists_but_drought_impact_missing",
            },
            "drought",
        ),
        (
            {
                "target_gap": "impact",
                "source_lane": "agriculture_drought_impact",
            },
            "drought",
        ),
        ({"target_gap": "impact_enrichment"}, "wet"),
    ],
)
def test_target_hazard_axis_uses_trigger_and_source_lane(
    page: dict[str, str],
    expected_axis: str,
) -> None:
    assert validation.target_hazard_axis_for_page(page) == expected_axis


def test_drought_targeted_page_can_pass_same_evidence_guards() -> None:
    drought_quote = (
        "Example County experienced severe drought conditions on June 15, 2022."
    )
    drought_body = (
        f"{drought_quote} County officials documented the observed conditions "
        "and published the report for local residents."
    )
    payload = semantic_v2(
        event_summary="Observed Example County drought on June 15, 2022.",
        date_evidence=[
            {
                "semantic_role": "event_day",
                "precision": "exact_day",
                "normalized_start": "2022-06-15",
                "normalized_end": "2022-06-15",
                "continuous_event_period": False,
                "relative_anchor": None,
                "quote": "drought conditions on June 15, 2022",
            }
        ],
        location_evidence=[
            {
                "location": "Example County",
                "kind": "event_location",
                "quote": drought_quote,
            }
        ],
        hazard_evidence=[
            {
                "hazard": "drought",
                "actuality": "observed",
                "quote": drought_quote,
            }
        ],
        realized_impacts=[],
        observed_impact="no",
        attribution="no",
        attribution_quotes=[],
        supporting_quotes=[drought_quote],
    )
    candidate = {
        **CANDIDATE,
        "drought_window": "2022-05 to 2022-12",
        "target_hazard_axis": "drought",
        "target_window_start": "2022-05-01",
        "target_window_end": "2022-12-31",
        "target_hazards": ["drought", "dry conditions"],
    }

    guarded = result(payload, body=drought_body, candidate=candidate)

    assert guarded["page_result"] == "supports"
    assert guarded["candidate_hazard_support"] == "yes"
    assert guarded["realized_impact_support"] == "no"


def test_publication_or_warning_dates_cannot_establish_event_timing() -> None:
    payload = semantic(
        date_mentions=[
            {
                "text": "January 10, 2023",
                "semantic_role": "publication",
                "normalized_start": "2023-01-10",
                "normalized_end": "2023-01-10",
                "quote": "January 10, 2023",
            },
            {
                "text": "January 8",
                "semantic_role": "warning",
                "normalized_start": "2023-01-08",
                "normalized_end": "2023-01-08",
                "quote": "A warning issued January 8",
            },
        ]
    )

    guarded = result(payload)

    assert guarded["page_result"] == "unresolved"
    assert any(
        action["guard"] in {"candidate_event_binding", "event_date"} and action["action"] == "blocked"
        for action in guarded["guard_actions"]
    )


def test_wrong_window_event_cannot_support_candidate() -> None:
    payload = semantic()
    payload["event_date_relation"] = "outside_candidate"
    payload["date_mentions"][0]["text"] = "January 10, 2024"
    payload["date_mentions"][0]["normalized_start"] = "2024-01-10"
    payload["date_mentions"][0]["normalized_end"] = "2024-01-10"
    payload["date_mentions"][0]["quote"] = "heavy rain caused flooding on January 10, 2024"
    payload["supporting_quotes"] = [
        "Officials in Example County reported that heavy rain caused flooding on January 10, 2024."
    ]
    body = BODY + " Officials in Example County reported that heavy rain caused flooding on January 10, 2024."

    guarded = result(payload, body=body)

    assert guarded["page_result"] == "does_not_support"
    assert guarded["candidate_hazard_support"] == "no"
    assert guarded["source_event_axes"]["realized_target_hazard_impact"] == "yes"
    assert guarded["source_event_axes"]["explicit_target_hazard_to_impact_attribution"] == "yes"


def test_model_normalized_weekday_bound_to_event_quote_can_close_outside_window() -> None:
    payload = semantic(event_date_relation="outside_candidate", candidate_support_recommendation="does_not_support")
    payload["date_mentions"][0].update(
        {
            "text": "Tuesday afternoon",
            "normalized_start": "2024-01-09",
            "normalized_end": "2024-01-09",
            "quote": "heavy rain caused flooding on Tuesday afternoon",
        }
    )
    body = BODY + " heavy rain caused flooding on Tuesday afternoon"
    assert result(payload, body=body)["page_result"] == "does_not_support"


def test_wrong_location_event_cannot_support_candidate() -> None:
    body = BODY.replace("Example County", "Other County")
    payload = semantic(
        event_locations=["Other County"],
        location_quotes=["Officials in Other County reported"],
        event_location_relation="not_aligned",
        supporting_quotes=["Officials in Other County reported that heavy rain caused flooding on January 10, 2023."],
    )

    guarded = result(payload, body=body)

    assert guarded["page_result"] == "does_not_support"


def test_wind_only_cannot_support_rainfall_candidate() -> None:
    body = BODY.replace("heavy rain caused flooding", "extreme wind toppled trees")
    payload = semantic(
        observed_hazards=["extreme wind"],
        hazard_quotes=["extreme wind toppled trees"],
        realized_impacts=[],
        target_hazard_observed_impact="no",
        explicit_target_hazard_to_impact_attribution="no",
        attribution_quotes=[],
        supporting_quotes=["Officials in Example County reported that extreme wind toppled trees on January 10, 2023."],
    )
    payload["date_mentions"][0]["quote"] = "extreme wind toppled trees on January 10, 2023"

    guarded = result(payload, body=body)

    assert guarded["page_result"] == "unresolved"


def test_warning_or_order_cannot_become_realized_impact() -> None:
    payload = semantic(
        event_actuality="warning_or_forecast",
        observed_hazards=[],
        prospective_hazards=["rainfall", "flood"],
        realized_impacts=[],
        potential_impacts_or_risks=[{"claim": "possible flooding", "quote": "predicted additional rain"}],
        target_hazard_observed_impact="no",
        explicit_target_hazard_to_impact_attribution="no",
        attribution_quotes=[],
        candidate_support_recommendation="does_not_support",
    )

    guarded = result(payload)

    assert guarded["page_result"] == "does_not_support"
    assert guarded["realized_impact_support"] == "no"


def test_mixed_page_with_only_prospective_target_hazard_and_no_observed_impact_is_non_support() -> None:
    payload = semantic(
        event_actuality="mixed_or_uncertain",
        target_hazard_status="anticipated_or_warning",
        observed_hazards=[],
        prospective_hazards=["forecast heavy rain", "flood warning"],
        hazard_quotes=["predicted additional rain"],
        realized_impacts=[],
        target_hazard_observed_impact="no",
        explicit_target_hazard_to_impact_attribution="no",
        attribution_quotes=[],
        candidate_support_recommendation="does_not_support",
    )

    guarded = result(payload)

    assert guarded["page_result"] == "does_not_support"
    assert guarded["realized_impact_support"] == "no"


def test_asset_or_crop_value_cannot_become_actual_damage() -> None:
    body = BODY + " The annual crop value was $50 million."
    payload = semantic(
        realized_impacts=[],
        asset_or_economic_values=[{"claim": "$50 million crop value", "quote": "The annual crop value was $50 million."}],
        target_hazard_observed_impact="yes",  # deliberately inconsistent model field
        explicit_target_hazard_to_impact_attribution="no",
        attribution_quotes=[],
    )

    guarded = result(payload, body=body)

    assert guarded["page_result"] == "supports"
    assert guarded["realized_impact_support"] == "no"


def test_cooccurrence_cannot_become_attribution() -> None:
    payload = semantic(
        explicit_target_hazard_to_impact_attribution="yes",
        attribution_quotes=[],
    )

    guarded = result(payload)

    assert guarded["explicit_attribution_support"] == "no"


def test_chronology_cannot_become_transition() -> None:
    payload = semantic(
        explicit_drought_to_wet_transition="yes",
        transition_quotes=[],
    )

    guarded = result(payload)

    assert guarded["explicit_drought_to_wet_transition_support"] == "no"


def test_separate_legacy_arrays_cannot_construct_same_event_support() -> None:
    payload = semantic(
        supporting_quotes=[
            "Officials in Example County reported",
            "heavy rain caused flooding on January 10, 2023",
        ]
    )

    guarded = result(payload)

    assert guarded["page_result"] == "unresolved"
    assert any(
        action["guard"] == "legacy_same_event_package" and action["action"] == "blocked"
        for action in guarded["guard_actions"]
    )


def test_single_legacy_quote_can_mechanically_bind_decisive_support_fields() -> None:
    guarded = result(semantic())
    assert guarded["page_result"] == "supports"
    assert any(
        action["guard"] == "legacy_same_event_package" and action["action"] == "passed"
        for action in guarded["guard_actions"]
    )


def test_narrow_ocr_spacing_reversal_can_bind_an_ordinal_event_day() -> None:
    payload = semantic()
    payload["date_mentions"][0].update(
        {
            "text": "afternoon of the 10th",
            "quote": "heavy rain caused flooding in Example County on the afterno on of the 10 th",
        }
    )
    payload["supporting_quotes"] = [
        "heavy rain caused flooding in Example County on the afterno on of the 10 th"
    ]
    body = BODY + " heavy rain caused flooding in Example County on the afterno on of the 10 th"
    assert result(payload, body=body)["page_result"] == "supports"


def test_model_unusable_on_objectively_readable_body_is_unresolved() -> None:
    guarded = result(semantic(source_readability="unusable"))
    assert guarded["page_result"] == "unresolved"


def test_broad_attribution_and_transition_are_not_candidate_event_axes() -> None:
    body = BODY + " Statewide, flooding damaged roads. January followed years of drought."
    payload = semantic(
        attribution_quotes=["Statewide, flooding damaged roads."],
        explicit_drought_to_wet_transition="yes",
        transition_quotes=["January followed years of drought."],
    )
    guarded = result(payload, body=body)
    assert guarded["page_result"] == "supports"
    assert guarded["explicit_attribution_support"] == "no"
    assert guarded["explicit_drought_to_wet_transition_support"] == "no"


def test_ungrounded_nondecisive_quote_is_filtered_without_overriding_wrong_window() -> None:
    payload = semantic(supporting_quotes=["This sentence is not in the body."])
    payload["event_date_relation"] = "outside_candidate"
    payload["date_mentions"][0]["text"] = "January 10, 2024"
    payload["date_mentions"][0]["normalized_start"] = "2024-01-10"
    payload["date_mentions"][0]["normalized_end"] = "2024-01-10"
    payload["date_mentions"][0]["quote"] = "heavy rain caused flooding on January 10, 2024"
    body = BODY + " heavy rain caused flooding on January 10, 2024"

    guarded = result(payload, body=body)

    assert guarded["page_result"] == "does_not_support"
    assert any(action["guard"] == "quotation_grounding" and action["action"] == "filtered" for action in guarded["guard_actions"])


def test_ungrounded_realized_impact_cannot_support_impact() -> None:
    payload = semantic(
        realized_impacts=[{"claim": "invented damage", "quote": "Invented damage quote."}],
        attribution_quotes=["Invented attribution quote."],
    )

    guarded = result(payload)

    assert guarded["page_result"] == "supports"
    assert guarded["realized_impact_support"] == "no"
    assert guarded["explicit_attribution_support"] == "no"


class _Response:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        assert size >= 0, "production response reads must always be bounded"
        return json.dumps(self.payload).encode("utf-8")[:size]


def _valid_response() -> dict:
    return {
        "status": "completed",
        "output": [{"content": [{"type": "output_text", "text": json.dumps(semantic_v2())}]}],
        "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
    }


def test_unreadable_content_does_not_generate_model_request() -> None:
    calls = []

    def opener(*args: object, **kwargs: object) -> _Response:
        calls.append((args, kwargs))
        return _Response(_valid_response())

    client = direct.DirectPageJudgeClient(api_key="test-key", opener=opener)
    call = client.judge(candidate=CANDIDATE, source={}, body_text="Title:\n\nMarkdown Content:")

    assert call.status == "not_requested_unusable_source"
    assert call.attempts == 0
    assert calls == []


def test_unreadable_page_does_not_contaminate_candidate_axes() -> None:
    guarded = direct.derive_local_page_result(
        semantic=None,
        candidate=CANDIDATE,
        body_text="Title:\n\nMarkdown Content:",
        call_status="not_requested_unusable_source",
    )
    assert guarded["page_result"] == "insufficient_source_content"
    assert guarded["candidate_hazard_support"] == "no"
    assert guarded["realized_impact_support"] == "no"
    assert guarded["explicit_attribution_support"] == "no"
    assert guarded["explicit_drought_to_wet_transition_support"] == "no"


def test_explicit_incomplete_retries_once_with_identical_request() -> None:
    request_bodies: list[bytes] = []
    responses = iter(
        [
            {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "usage": {"input_tokens": 100, "output_tokens": 5000}},
            _valid_response(),
        ]
    )

    def opener(request: object, **_: object) -> _Response:
        request_bodies.append(request.data)  # type: ignore[attr-defined]
        return _Response(next(responses))

    client = direct.DirectPageJudgeClient(api_key="test-key", opener=opener)
    call = client.judge(candidate=CANDIDATE, source={}, body_text=BODY)

    assert call.status == "ok"
    assert call.attempts == 2
    assert len(request_bodies) == 2
    assert request_bodies[0] == request_bodies[1]


def test_invalid_output_is_terminal_and_never_retried_more_than_once() -> None:
    calls = 0

    def opener(*_: object, **__: object) -> _Response:
        nonlocal calls
        calls += 1
        return _Response({"status": "completed", "output": [{"content": [{"type": "output_text", "text": "{}"}]}]})

    client = direct.DirectPageJudgeClient(api_key="test-key", opener=opener)
    call = client.judge(candidate=CANDIDATE, source={}, body_text=BODY)

    assert call.status == "invalid_output"
    assert call.attempts == 1
    assert calls == 1


def test_unresolved_values_are_not_silently_converted() -> None:
    payload = semantic(event_identity="unresolved_event_identity", candidate_support_recommendation="does_not_support")
    assert result(payload)["page_result"] == "unresolved"

    payload = semantic(target_hazard_observed_impact="uncertain")
    guarded = result(payload)
    assert guarded["page_result"] == "supports"
    assert guarded["realized_impact_support"] == "unresolved"


def test_exact_model_reasoning_strict_schema_and_output_budget_are_frozen() -> None:
    client = direct.DirectPageJudgeClient(api_key="test-key", opener=lambda *_args, **_kwargs: _Response(_valid_response()))
    payload = client._payload(candidate=CANDIDATE, source={}, body_text=BODY)

    assert payload["model"] == "gpt-5.5-2026-04-23"
    assert payload["reasoning"] == {"effort": "high"}
    assert payload["max_output_tokens"] >= 5_000
    assert payload["text"]["format"]["strict"] is True


def test_no_case_county_url_or_known_quote_specific_rule_exists() -> None:
    source = inspect.getsource(direct).lower()
    forbidden = (
        "rawce_",
        "shasta",
        "placer",
        "mariposa",
        "redding.com",
        "abcnews.com",
        "instagram.com",
        "crop_asset_value_not_damage",
        "later_memorandum_explicitly_anchors_older_event",
    )

    assert not any(value in source for value in forbidden)


def test_enabled_validation_path_uses_one_full_body_call_and_no_other_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    class FakeDirectClient:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["model"] == direct.EXACT_MODEL

        def judge(self, *, candidate: dict, source: dict, body_text: str) -> SimpleNamespace:
            calls.append({"candidate": candidate, "source": source, "body_text": body_text})
            return SimpleNamespace(
                model=direct.EXACT_MODEL,
                status="ok",
                parsed=semantic(),
                raw_output_text=json.dumps(semantic()),
                raw_responses=[{"id": "response-fixture"}],
                attempts=1,
                usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
                error="",
                retry_reason="",
                request_sha256="request-fixture",
            )

    monkeypatch.setattr(validation, "read_openai_api_key", lambda path=None: "test-key")
    monkeypatch.setattr(validation, "DirectPageJudgeClient", FakeDirectClient)
    page = {
        "phase2_3_sample_id": "candidate-fixture",
        "candidate_id": "candidate-fixture",
        "county": "Example County",
        "state": "California",
        "source_url": "https://example.test/event",
        "source_title": "Example event",
        "body_text_or_archived_body_text": BODY,
    }

    output = validation.validate_fetched_pages_with_llm_judge(
        pages=[page],
        case_cards={
            "candidate-fixture": {
                "county": "Example County",
                "wet_window": "2023-01-09 to 2023-01-11",
            }
        },
        judge_config=validation.LLMEvidenceJudgeConfig(use_llm_evidence_judge=True),
    )

    assert len(calls) == 1
    assert calls[0]["body_text"] == BODY
    assert output.raw_model_outputs["skeptic"] == []
    assert output.raw_model_outputs["arbiter"] == []
    assert output.normalized_rows[0]["guarded_local_result"]["page_result"] == "supports"
    assert output.judgment_rows[0]["integrated_llm_status"] == "accepted"
