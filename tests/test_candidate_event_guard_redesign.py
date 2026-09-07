from __future__ import annotations

import copy
import inspect

import pytest

from climate_pipeline import full_body_direct_judge as direct


BODY = (
    "Example County experienced heavy rain and flooding on January 10, 2023. "
    "Floodwater damaged two homes. Published January 12, 2023. "
    "A warning was issued January 8, 2023. State officials later declared an emergency."
)
CANDIDATE = {
    "candidate_id": "generic-candidate",
    "county": "Example County",
    "locality_hints": [],
    "wet_window_start": "2023-01-09",
    "wet_window_end": "2023-01-11",
    "target_hazards": ["rainfall", "flood"],
}


def judgment() -> dict:
    return {
        "source_readability": "readable",
        "readability_reason": "readable",
        "event_identity": "one_uniquely_relevant_event",
        "candidate_event": {
            "event_summary": "Example County rain and flood event",
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
                    "quote": "heavy rain and flooding on January 10, 2023",
                }
            ],
            "observed_status_as_of": None,
            "location_relation": "aligned",
            "location_evidence": [
                {
                    "location": "Example County",
                    "kind": "event_location",
                    "quote": "Example County experienced heavy rain and flooding",
                }
            ],
            "hazard_status": "observed",
            "hazard_evidence": [
                {
                    "hazard": "heavy rain and flooding",
                    "actuality": "observed",
                    "quote": "heavy rain and flooding on January 10, 2023",
                }
            ],
            "realized_impacts": [{"claim": "two homes damaged", "quote": "Floodwater damaged two homes."}],
            "potential_impacts_or_risks": [],
            "asset_or_economic_values": [],
            "aid_or_administrative_actions": [],
            "observed_impact": "yes",
            "attribution": "yes",
            "attribution_quotes": ["Floodwater damaged two homes."],
            "transition": "no",
            "transition_quotes": [],
            "supporting_quotes": ["Example County experienced heavy rain and flooding on January 10, 2023."],
        },
        "candidate_support_recommendation": "supports",
        "uncertainty": "",
    }


def guarded(payload: dict, body: str = BODY) -> dict:
    return direct.derive_local_page_result(semantic=payload, candidate=CANDIDATE, body_text=body)


def test_single_bound_candidate_event_supports() -> None:
    assert guarded(judgment())["page_result"] == "supports"


@pytest.mark.parametrize("precision", ["month", "year"])
def test_coarse_date_cannot_establish_narrow_window(precision: str) -> None:
    payload = judgment()
    payload["candidate_event"]["date_evidence"] = [
        {
            "semantic_role": "incident_period",
            "precision": precision,
            "normalized_start": "2023-01-01",
            "normalized_end": "2023-01-31" if precision == "month" else "2023-12-31",
            "continuous_event_period": False,
            "relative_anchor": None,
            "quote": "January 2023" if precision == "month" else "2023",
        }
    ]
    assert guarded(payload, BODY + " January 2023 2023")["page_result"] == "unresolved"


def test_explicit_continuous_event_period_can_cover_window() -> None:
    payload = judgment()
    payload["candidate_event"]["date_evidence"] = [
        {
            "semantic_role": "incident_period",
            "precision": "exact_interval",
            "normalized_start": "2023-01-09",
            "normalized_end": "2023-01-11",
            "continuous_event_period": True,
            "relative_anchor": None,
            "quote": "The event continued January 9, 2023 through January 11, 2023.",
        }
    ]
    body = BODY + " The event continued January 9, 2023 through January 11, 2023."
    assert guarded(payload, body)["page_result"] == "supports"


@pytest.mark.parametrize(
    "role", ["publication", "update", "warning_period", "declaration_period", "administrative_date"]
)
def test_non_event_date_roles_cannot_rescue_timing(role: str) -> None:
    payload = judgment()
    payload["candidate_event"]["date_evidence"][0]["semantic_role"] = role
    assert guarded(payload)["page_result"] == "unresolved"


@pytest.mark.parametrize("kind", ["document_scope", "agency_jurisdiction", "eligibility_area"])
def test_non_event_geography_cannot_establish_location(kind: str) -> None:
    payload = judgment()
    payload["candidate_event"]["location_evidence"][0]["kind"] = kind
    assert guarded(payload)["page_result"] == "unresolved"


def test_location_field_cannot_override_quote_from_other_event() -> None:
    payload = judgment()
    payload["candidate_event"]["location_evidence"][0]["quote"] = "Other County had flooding."
    assert guarded(payload, BODY + " Other County had flooding.")["page_result"] == "does_not_support"


def test_hazard_field_cannot_override_unrelated_grounded_quote() -> None:
    payload = judgment()
    payload["candidate_event"]["hazard_evidence"][0]["quote"] = "State officials later declared an emergency."
    assert guarded(payload)["page_result"] == "unresolved"


def test_date_normalization_must_be_bound_to_its_quote() -> None:
    payload = judgment()
    payload["candidate_event"]["date_evidence"][0]["normalized_start"] = "2023-01-09"
    payload["candidate_event"]["date_evidence"][0]["normalized_end"] = "2023-01-09"
    assert guarded(payload)["page_result"] == "unresolved"


@pytest.mark.parametrize("recommendation", ["does_not_support", "needs_review"])
def test_model_negative_or_uncertain_recommendation_is_a_positive_ceiling(recommendation: str) -> None:
    payload = judgment()
    payload["candidate_support_recommendation"] = recommendation
    assert guarded(payload)["page_result"] == "unresolved"


@pytest.mark.parametrize("relation", ["outside_candidate", "unknown"])
def test_model_negative_or_uncertain_date_is_never_upgraded(relation: str) -> None:
    payload = judgment()
    payload["candidate_event"]["date_relation"] = relation
    if relation == "outside_candidate":
        payload["candidate_event"]["date_evidence"][0].update(
            {
                "normalized_start": "2024-01-10",
                "normalized_end": "2024-01-10",
                "quote": "heavy rain and flooding on January 10, 2024",
            }
        )
    expected = "does_not_support" if relation == "outside_candidate" else "unresolved"
    body = BODY + " Example County experienced heavy rain and flooding on January 10, 2024."
    assert guarded(payload, body)["page_result"] == expected


def test_unverified_outside_relation_is_unresolved_not_negative_closure() -> None:
    payload = judgment()
    payload["candidate_event"]["date_relation"] = "outside_candidate"
    assert guarded(payload)["page_result"] == "unresolved"


def test_unresolved_binding_cannot_be_rescued_by_complete_page_values() -> None:
    payload = judgment()
    payload["candidate_event"]["binding_status"] = "unresolved"
    result = guarded(payload)
    assert result["page_result"] == "unresolved"
    assert result["explicit_drought_to_wet_transition_support"] == "no"


def test_broad_transition_uncertainty_does_not_contaminate_candidate_linkage_axis() -> None:
    payload = judgment()
    payload["candidate_event"]["binding_status"] = "unresolved"
    payload["candidate_event"]["transition"] = "uncertain"
    payload["candidate_event"]["transition_quotes"] = [
        "Statewide drought conditions preceded a wet winter."
    ]
    body = BODY + " Statewide drought conditions preceded a wet winter."
    result = guarded(payload, body)
    assert result["page_result"] == "unresolved"
    assert result["explicit_drought_to_wet_transition_support"] == "no"


@pytest.mark.parametrize("field", ["date_evidence", "location_evidence", "hazard_evidence"])
def test_deleting_required_event_bound_evidence_removes_support(field: str) -> None:
    payload = judgment()
    payload["candidate_event"][field] = []
    assert guarded(payload)["page_result"] != "supports"


def test_adding_unrelated_page_text_never_improves_result() -> None:
    payload = judgment()
    payload["candidate_event"]["date_relation"] = "outside_candidate"
    baseline = guarded(payload)["page_result"]
    expanded = guarded(payload, BODY + " Example County flooded on January 10, 2023 after heavy rain.")["page_result"]
    assert baseline == expanded == "unresolved"


def test_schema_has_one_nullable_candidate_event_and_no_events_array() -> None:
    assert "candidate_event" in direct.DIRECT_JUDGMENT_SCHEMA["properties"]
    assert "events" not in direct.DIRECT_JUDGMENT_SCHEMA["properties"]
    date_schema = direct.CANDIDATE_EVENT_SCHEMA["properties"]["date_evidence"]["items"]
    assert "text" not in date_schema["properties"]
    assert "relative_anchor" in date_schema["properties"]
    assert "observed_status_as_of" in direct.CANDIDATE_EVENT_SCHEMA["properties"]


def test_grounded_relative_yesterday_can_establish_same_event_date() -> None:
    payload = judgment()
    payload["candidate_event"]["date_evidence"] = [
        {
            "semantic_role": "event_day",
            "precision": "relative",
            "normalized_start": "2023-01-10",
            "normalized_end": "2023-01-10",
            "continuous_event_period": False,
            "relative_anchor": {
                "anchor_date": "2023-01-11",
                "anchor_role": "publication",
                "anchor_quote": "Published January 11, 2023",
            },
            "quote": "Yesterday, Example County experienced heavy rain and flooding.",
        }
    ]
    body = BODY + " Published January 11, 2023. Yesterday, Example County experienced heavy rain and flooding."
    assert guarded(payload, body)["page_result"] == "supports"


def test_relative_date_with_ungrounded_or_wrong_anchor_is_unresolved() -> None:
    payload = judgment()
    payload["candidate_event"]["date_evidence"] = [
        {
            "semantic_role": "event_day",
            "precision": "relative",
            "normalized_start": "2023-01-10",
            "normalized_end": "2023-01-10",
            "continuous_event_period": False,
            "relative_anchor": {
                "anchor_date": "2023-01-12",
                "anchor_role": "update",
                "anchor_quote": "Updated January 11, 2023",
            },
            "quote": "Yesterday, Example County experienced heavy rain and flooding.",
        }
    ]
    body = BODY + " Updated January 11, 2023. Yesterday, Example County experienced heavy rain and flooding."
    assert guarded(payload, body)["page_result"] == "unresolved"


@pytest.mark.parametrize(
    ("phrase", "expected_start", "expected_end"),
    [
        ("last Sunday", "2023-01-15", "2023-01-15"),
        ("the previous Monday", "2023-01-16", "2023-01-16"),
        ("last Sunday and last Monday", "2023-01-15", "2023-01-16"),
    ],
)
def test_explicitly_modified_weekdays_resolve_from_grounded_anchor(
    phrase: str, expected_start: str, expected_end: str
) -> None:
    payload = judgment()
    payload["candidate_support_recommendation"] = "does_not_support"
    payload["candidate_event"]["date_relation"] = "outside_candidate"
    payload["candidate_event"]["date_evidence"] = [
        {
            "semantic_role": "event_day",
            "precision": "relative",
            "normalized_start": "",
            "normalized_end": "",
            "continuous_event_period": False,
            "relative_anchor": {
                "anchor_date": "2023-01-18",
                "anchor_role": "publication",
                "anchor_quote": "Published January 18, 2023",
            },
            "quote": f"Example County experienced flooding {phrase}.",
        }
    ]
    body = BODY + f" Published January 18, 2023. Example County experienced flooding {phrase}."
    mention = payload["candidate_event"]["date_evidence"][0]
    assert direct._verified_date_evidence_interval(mention) == (
        direct._parse_iso_date(expected_start),
        direct._parse_iso_date(expected_end),
    )
    assert guarded(payload, body)["page_result"] == "does_not_support"


@pytest.mark.parametrize("phrase", ["Sunday", "Monday", "Sunday and Monday"])
def test_bare_weekday_interpretations_remain_unresolved(phrase: str) -> None:
    payload = judgment()
    payload["candidate_support_recommendation"] = "does_not_support"
    payload["candidate_event"]["date_relation"] = "outside_candidate"
    payload["candidate_event"]["date_evidence"] = [
        {
            "semantic_role": "event_day",
            "precision": "relative",
            "normalized_start": "",
            "normalized_end": "",
            "continuous_event_period": False,
            "relative_anchor": {
                "anchor_date": "2023-01-18",
                "anchor_role": "publication",
                "anchor_quote": "Published January 18, 2023",
            },
            "quote": f"Example County experienced flooding {phrase}.",
        }
    ]
    body = BODY + f" Published January 18, 2023. Example County experienced flooding {phrase}."
    assert guarded(payload, body)["page_result"] == "unresolved"


def test_publication_anchor_is_not_itself_an_event_date() -> None:
    payload = judgment()
    payload["candidate_support_recommendation"] = "does_not_support"
    payload["candidate_event"]["date_relation"] = "outside_candidate"
    payload["candidate_event"]["date_evidence"] = [
        {
            "semantic_role": "publication",
            "precision": "exact_day",
            "normalized_start": "2024-01-18",
            "normalized_end": "2024-01-18",
            "continuous_event_period": False,
            "relative_anchor": None,
            "quote": "Published January 18, 2024",
        }
    ]
    assert guarded(payload, BODY + " Published January 18, 2024")["page_result"] == "unresolved"


def test_observed_status_as_of_can_close_outside_date_but_bare_update_cannot() -> None:
    payload = judgment()
    event = payload["candidate_event"]
    event["date_relation"] = "outside_candidate"
    event["date_evidence"] = []
    event["observed_status_as_of"] = {
        "normalized_date": "2024-01-10",
        "date_quote": "Example County roads remain closed due to flooding as of January 10, 2024.",
        "status_quote": "roads remain closed due to flooding",
    }
    body = BODY + " Updated January 10, 2024. Example County roads remain closed due to flooding as of January 10, 2024."
    assert guarded(payload, body)["page_result"] == "does_not_support"
    event["observed_status_as_of"]["date_quote"] = "Updated January 10, 2024"
    event["observed_status_as_of"]["status_quote"] = "Updated January 10, 2024"
    assert guarded(payload, body)["page_result"] == "unresolved"


def test_one_complete_observed_status_span_can_close_negative() -> None:
    payload = judgment()
    event = payload["candidate_event"]
    payload["candidate_support_recommendation"] = "does_not_support"
    event["date_relation"] = "outside_candidate"
    event["date_evidence"] = [
        {
            "semantic_role": "observed_status_as_of",
            "precision": "exact_day",
            "normalized_start": "2024-01-20",
            "normalized_end": "2024-01-20",
            "continuous_event_period": False,
            "relative_anchor": None,
            "quote": "Updated January 20, 2024: Example County roads remain closed due to flooding.",
        }
    ]
    event["observed_status_as_of"] = None
    body = BODY + " Updated January 20, 2024: Example County roads remain closed due to flooding."
    assert guarded(payload, body)["page_result"] == "does_not_support"


def test_observed_status_does_not_combine_unrelated_quotes() -> None:
    payload = judgment()
    event = payload["candidate_event"]
    payload["candidate_support_recommendation"] = "does_not_support"
    event["date_relation"] = "outside_candidate"
    event["date_evidence"] = [
        {
            "semantic_role": "observed_status_as_of",
            "precision": "exact_day",
            "normalized_start": "2024-01-20",
            "normalized_end": "2024-01-20",
            "continuous_event_period": False,
            "relative_anchor": None,
            "quote": "Updated January 20, 2024",
        }
    ]
    event["observed_status_as_of"] = {
        "normalized_date": "2024-01-20",
        "date_quote": "Updated January 20, 2024",
        "status_quote": "Example County roads remain closed due to flooding.",
    }
    body = BODY + " Updated January 20, 2024. Example County roads remain closed due to flooding."
    assert guarded(payload, body)["page_result"] == "unresolved"


def test_unique_event_in_explicit_other_county_can_close_negative() -> None:
    payload = judgment()
    payload["candidate_support_recommendation"] = "does_not_support"
    event = payload["candidate_event"]
    event["location_relation"] = "unknown"
    event["location_evidence"] = [
        {
            "location": "Other County",
            "kind": "event_location",
            "quote": "The flooding occurred in Other County.",
        }
    ]
    assert guarded(payload, BODY + " The flooding occurred in Other County.")["page_result"] == "does_not_support"


def test_nonexhaustive_other_location_example_remains_unresolved() -> None:
    payload = judgment()
    payload["candidate_support_recommendation"] = "does_not_support"
    event = payload["candidate_event"]
    event["location_relation"] = "unknown"
    event["location_evidence"] = [
        {
            "location": "Other County",
            "kind": "event_location",
            "quote": "The flooding affected several places, including Other County.",
        }
    ]
    body = BODY + " The flooding affected several places, including Other County."
    assert guarded(payload, body)["page_result"] == "unresolved"


@pytest.mark.parametrize("kind", ["document_scope", "agency_jurisdiction", "eligibility_area"])
def test_non_event_location_roles_cannot_trigger_negative_geography_closure(kind: str) -> None:
    payload = judgment()
    payload["candidate_support_recommendation"] = "does_not_support"
    event = payload["candidate_event"]
    event["location_relation"] = "unknown"
    event["location_evidence"] = [
        {"location": "Other County", "kind": kind, "quote": "The program applies in Other County."}
    ]
    assert guarded(payload, BODY + " The program applies in Other County.")["page_result"] == "unresolved"


def test_broad_incident_period_without_two_grounded_endpoints_is_unresolved() -> None:
    payload = judgment()
    payload["candidate_event"]["date_evidence"] = [
        {
            "semantic_role": "incident_period",
            "precision": "exact_interval",
            "normalized_start": "2023-01-01",
            "normalized_end": "2023-01-31",
            "continuous_event_period": True,
            "relative_anchor": None,
            "quote": "The January 2023 incident period covered the region.",
        }
    ]
    assert guarded(payload, BODY + " The January 2023 incident period covered the region.")["page_result"] == "unresolved"


def test_multiple_plausible_events_require_null_candidate_event() -> None:
    payload = judgment()
    payload["event_identity"] = "unresolved_event_identity"
    payload["candidate_event"] = None
    payload["candidate_support_recommendation"] = "needs_review"
    assert guarded(payload)["page_result"] == "unresolved"
    direct.validate_direct_judgment(payload)


def test_native_schema_rejects_wrong_event_text_field_and_overlong_arrays() -> None:
    payload = judgment()
    payload["candidate_event"]["date_evidence"][0]["text"] = "January 10, 2023"
    with pytest.raises(direct.DirectJudgeSchemaError):
        direct.validate_direct_judgment(payload)
    payload = judgment()
    payload["candidate_event"]["supporting_quotes"] *= 5
    with pytest.raises(direct.DirectJudgeSchemaError):
        direct.validate_direct_judgment(payload)


def test_no_case_county_url_page_quote_or_expected_result_specific_rule() -> None:
    source = inspect.getsource(direct).lower()
    forbidden = ("rawce_", "expected_page_changes", "12 → 4", "12 -> 4", "glenn", "santa clara")
    assert not any(value in source for value in forbidden)


def test_permuting_or_duplicating_nondecisive_quotes_cannot_create_support() -> None:
    payload = judgment()
    payload["candidate_event"]["binding_status"] = "unresolved"
    baseline = guarded(payload)["page_result"]
    for quotes in (
        [],
        list(reversed(payload["candidate_event"]["supporting_quotes"])),
        payload["candidate_event"]["supporting_quotes"] * 3,
    ):
        variant = copy.deepcopy(payload)
        variant["candidate_event"]["supporting_quotes"] = quotes
        assert guarded(variant)["page_result"] == baseline == "unresolved"


def test_explicit_adjacent_anaphora_can_bind_one_event_without_page_wide_search() -> None:
    assert direct._explicit_adjacent_event_coreference(
        "On January 10, the first in a series of storms struck Example County.",
        "This series of storms produced heavy rain and flooding.",
    )
    assert not direct._explicit_adjacent_event_coreference(
        "A storm struck Other County last year.",
        "Example County experienced heavy rain this year.",
    )


def test_incomplete_output_cannot_contaminate_linkage_when_drought_is_absent() -> None:
    result = direct.derive_local_page_result(
        semantic=None,
        candidate=CANDIDATE,
        body_text=BODY,
        call_status="incomplete",
    )
    assert result["page_result"] == "unresolved"
    assert result["candidate_hazard_support"] == "unresolved"
    assert result["explicit_drought_to_wet_transition_support"] == "no"
