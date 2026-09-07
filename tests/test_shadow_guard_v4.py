from __future__ import annotations

import copy

import pytest

from climate_pipeline import full_body_direct_judge as direct


CANDIDATE = {
    "candidate_id": "shadow-fixture",
    "county": "Example County",
    "state": "California",
    "locality_hints": [],
    "wet_window_start": "2023-01-09",
    "wet_window_end": "2023-01-11",
    "target_hazards": ["rainfall", "flood"],
}


def date_item(
    *,
    role: str,
    precision: str,
    start: str,
    end: str,
    quote: str,
    anchor: dict | None = None,
    continuous: bool = False,
) -> dict:
    return {
        "semantic_role": role,
        "precision": precision,
        "normalized_start": start,
        "normalized_end": end,
        "continuous_event_period": continuous,
        "relative_anchor": anchor,
        "quote": quote,
    }


def judgment(*, dates: list[dict], recommendation: str = "supports", actuality: str = "observed") -> dict:
    return {
        "source_readability": "readable",
        "readability_reason": "Readable event report.",
        "event_identity": "one_uniquely_relevant_event",
        "candidate_event": {
            "event_summary": "Observed rain and flooding affected Example County during the selected event.",
            "binding_status": "explicit_same_event",
            "actuality": actuality,
            "date_relation": "aligned",
            "date_evidence": dates,
            "observed_status_as_of": None,
            "location_relation": "aligned",
            "location_evidence": [
                {
                    "location": "Example County, California",
                    "kind": "event_location",
                    "quote": "Heavy rain affected Example County.",
                }
            ],
            "hazard_status": "observed",
            "hazard_evidence": [
                {
                    "hazard": "heavy rain",
                    "actuality": "observed",
                    "quote": "Heavy rain affected Example County.",
                }
            ],
            "realized_impacts": [],
            "potential_impacts_or_risks": [],
            "asset_or_economic_values": [],
            "aid_or_administrative_actions": [],
            "observed_impact": "no",
            "attribution": "no",
            "attribution_quotes": [],
            "transition": "no",
            "transition_quotes": [],
            "supporting_quotes": ["Heavy rain affected Example County."],
        },
        "candidate_support_recommendation": recommendation,
        "uncertainty": "",
    }


def body_for(payload: dict) -> str:
    event = payload["candidate_event"]
    quotes = [
        item.get("quote", "")
        for key in (
            "date_evidence",
            "location_evidence",
            "hazard_evidence",
            "realized_impacts",
            "potential_impacts_or_risks",
            "aid_or_administrative_actions",
        )
        for item in event.get(key, [])
    ]
    quotes.extend(event.get("supporting_quotes", []))
    return " Event report. ".join(value for value in quotes if value) + " This is a complete readable report body."


def evaluate(payload: dict, *, shadow: bool, candidate: dict | None = None) -> dict:
    function = direct.derive_local_page_result_shadow_v4 if shadow else direct.derive_local_page_result
    return function(semantic=copy.deepcopy(payload), candidate=candidate or CANDIDATE, body_text=body_for(payload))


def test_shadow_accepts_bare_weekday_only_with_exact_same_page_anchor() -> None:
    anchor = {
        "anchor_role": "publication",
        "anchor_date": "2023-01-09",
        "anchor_quote": "Published Monday, January 9, 2023",
    }
    payload = judgment(
        dates=[
            date_item(
                role="publication",
                precision="exact_day",
                start="2023-01-09",
                end="2023-01-09",
                quote=anchor["anchor_quote"],
            ),
            date_item(
                role="event_day",
                precision="relative",
                start="2023-01-09",
                end="2023-01-09",
                quote="Heavy rain affected Example County early Monday.",
                anchor=anchor,
            ),
        ]
    )
    payload["candidate_event"]["hazard_evidence"][0]["quote"] = "Heavy rain affected Example County early Monday."
    payload["candidate_event"]["location_evidence"][0]["quote"] = "Heavy rain affected Example County early Monday."
    payload["candidate_event"]["supporting_quotes"] = ["Heavy rain affected Example County early Monday."]

    assert evaluate(payload, shadow=False)["page_result"] == "unresolved"
    assert evaluate(payload, shadow=True)["page_result"] == "supports"


def test_shadow_accepts_month_day_without_year_only_when_anchor_is_same_day() -> None:
    payload = judgment(
        dates=[
            date_item(
                role="event_day",
                precision="exact_day",
                start="2023-01-09",
                end="2023-01-09",
                quote="January 9th storm",
            ),
            date_item(
                role="update",
                precision="exact_day",
                start="2023-01-09",
                end="2023-01-09",
                quote="Updated January 9, 2023",
            ),
        ]
    )
    assert evaluate(payload, shadow=True)["page_result"] == "supports"

    payload["candidate_event"]["date_evidence"][1] = date_item(
        role="publication",
        precision="exact_day",
        start="2023-04-18",
        end="2023-04-18",
        quote="Published April 18, 2023",
    )
    assert evaluate(payload, shadow=True)["page_result"] == "unresolved"


def test_shadow_accepts_explicit_incident_range_without_continuity_claim() -> None:
    payload = judgment(
        dates=[
            date_item(
                role="incident_period",
                precision="exact_interval",
                start="2023-01-08",
                end="2023-01-09",
                quote="Developments on Jan. 8 and 9 are below.",
            ),
            date_item(
                role="update",
                precision="exact_day",
                start="2023-01-10",
                end="2023-01-10",
                quote="Updated January 10, 2023",
            ),
        ]
    )
    assert evaluate(payload, shadow=False)["page_result"] == "unresolved"
    assert evaluate(payload, shadow=True)["page_result"] == "supports"


def test_shadow_accepts_observed_status_date_but_not_publication_alone() -> None:
    payload = judgment(
        dates=[
            date_item(
                role="observed_status_as_of",
                precision="exact_day",
                start="2023-01-09",
                end="2023-01-09",
                quote="Observed status as of January 9, 2023",
            )
        ]
    )
    assert evaluate(payload, shadow=True)["page_result"] == "supports"
    payload["candidate_event"]["date_evidence"][0]["semantic_role"] = "publication"
    assert evaluate(payload, shadow=True)["page_result"] == "unresolved"


def test_shadow_extends_explicit_ongoing_event_start_only_to_grounded_anchor() -> None:
    payload = judgment(
        dates=[
            date_item(
                role="event_start",
                precision="exact_day",
                start="2022-12-26",
                end="",
                quote="On Dec. 26, 2022, the first in an ongoing series of storms affected Example County.",
            ),
            date_item(
                role="publication",
                precision="exact_day",
                start="2023-01-11",
                end="2023-01-11",
                quote="Published January 11, 2023",
            ),
        ]
    )
    assert evaluate(payload, shadow=False)["page_result"] == "does_not_support"
    assert evaluate(payload, shadow=True)["page_result"] == "supports"


def test_shadow_promotes_only_separable_observed_hazard_from_needs_review() -> None:
    anchor = {
        "anchor_role": "publication",
        "anchor_date": "2023-01-09",
        "anchor_quote": "Posted January 9, 2023",
    }
    observed_quote = "The advisory remains active this morning as heavy rain falls in Example County."
    payload = judgment(
        recommendation="needs_review",
        actuality="mixed_or_uncertain",
        dates=[
            date_item(
                role="warning_period",
                precision="relative",
                start="",
                end="2023-01-09",
                quote=observed_quote,
                anchor=anchor,
            ),
            date_item(
                role="publication",
                precision="exact_day",
                start="2023-01-09",
                end="2023-01-09",
                quote=anchor["anchor_quote"],
            ),
        ],
    )
    payload["candidate_event"]["hazard_evidence"][0]["quote"] = observed_quote
    payload["candidate_event"]["location_evidence"][0]["quote"] = observed_quote
    payload["candidate_event"]["supporting_quotes"] = [observed_quote]

    assert evaluate(payload, shadow=False)["page_result"] == "unresolved"
    shadow = evaluate(payload, shadow=True)
    assert shadow["page_result"] == "supports", shadow["guard_actions"]
    assert shadow["candidate_hazard_support"] == "yes"
    assert shadow["realized_impact_support"] == "no"

    prospective = copy.deepcopy(payload)
    prospective["candidate_event"]["hazard_evidence"][0]["actuality"] = "prospective"
    prospective["candidate_event"]["hazard_status"] = "anticipated_or_warning"
    assert evaluate(prospective, shadow=True)["page_result"] != "supports"


def test_shadow_closes_same_day_anchored_outside_event_but_not_coarse_date() -> None:
    outside_candidate = {**CANDIDATE, "wet_window_start": "2023-01-01", "wet_window_end": "2023-01-02"}
    payload = judgment(
        recommendation="does_not_support",
        dates=[
            date_item(
                role="event_day",
                precision="exact_day",
                start="2023-01-09",
                end="2023-01-09",
                quote="January 9th storm",
            ),
            date_item(
                role="update",
                precision="exact_day",
                start="2023-01-09",
                end="2023-01-09",
                quote="Updated January 9, 2023",
            ),
        ],
    )
    payload["candidate_event"]["date_relation"] = "outside_candidate"
    assert evaluate(payload, shadow=False, candidate=outside_candidate)["page_result"] == "unresolved"
    assert evaluate(payload, shadow=True, candidate=outside_candidate)["page_result"] == "does_not_support"

    payload["candidate_event"]["date_evidence"] = [
        date_item(
            role="incident_period",
            precision="month",
            start="2023-01-01",
            end="2023-01-31",
            quote="January 2023 storms",
        )
    ]
    assert evaluate(payload, shadow=True, candidate=outside_candidate)["page_result"] == "unresolved"


def test_unknown_guard_policy_version_fails_closed() -> None:
    payload = judgment(
        dates=[
            date_item(
                role="event_day",
                precision="exact_day",
                start="2023-01-09",
                end="2023-01-09",
                quote="January 9, 2023",
            )
        ]
    )
    with pytest.raises(ValueError, match="unsupported_guard_policy_version"):
        direct.derive_local_page_result(
            semantic=payload,
            candidate=CANDIDATE,
            body_text=body_for(payload),
            guard_policy_version="not-a-real-policy",
        )


def test_shadow_accepts_completed_full_body_negative_when_no_unique_supporting_event_exists() -> None:
    payload = judgment(recommendation="does_not_support", dates=[])
    payload["event_identity"] = "unresolved_event_identity"
    payload["candidate_event"] = None
    payload["uncertainty"] = "Multiple concrete events are outside the candidate window."
    body = "Multiple unrelated historical storm reports are outside the candidate window. " * 12
    assert direct.derive_local_page_result(
        semantic=copy.deepcopy(payload), candidate=CANDIDATE, body_text=body
    )["page_result"] == "unresolved"
    shadow = direct.derive_local_page_result_shadow_v4(
        semantic=copy.deepcopy(payload), candidate=CANDIDATE, body_text=body
    )
    assert shadow["page_result"] == "does_not_support"
    assert shadow["candidate_hazard_support"] == "no"


def test_shadow_distant_model_negative_anchor_cannot_override_near_candidate_date() -> None:
    candidate = {
        **CANDIDATE,
        "wet_window_start": "2025-04-02",
        "wet_window_end": "2025-04-03",
    }
    distant = judgment(
        recommendation="does_not_support",
        dates=[
            date_item(
                role="publication",
                precision="exact_day",
                start="2025-12-24",
                end="2025-12-24",
                quote="Published December 24, 2025",
            )
        ],
    )
    distant["candidate_event"]["date_relation"] = "outside_candidate"
    assert evaluate(distant, shadow=False, candidate=candidate)["page_result"] == "unresolved"
    assert evaluate(distant, shadow=True, candidate=candidate)["page_result"] == "does_not_support"

    near = judgment(
        recommendation="does_not_support",
        dates=[
            date_item(
                role="publication",
                precision="exact_day",
                start="2025-04-04",
                end="2025-04-04",
                quote="Published April 4, 2025",
            )
        ],
    )
    near["candidate_event"]["date_relation"] = "outside_candidate"
    assert evaluate(near, shadow=True, candidate=candidate)["page_result"] == "unresolved"
