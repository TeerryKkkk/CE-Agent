from __future__ import annotations

import inspect
from pathlib import Path

import src.raw_ce_candidate_pilot as pilot


def _module_source() -> str:
    return Path("src/raw_ce_candidate_pilot.py").read_text(encoding="utf-8")


def test_construction_phase_does_not_reference_dter_fields() -> None:
    source = inspect.getsource(pilot.construct_raw_candidates).lower()

    assert "dter" not in source
    assert "dtp_event_id" not in source
    assert "rain_event_id" not in source


def test_no_network_client_calls_are_present_in_raw_pilot_module() -> None:
    source = _module_source().lower()
    forbidden_tokens = [
        "requests.",
        "httpx.",
        "urllib.request",
        "urlopen",
        "socket.",
        "tavily",
        "bocha",
        "openai",
    ]

    for token in forbidden_tokens:
        assert token not in source


def test_p99_rainfall_is_not_named_as_a_public_impact_event() -> None:
    source = _module_source().lower()
    forbidden_word = "fl" + "ood"

    assert forbidden_word not in source
    assert "p99 extreme-rainfall" in source


def test_v2_labels_and_schema_are_not_referenced_or_modified() -> None:
    source = _module_source()
    forbidden_labels = [
        "confirmed_" + "compound_event",
        "confirmed_" + "disaster",
        "drought_" + "caused_" + "flood",
        "false " + "positive",
        "false " + "negative",
    ]

    assert "ce_label_schema" not in source
    for label in forbidden_labels:
        assert label not in source


def test_old_california_prototype_module_is_not_used() -> None:
    source = _module_source()

    assert "us_real_slice_builder" not in source
    assert "California" not in source
