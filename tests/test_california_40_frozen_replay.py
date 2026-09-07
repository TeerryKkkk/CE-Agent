from __future__ import annotations

import inspect
import json
from pathlib import Path

from scripts import replay as replay


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "examples" / "california40"


def test_frozen_replay_matches_canonical_snapshot() -> None:
    replay.verify_replay_input_hashes(SNAPSHOT)
    pages, _, cases = replay.replay(SNAPSHOT)
    replay.verify_expected(SNAPSHOT, pages, cases)
    assert replay.summary(pages, cases)["page_result_counts"] == {
        "does_not_support": 65,
        "insufficient_source_content": 7,
        "supports": 6,
        "unresolved": 32,
    }


def test_frozen_acceptance_metrics_match_declared_baseline() -> None:
    baseline = json.loads((SNAPSHOT / "acceptance_baseline.json").read_text(encoding="utf-8-sig"))
    assert baseline["page_support_precision"] == 1.0
    assert baseline["page_support_recall"] == 0.75
    assert baseline["false_support_page_ids"] == []
    assert baseline["unsafe_local_positive_upgrade_page_ids"] == []
    assert baseline["historical_unsafe_support_returned_page_ids"] == []
    assert baseline["known_aggregation_errors_fixed"] == 6
    assert baseline["known_aggregation_error_count"] == 6
    assert baseline["case_full_field_agreement_count"] == 28
    assert baseline["case_final_tier_agreement_count"] == 37


def test_replay_is_zero_api_and_has_no_audit_oracle_dependency() -> None:
    source = inspect.getsource(replay).lower()
    assert "external_api_calls\": 0" in source
    forbidden = (
        "page_level_independent_audit",
        "case_level_independent_audit",
        "independent_page_judgment",
        "expected_answer",
        "apikey",
    )
    assert not any(value in source for value in forbidden)
