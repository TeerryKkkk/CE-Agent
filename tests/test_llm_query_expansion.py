from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from climate_pipeline.ce_impact_labeling import derive_case_level_split_labels
from climate_pipeline.controlled_open_retrieval import TavilyCostController, initial_cost_ledger
from climate_pipeline.llm_evidence_judge import ModelCallResult
from climate_pipeline.llm_query_expansion import (
    FOUND_BY,
    LLMQueryExpansionConfig,
    annotate_expansion_report_with_outcomes,
    diagnose_expansion_gaps,
    plan_llm_query_expansion,
    validate_generated_query,
)
from climate_pipeline.web_reader import WebReaderResult
from scripts.retrieve import build_case_outputs, execute_open_query_rows, new_retrieval_state, web_evidence_rows_from_validation


class FakeGenerator:
    def __init__(self, queries: list[dict[str, str]]) -> None:
        self.queries = queries
        self.called = 0

    def generate_queries(self, payload):  # type: ignore[no-untyped-def]
        self.called += 1
        return ModelCallResult(
            role="llm_query_expansion",
            model="fake-model",
            status="ok",
            parsed={"queries": self.queries},
            raw_output_text="{}",
            raw_response=None,
            attempts=1,
            usage={},
        )


class RaisingGenerator:
    def generate_queries(self, payload):  # type: ignore[no-untyped-def]
        raise AssertionError("generator should not be called when no trigger fires")


def _case() -> dict[str, str]:
    return {
        "candidate_id": "rawce_kern_001",
        "phase2_3_sample_id": "rawce_kern_001",
        "raw_candidate_id": "rawce_kern_001",
        "county": "Kern County",
        "state": "California",
        "state_abbrev": "CA",
        "FIPS": "06029",
        "drought_start": "2022-05",
        "drought_end": "2022-12",
        "drought_end_month_end": "2022-12-31",
        "rain_start": "2023-03-10",
        "rain_end": "2023-03-12",
        "rainfall_year": "2023",
        "drought_window": "2022-05 to 2022-12-31",
        "wet_event_window": "2023-03-10 to 2023-03-12",
        "event_window": "drought 2022-05 to 2022-12-31; wet 2023-03-10 to 2023-03-12",
    }


def _accepted_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": "rawce_kern_001",
        "county": "Kern County",
        "final_status": "accepted",
        "location_match": "same_county",
        "time_match": "same_month",
        "component_supported": "none",
        "impact_supported": "false",
        "impact_types": "",
        "source_lane": "local_news",
        "source_family": "credible_local_news",
        "evidence_origin": "validated_open_web_body",
        "quoted_supporting_spans": "Kern County officials reported March 2023 flooding and storm damage.",
    }
    row.update(overrides)
    return row


def _complete_evidence() -> list[dict[str, object]]:
    return [
        _accepted_row(component_supported="drought", source_lane="usdm_county_statistics", source_family="usdm_structured", evidence_origin="structured_official_reference"),
        _accepted_row(component_supported="wet", source_lane="noaa_storm_events", source_family="noaa_ncei_storm_events_structured", evidence_origin="structured_official_reference"),
        _accepted_row(
            component_supported="both",
            impact_supported="true",
            impact_types="property / housing; roads / transport",
            wet_impact_support="true",
            drought_impact_support="true",
            explicit_transition_support="true",
            quoted_supporting_spans="Kern County March 2023 flooding caused evacuation orders, property damage, and road closures after drought conditions.",
        ),
    ]


def test_expansion_disabled_returns_no_query_rows() -> None:
    plan = plan_llm_query_expansion(
        cases=[_case()],
        evidence_rows_by_case={"rawce_kern_001": []},
        existing_query_rows=[],
        expansion_config=LLMQueryExpansionConfig(enabled=False),
        run_id="test_run",
    )

    assert plan.query_rows == []
    assert plan.raw_model_outputs == []
    assert plan.report_rows[0]["validation_status"] == "disabled"


def test_expansion_enabled_but_no_trigger_does_not_call_generator() -> None:
    plan = plan_llm_query_expansion(
        cases=[_case()],
        evidence_rows_by_case={"rawce_kern_001": _complete_evidence()},
        existing_query_rows=[],
        expansion_config=LLMQueryExpansionConfig(enabled=True),
        run_id="test_run",
        generator=RaisingGenerator(),
    )

    assert plan.query_rows == []
    assert plan.report_rows[0]["validation_status"] == "no_trigger"


def test_wet_gap_triggers_only_bounded_hazard_eligibility() -> None:
    triggers = diagnose_expansion_gaps(
        _case(),
        [_accepted_row(component_supported="drought")],
        LLMQueryExpansionConfig(enabled=True, max_hazard_queries_per_case=1),
    )
    wet = [row for row in triggers if row.target_gap == "wet_hazard_public_corroboration"]
    assert len(wet) == 1
    assert wet[0].query_intent == "hazard_public_corroboration"
    assert wet[0].max_queries == 1


def test_drought_gap_triggers_only_bounded_hazard_eligibility() -> None:
    triggers = diagnose_expansion_gaps(
        _case(),
        [_accepted_row(component_supported="wet")],
        LLMQueryExpansionConfig(enabled=True, max_hazard_queries_per_case=1),
    )
    drought = [
        row
        for row in triggers
        if row.target_gap == "drought_hazard_public_corroboration"
    ]
    assert len(drought) == 1
    assert drought[0].query_intent == "hazard_public_corroboration"
    assert drought[0].max_queries == 1


def test_both_missing_hazard_legs_receive_independent_expansion_slots() -> None:
    triggers = diagnose_expansion_gaps(
        _case(),
        [],
        LLMQueryExpansionConfig(
            enabled=True,
            max_queries_per_case=4,
            max_hazard_queries_per_case=2,
            max_impact_queries_per_case=0,
        ),
    )
    assert {
        row.target_gap
        for row in triggers
        if row.query_intent == "hazard_public_corroboration"
    } == {
        "drought_hazard_public_corroboration",
        "wet_hazard_public_corroboration",
    }


def test_realized_impact_gap_triggers_impact_expansion() -> None:
    triggers = diagnose_expansion_gaps(
        _case(),
        [_accepted_row(component_supported="drought"), _accepted_row(component_supported="wet")],
        LLMQueryExpansionConfig(enabled=True, max_impact_queries_per_case=3),
    )
    impact = [row for row in triggers if row.target_gap == "impact_enrichment"]
    assert len(impact) == 1
    assert impact[0].query_intent == "impact_enrichment"
    assert impact[0].max_queries == 3


def test_linkage_expansion_requires_separately_supported_drought_and_wet() -> None:
    config = LLMQueryExpansionConfig(enabled=True, max_queries_per_case=4, max_impact_queries_per_case=0)
    only_drought = diagnose_expansion_gaps(_case(), [_accepted_row(component_supported="drought")], config)
    both = diagnose_expansion_gaps(
        _case(),
        [_accepted_row(component_supported="drought"), _accepted_row(component_supported="wet")],
        config,
    )
    assert not any(row.target_gap == "explicit_linkage_absent" for row in only_drought)
    assert any(row.target_gap == "explicit_linkage_absent" for row in both)


def test_expansion_query_caps_are_independent_and_total_bounded() -> None:
    triggers = diagnose_expansion_gaps(
        _case(),
        [],
        LLMQueryExpansionConfig(
            enabled=True,
            max_queries_per_case=4,
            max_hazard_queries_per_case=2,
            max_impact_queries_per_case=3,
            max_linkage_queries_per_case=1,
            max_pages_per_case=3,
        ),
    )
    assert sum(row.max_queries for row in triggers) == 4
    assert sum(row.max_queries for row in triggers if row.query_intent == "hazard_public_corroboration") == 2
    assert sum(row.max_queries for row in triggers if row.query_intent == "impact_enrichment") == 2


def test_invalid_generated_queries_are_rejected_before_search() -> None:
    generator = FakeGenerator(
        [
            {
                "query_text": "California flood damage 2023",
                "intended_purpose": "impact_enrichment",
                "target_gap": "impact_enrichment",
                "location_anchor_used": "California",
                "time_anchor_used": "2023",
                "hazard_or_impact_anchor_used": "flood damage",
                "risk_note": "too broad",
            },
            {
                "query_text": "Kern County Highway 58 washed out March 2023",
                "intended_purpose": "impact_enrichment",
                "target_gap": "impact_enrichment",
                "location_anchor_used": "Kern County",
                "time_anchor_used": "March 2023",
                "hazard_or_impact_anchor_used": "washed out",
                "risk_note": "ungrounded road name",
            },
            {
                "query_text": "SPEI DTER compound event rawce_kern_001 Kern County 2023",
                "intended_purpose": "impact_enrichment",
                "target_gap": "impact_enrichment",
                "location_anchor_used": "Kern County",
                "time_anchor_used": "2023",
                "hazard_or_impact_anchor_used": "compound event",
                "risk_note": "research-only terms",
            },
            {
                "query_text": "\"Kern County\" \"March 2023\" flooding road closure",
                "intended_purpose": "impact_enrichment",
                "target_gap": "impact_enrichment",
                "location_anchor_used": "Kern County",
                "time_anchor_used": "March 2023",
                "hazard_or_impact_anchor_used": "flooding road closure",
                "risk_note": "bounded local impact query",
            },
        ]
    )

    plan = plan_llm_query_expansion(
        cases=[_case()],
        evidence_rows_by_case={"rawce_kern_001": [_accepted_row(component_supported="drought"), _accepted_row(component_supported="wet")]},
        existing_query_rows=[],
        expansion_config=LLMQueryExpansionConfig(enabled=True, max_queries_per_case=4, max_impact_queries_per_case=4),
        run_id="test_run",
        generator=generator,
    )

    assert len(plan.query_rows) == 1
    assert plan.query_rows[0]["found_by"] == FOUND_BY
    assert plan.query_rows[0]["query_source"] == "llm_expansion"
    rejected = [row for row in plan.report_rows if row["validation_status"] == "rejected"]
    assert len(rejected) == 3
    assert any("broad_state_only_query" in row["validation_reasons"] for row in rejected)
    assert any("ungrounded_proper_nouns" in row["validation_reasons"] for row in rejected)
    assert any("internal_research_terms" in row["validation_reasons"] for row in rejected)


def test_generated_queries_alone_do_not_change_case_labels() -> None:
    case = _case()
    rows = [_accepted_row(component_supported="drought"), _accepted_row(component_supported="wet")]
    before = derive_case_level_split_labels(case, rows).as_dict()

    plan_llm_query_expansion(
        cases=[case],
        evidence_rows_by_case={"rawce_kern_001": rows},
        existing_query_rows=[],
        expansion_config=LLMQueryExpansionConfig(enabled=True),
        run_id="test_run",
        generator=FakeGenerator(
            [
                {
                    "query_text": "\"Kern County\" \"March 2023\" flooding road closure",
                    "intended_purpose": "impact_enrichment",
                    "target_gap": "impact_enrichment",
                    "location_anchor_used": "Kern County",
                    "time_anchor_used": "March 2023",
                    "hazard_or_impact_anchor_used": "flooding road closure",
                    "risk_note": "bounded local impact query",
                }
            ]
        ),
    )
    after = derive_case_level_split_labels(case, rows).as_dict()

    assert after == before


def test_report_marks_label_change_supported_only_by_accepted_expansion_evidence() -> None:
    report_rows = [
        {
            "candidate_id": "rawce_kern_001",
            "first_pass_ce_status": "",
            "first_pass_impact_status": "",
            "first_pass_case_use_label": "",
            "first_pass_material_impact_pattern": "",
        }
    ]

    annotate_expansion_report_with_outcomes(
        report_rows,
        first_pass_gate_rows=[{"candidate_id": "rawce_kern_001", "ce_status": "ce_partial", "impact_status": "impact_not_found", "case_use_label": "ce_without_public_impact"}],
        final_gate_rows=[{"candidate_id": "rawce_kern_001", "ce_status": "ce_supported", "impact_status": "impact_material", "case_use_label": "strong_ce_with_material_impact"}],
        expansion_pages=[{"candidate_id": "rawce_kern_001"}],
        expansion_evidence_rows=[_accepted_row(component_supported="wet", impact_supported="true", found_by=FOUND_BY)],
    )

    assert report_rows[0]["case_label_changed"] == "true"
    assert report_rows[0]["changed_label_supported_by_accepted_evidence"] == "true"


def test_expansion_page_validation_conversion_preserves_provenance() -> None:
    page = {
        "candidate_id": "rawce_kern_001",
        "source_url": "https://news.example/kern-flooding",
        "source_title": "Kern flooding road closure",
        "source_lane": "transportation_roads",
        "query_id": "q_expansion_1",
        "found_by": FOUND_BY,
        "query_source": "llm_expansion",
        "retrieval_round": "2",
        "target_gap": "impact_enrichment",
        "query_intent": "impact_enrichment",
        "expansion_trigger": "no_accepted_impact_evidence",
        "generated_query": "\"Kern County\" \"March 2023\" flooding road closure",
        "validated_query": "\"Kern County\" \"March 2023\" flooding road closure",
        "body_text_or_archived_body_text": "Kern County March 2023 flooding road closure damaged a local road.",
        "accepted_from_snippet": False,
    }
    result = SimpleNamespace(
        normalized_rows=[
            {
                "candidate_id": "rawce_kern_001",
                "source_url": page["source_url"],
                "primary_judgment": {
                    "explicit_transition_support": False,
                    "location_match": "same_county",
                    "time_match": "same_month",
                    "page_type": "news_impact_page",
                },
                "final_judgment": {
                    "final_status": "accepted",
                    "final_component_supported": "wet",
                    "final_impact_supported": True,
                    "final_impact_types": ["transportation"],
                    "final_quoted_spans": ["Kern County March 2023 flooding road closure damaged a local road."],
                    "failure_reason_if_rejected": "none",
                    "final_reason": "accepted",
                },
                "source_metadata": {"source_lane": "transportation_roads", "source_family": "credible_local_news", "query_id": "q_expansion_1"},
            }
        ]
    )

    rows = web_evidence_rows_from_validation(result, [page], {"rawce_kern_001": _case()})

    assert rows[0]["found_by"] == FOUND_BY
    assert rows[0]["query_source"] == "llm_expansion"
    assert rows[0]["retrieval_round"] == "2"
    assert rows[0]["final_status"] == "accepted"


def test_validator_rejects_duplicate_fixed_query() -> None:
    validation = validate_generated_query(
        query='"Kern County" "March 2023" flooding road closure',
        case=_case(),
        evidence_rows=[],
        existing_query_texts=['"Kern County" "March 2023" flooding road closure'],
    )

    assert validation.accepted is False
    assert "duplicate_or_near_duplicate_fixed_query" in validation.reasons



def test_validator_rejects_linkage_wording_without_linkage_trigger() -> None:
    validation = validate_generated_query(
        query='Kern County California drought to flood transition January 2023 linkage local news',
        case=_case(),
        evidence_rows=[],
        existing_query_texts=[],
        query_intent='impact_enrichment',
    )

    assert validation.accepted is False
    assert 'linkage_language_without_linkage_trigger' in ';'.join(validation.reasons)

    linkage_validation = validate_generated_query(
        query='Kern County California drought followed by flooding January 2023 local news',
        case=_case(),
        evidence_rows=[],
        existing_query_texts=[],
        query_intent='linkage_search',
    )

    assert linkage_validation.accepted is True


def test_hazard_corroboration_query_must_match_target_gap() -> None:
    validation = validate_generated_query(
        query='Kern County California January 2023 flooding road closure local news',
        case=_case(),
        evidence_rows=[],
        existing_query_texts=[],
        query_intent='hazard_public_corroboration',
        target_gap='drought_hazard_public_corroboration',
    )

    assert validation.accepted is False
    assert 'target_gap_anchor_mismatch:drought' in validation.reasons


class FakeTavilyClient:
    def search_raw(self, query_text, *, max_results, include_raw_content):  # type: ignore[no-untyped-def]
        county = 'Kern' if 'Kern' in query_text else 'Placer'
        url = f'https://local.example/{county.lower()}-storm-damage'
        return (
            [
                {
                    'url': url,
                    'title': f'{county} County March 2023 flooding road closure emergency management',
                    'content': f'{county} County California March 2023 flooding caused road closure and storm damage.',
                    'raw_content': f'{county} County California March 2023 flooding caused road closure and storm damage. ' * 3,
                }
            ],
            {},
        )


class FakeWebReader:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path

    def read(self, url: str, *, title_hint: str = '') -> WebReaderResult:
        cleaned = self.tmp_path / (url.rsplit('/', 1)[-1] + '.txt')
        cleaned.write_text('County California March 2023 flooding caused road closure and storm damage. ' * 3, encoding='utf-8')
        return WebReaderResult(
            original_url=url,
            final_url=url,
            fetch_method='fake_reader',
            http_status=200,
            raw_html_path='',
            cleaned_text_path=str(cleaned),
            raw_html_len=0,
            cleaned_text_len=cleaned.stat().st_size,
            content_hash='fake',
            encoding='utf-8',
            failure_reason=None,
            fallback_attempted=False,
            fallback_used=False,
            title=title_hint,
        )


def _expansion_query(case: dict[str, str]) -> dict[str, str]:
    query = f'{case["county"]} California March 2023 flooding road closure emergency management'
    return {
        'schema_version': 'llm_query_expansion_v1',
        'candidate_id': case['candidate_id'],
        'phase2_3_sample_id': case['candidate_id'],
        'lane': 'local_government_emergency',
        'missing_component': 'impact',
        'query_text': query,
        'round': '2',
        'retrieval_round': '2',
        'query_source': 'llm_expansion',
        'found_by': FOUND_BY,
        'search_backend': 'tavily',
        'query_id': f'q_{case["candidate_id"]}',
        'target_gap': 'impact_enrichment',
        'query_intent': 'impact_enrichment',
        'expansion_trigger': 'no_accepted_impact_evidence',
        'generated_query': query,
        'validated_query': query,
        'retrieval_tier': 'controlled_tavily_open_web',
        'fallback_reason': 'no_accepted_impact_evidence',
        'unmet_gap_before_search': 'impact_enrichment',
        'domains_or_source_families_targeted': 'local_government_emergency',
    }


def test_expansion_fetch_budget_independent_from_first_pass_global_count(tmp_path: Path) -> None:
    cases = [
        {**_case(), 'candidate_id': 'rawce_kern_001', 'phase2_3_sample_id': 'rawce_kern_001', 'raw_candidate_id': 'rawce_kern_001', 'county': 'Kern County', 'FIPS': '06029'},
        {**_case(), 'candidate_id': 'rawce_placer_001', 'phase2_3_sample_id': 'rawce_placer_001', 'raw_candidate_id': 'rawce_placer_001', 'county': 'Placer County', 'FIPS': '06061'},
    ]
    controls = TavilyCostController(
        max_results_per_query=1,
        max_open_fetches_per_candidate_round2=1,
        max_open_fetches_per_candidate_total=1,
        max_fetches_per_lane_per_candidate=1,
        max_open_fetches_total=2,
    )
    ledger = initial_cost_ledger(controls)
    state = new_retrieval_state()
    state['global_counts']['open_fetches_total'] = 2
    state['global_counts_by_round']['1'] = {'open_fetches_total': 2}

    _, _, fetch_rows, pages, _ = execute_open_query_rows(
        cases=cases,
        query_rows=[_expansion_query(case) for case in cases],
        controls=controls,
        tavily_client=FakeTavilyClient(),
        web_reader=FakeWebReader(tmp_path),
        ledger=ledger,
        round_number=2,
        retrieval_state=state,
    )

    assert len(pages) == 2
    assert {page['candidate_id'] for page in pages} == {'rawce_kern_001', 'rawce_placer_001'}
    assert all(page['query_source'] == 'llm_expansion' for page in pages)
    assert sum(1 for row in fetch_rows if row['fetch_decision'] == 'fetch') == 2



def test_non_accepted_expansion_rows_do_not_change_main_labels() -> None:
    case = {
        **_case(),
        "demo_group": "test",
        "FIPS": "06029",
        "rainfall_year": "2023",
        "spatial_match_level": "same_county",
        "lag_days": "30",
        "lag_bin": "short",
        "drought_severity": "D2",
        "min_spei": "-1.5",
        "rainfall_days": "3",
        "rainfall_grid_count": "1",
        "old_positive_label_if_any": "",
    }
    first_pass_rows = [
        _accepted_row(component_supported="drought", evidence_origin="structured_official_reference"),
        _accepted_row(component_supported="wet", evidence_origin="structured_official_reference"),
    ]
    expansion_context_row = {
        **_accepted_row(
            final_status="context_only",
            component_supported="none",
            impact_supported="false",
            impact_types="",
            found_by=FOUND_BY,
            query_source="llm_expansion",
            retrieval_round="2",
            evidence_origin="validated_open_web_body",
            location_match="mapped_city_or_place",
            time_match="year_only",
        ),
        "quoted_supporting_spans": "Kern County annual context but no accepted event impact.",
    }
    expansion_needs_review_row = {
        **_accepted_row(
            final_status="needs_review",
            component_supported="wet",
            impact_supported="true",
            impact_types="transportation",
            found_by=FOUND_BY,
            query_source="llm_expansion",
            retrieval_round="2",
            evidence_origin="validated_open_web_body",
            location_match="mapped_city_or_place",
            time_match="same_month",
        ),
        "quoted_supporting_spans": "Kern County possible flooding road closure needs review.",
    }

    registry_rows, gate_rows, _ = build_case_outputs(
        cases=[case],
        evidence_rows=first_pass_rows + [expansion_context_row, expansion_needs_review_row],
        official_attempts_by_case={case["candidate_id"]: "test"},
    )

    assert registry_rows[0]["current_impact_status"] == "impact_not_found"
    assert registry_rows[0]["current_case_use_label"] == "ce_without_public_impact"
    assert gate_rows[0]["impact_status"] == "impact_not_found"
