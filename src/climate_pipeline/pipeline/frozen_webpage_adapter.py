"""Read-only adapter for the frozen California 40-case webpage artifacts.

The adapter validates and maps already-frozen bodies, GPT-5.5 judgments,
deterministic guard outputs, and page axes.  It never retrieves a page, calls a
model, or reruns a semantic/local guard.
"""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schemas import (
    AGGREGATOR_VERSION_ENVELOPE,
    FROZEN_WEBPAGE_ADAPTER_VERSION,
    AxisCoverageState,
    AxisName,
    AxisValue,
    CanonicalSchemaError,
    CaseAxisRecord,
    EvidenceDisposition,
    FrozenWebpageExecutionDetails,
    JudgmentCoverageState,
    LaneExecutionRecord,
    LaneExecutionStatus,
    LaneReadiness,
    ReasonCode,
    RetrievalCoverageState,
    RuleId,
    SourceRole,
    SpatialAlignmentGrade,
    StructuredEvidenceRecord,
    TemporalAlignmentGrade,
    VersionLineage,
    WebpageCoverage,
    validate_unique_lane_execution,
)


EXPECTED_PAGE_COUNT = 110
EXPECTED_CASE_COUNT = 40
ZERO_PAGE_CANDIDATE_ID = "rawce_ca434870b4e3aee0"
FROZEN_WEBPAGE_LANE = "frozen_webpage"
ACTIVE_WEB_AXIS_SCOPE_POLICY_VERSION = "web_axis_scope_v1_production"
SHADOW_WEB_AXIS_SCOPE_POLICY_VERSION = "web_axis_scope_v2_validated"
WEB_AXIS_SCOPE_POLICY_VERSIONS = {
    ACTIVE_WEB_AXIS_SCOPE_POLICY_VERSION,
    SHADOW_WEB_AXIS_SCOPE_POLICY_VERSION,
}

REQUIRED_SNAPSHOT_FILES = (
    "candidate_manifest.csv",
    "final_110_candidate_page_results.csv",
    "final_110_page_guard_details.jsonl",
    "provenance_manifest.json",
    "replay_inputs/candidate_page_manifest.jsonl",
    "replay_inputs/final_semantic_judgments.jsonl",
    "replay_inputs/logical_frozen_pages.jsonl",
)

PAGE_AXIS_COLUMNS: Mapping[AxisName, str | None] = {
    AxisName.DROUGHT_CORROBORATION: "drought_hazard_support",
    AxisName.WET_HAZARD_CORROBORATION: "candidate_hazard_support",
    AxisName.REALIZED_IMPACT: "realized_impact_support",
    AxisName.HAZARD_TO_IMPACT_ATTRIBUTION: "attribution_support",
    AxisName.DROUGHT_TO_WET_LINKAGE: "transition_support",
}


class FrozenWebpageAdapterError(CanonicalSchemaError):
    """Raised when frozen identity or required fields cannot be proven."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def stable_json_sha256(value: Any) -> str:
    return _sha256_bytes(_stable_json_bytes(value))


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenWebpageAdapterError(f"invalid_json:{path}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise FrozenWebpageAdapterError(
                        f"jsonl_row_must_be_object:{path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenWebpageAdapterError(f"invalid_jsonl:{path}") from exc
    return rows


def _load_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise FrozenWebpageAdapterError(f"invalid_csv:{path}") from exc


def _index_unique(
    rows: Iterable[Mapping[str, Any]],
    key: str,
    *,
    label: str,
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value:
            raise FrozenWebpageAdapterError(f"missing_identity:{label}:{key}")
        if value in indexed:
            raise FrozenWebpageAdapterError(f"duplicate_identity:{label}:{value}")
        indexed[value] = row
    return indexed


def _require_fields(row: Mapping[str, Any], names: Iterable[str], label: str) -> None:
    missing = [name for name in names if name not in row or row[name] is None]
    if missing:
        raise FrozenWebpageAdapterError(f"missing_required_fields:{label}:{missing}")


def _axis_value(raw: Any, *, page_id: str, field_name: str) -> AxisValue:
    try:
        return AxisValue(str(raw))
    except ValueError as exc:
        raise FrozenWebpageAdapterError(
            f"incompatible_page_axis_value:{page_id}:{field_name}:{raw}"
        ) from exc


@dataclass(frozen=True)
class FrozenWebpageAdapterResult:
    evidence_records: tuple[StructuredEvidenceRecord, ...]
    lane_execution_records: tuple[LaneExecutionRecord, ...]
    identity_summary: Mapping[str, Any]


@dataclass(frozen=True)
class _LoadedFrozenArtifacts:
    snapshot_dir: Path
    candidate_manifest: tuple[Mapping[str, str], ...]
    page_results: tuple[Mapping[str, str], ...]
    requests: Mapping[str, Mapping[str, Any]]
    logical_pages: Mapping[str, Mapping[str, Any]]
    judgments: Mapping[str, Mapping[str, Any]]
    guards: Mapping[str, Mapping[str, Any]]
    provenance: Mapping[str, Any]
    lineage_provenance: Mapping[str, Any]
    crosswalk: Mapping[str, Mapping[str, str]]
    source_file_hashes: Mapping[str, str]


def _verify_declared_package_hashes(snapshot_dir: Path, provenance: Mapping[str, Any]) -> None:
    declared = provenance.get("package_file_sha256")
    if not isinstance(declared, Mapping):
        raise FrozenWebpageAdapterError("missing_package_file_sha256_registry")
    for relative in REQUIRED_SNAPSHOT_FILES:
        if relative == "provenance_manifest.json":
            continue
        path = snapshot_dir / relative
        if not path.is_file():
            raise FrozenWebpageAdapterError(f"missing_frozen_artifact:{relative}")
        expected = declared.get(relative)
        if not isinstance(expected, str):
            raise FrozenWebpageAdapterError(f"missing_declared_hash:{relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise FrozenWebpageAdapterError(
                f"frozen_artifact_hash_mismatch:{relative}:{actual}"
            )


def _default_lineage_paths(snapshot_dir: Path) -> tuple[Path, Path]:
    repository_root = snapshot_dir.parents[2]
    lineage_root = repository_root / "dataset" / "mainline" / "california" / "lineage_v1"
    return (
        lineage_root / "lineage_provenance.json",
        lineage_root / "california_408_40_368_crosswalk.csv",
    )


def _load_frozen_artifacts(
    snapshot_dir: Path,
    *,
    lineage_provenance_path: Path | None = None,
    crosswalk_path: Path | None = None,
) -> _LoadedFrozenArtifacts:
    snapshot_dir = snapshot_dir.resolve()
    for relative in REQUIRED_SNAPSHOT_FILES:
        if not (snapshot_dir / relative).is_file():
            raise FrozenWebpageAdapterError(f"missing_frozen_artifact:{relative}")

    provenance = _load_json(snapshot_dir / "provenance_manifest.json")
    _verify_declared_package_hashes(snapshot_dir, provenance)

    default_lineage, default_crosswalk = _default_lineage_paths(snapshot_dir)
    lineage_path = (lineage_provenance_path or default_lineage).resolve()
    crosswalk_file = (crosswalk_path or default_crosswalk).resolve()
    if not lineage_path.is_file() or not crosswalk_file.is_file():
        raise FrozenWebpageAdapterError("frozen_execution_lineage_unavailable")

    manifest_rows = _load_csv(snapshot_dir / "candidate_manifest.csv")
    page_rows = _load_csv(snapshot_dir / "final_110_candidate_page_results.csv")
    request_rows = _load_jsonl(snapshot_dir / "replay_inputs" / "candidate_page_manifest.jsonl")
    logical_rows = _load_jsonl(snapshot_dir / "replay_inputs" / "logical_frozen_pages.jsonl")
    judgment_rows = _load_jsonl(snapshot_dir / "replay_inputs" / "final_semantic_judgments.jsonl")
    guard_rows = _load_jsonl(snapshot_dir / "final_110_page_guard_details.jsonl")
    crosswalk_rows = _load_csv(crosswalk_file)

    manifest = _index_unique(manifest_rows, "candidate_id", label="candidate_manifest")
    page_results = _index_unique(page_rows, "page_id", label="page_results")
    requests = _index_unique(request_rows, "page_id", label="page_requests")
    logical_pages = _index_unique(logical_rows, "logical_page_id", label="logical_pages")
    judgments = _index_unique(judgment_rows, "page_id", label="judgments")
    guards = _index_unique(guard_rows, "page_id", label="guards")
    crosswalk = _index_unique(crosswalk_rows, "candidate_id", label="crosswalk")

    if len(manifest) != EXPECTED_CASE_COUNT:
        raise FrozenWebpageAdapterError(f"expected_40_candidates:{len(manifest)}")
    page_maps = (page_results, requests, logical_pages, judgments, guards)
    if any(len(item) != EXPECTED_PAGE_COUNT for item in page_maps):
        raise FrozenWebpageAdapterError(
            "expected_110_pages_across_result_body_judgment_guard_artifacts"
        )
    page_ids = set(page_results)
    if any(set(item) != page_ids for item in page_maps[1:]):
        raise FrozenWebpageAdapterError("candidate_page_artifact_identity_sets_differ")
    if any(str(row.get("candidate_id") or "") not in manifest for row in page_results.values()):
        raise FrozenWebpageAdapterError("page_candidate_not_in_frozen_manifest")

    lineage = _load_json(lineage_path)
    if lineage.get("counts", {}).get("completed") != EXPECTED_CASE_COUNT:
        raise FrozenWebpageAdapterError("lineage_does_not_prove_40_completed_candidates")
    declared_page_hash = lineage.get("source_hashes", {}).get("frozen_page_results_sha256")
    actual_page_hash = sha256_file(snapshot_dir / "final_110_candidate_page_results.csv")
    if declared_page_hash != actual_page_hash:
        raise FrozenWebpageAdapterError("lineage_frozen_page_hash_mismatch")
    if lineage.get("zero_page_completed_ids") != [ZERO_PAGE_CANDIDATE_ID]:
        raise FrozenWebpageAdapterError("lineage_zero_page_identity_mismatch")

    manifest_ids = set(manifest)
    completed_crosswalk = {
        candidate_id
        for candidate_id, row in crosswalk.items()
        if str(row.get("in_completed_40") or "").lower() == "true"
    }
    if completed_crosswalk != manifest_ids:
        raise FrozenWebpageAdapterError("crosswalk_completed_candidate_set_mismatch")
    observed_counts: dict[str, int] = defaultdict(int)
    for row in page_results.values():
        observed_counts[str(row["candidate_id"])] += 1
    for candidate_id in manifest_ids:
        row = crosswalk[candidate_id]
        if int(row.get("frozen_page_count") or -1) != observed_counts[candidate_id]:
            raise FrozenWebpageAdapterError(f"crosswalk_page_count_mismatch:{candidate_id}")
    if str(crosswalk[ZERO_PAGE_CANDIDATE_ID].get("zero_page_candidate_retained") or "").lower() != "true":
        raise FrozenWebpageAdapterError("zero_page_retention_not_proven")

    source_file_hashes = {
        relative: sha256_file(snapshot_dir / relative)
        for relative in REQUIRED_SNAPSHOT_FILES
    }
    source_file_hashes["lineage_provenance.json"] = sha256_file(lineage_path)
    source_file_hashes["california_408_40_368_crosswalk.csv"] = sha256_file(crosswalk_file)

    return _LoadedFrozenArtifacts(
        snapshot_dir=snapshot_dir,
        candidate_manifest=tuple(manifest_rows),
        page_results=tuple(page_rows),
        requests=requests,
        logical_pages=logical_pages,
        judgments=judgments,
        guards=guards,
        provenance=provenance,
        lineage_provenance=lineage,
        crosswalk=crosswalk,
        source_file_hashes=source_file_hashes,
    )


def _page_identity(
    page_result: Mapping[str, Any],
    request: Mapping[str, Any],
    logical_page: Mapping[str, Any],
    judgment: Mapping[str, Any],
    guard: Mapping[str, Any],
) -> dict[str, Any]:
    page_id = str(page_result["page_id"])
    candidate_id = str(page_result["candidate_id"])
    source_url = str(page_result["source_url"])
    for label, row, row_page_key in (
        ("request", request, "page_id"),
        ("logical_page", logical_page, "logical_page_id"),
        ("judgment", judgment, "page_id"),
        ("guard", guard, "page_id"),
    ):
        if str(row.get(row_page_key) or "") != page_id:
            raise FrozenWebpageAdapterError(f"page_id_mismatch:{label}:{page_id}")
    for label, row in (("request", request), ("logical_page", logical_page), ("guard", guard)):
        if str(row.get("candidate_id") or "") != candidate_id:
            raise FrozenWebpageAdapterError(f"candidate_id_mismatch:{label}:{page_id}")
    for label, row in (("request", request), ("logical_page", logical_page), ("guard", guard)):
        if str(row.get("source_url") or "") != source_url:
            raise FrozenWebpageAdapterError(f"source_url_mismatch:{label}:{page_id}")

    _require_fields(
        page_result,
        (
            "semantic_provenance",
            "terminal_state",
            "model_support_recommendation",
            "page_result",
            "candidate_hazard_support",
            "realized_impact_support",
            "attribution_support",
            "transition_support",
            "previous_page_result",
        ),
        page_id,
    )
    _require_fields(request, ("body_sha256", "candidate", "readability_usable"), page_id)
    _require_fields(
        logical_page,
        ("body_text_or_archived_body_text", "logical_body_sha256", "fetch_status"),
        page_id,
    )
    _require_fields(judgment, ("terminal_state", "semantic_provenance"), page_id)
    if "parsed_semantic_result" not in judgment:
        raise FrozenWebpageAdapterError(f"missing_required_fields:{page_id}:parsed_semantic_result")
    if judgment["terminal_state"] == "ok" and judgment["parsed_semantic_result"] is None:
        raise FrozenWebpageAdapterError(f"completed_judgment_missing_result:{page_id}")
    _require_fields(guard, ("terminal_state", "guard_actions", "source_event_axes"), page_id)

    body = str(logical_page["body_text_or_archived_body_text"])
    body_sha256 = _sha256_bytes(body.encode("utf-8"))
    if body_sha256 != request["body_sha256"] or body_sha256 != logical_page["logical_body_sha256"]:
        raise FrozenWebpageAdapterError(f"frozen_body_hash_mismatch:{page_id}")
    if page_result["terminal_state"] != judgment["terminal_state"]:
        raise FrozenWebpageAdapterError(f"terminal_state_mismatch:{page_id}")
    if page_result["semantic_provenance"] != judgment["semantic_provenance"]:
        raise FrozenWebpageAdapterError(f"semantic_provenance_mismatch:{page_id}")

    guard_input = {
        "candidate": request["candidate"],
        "body_sha256": body_sha256,
        "semantic": judgment["parsed_semantic_result"],
        "call_status": judgment["terminal_state"],
    }
    page_axis_values = {
        "drought_corroboration": None,
        "wet_hazard_corroboration": page_result["candidate_hazard_support"],
        "realized_impact": page_result["realized_impact_support"],
        "hazard_to_impact_attribution": page_result["attribution_support"],
        "drought_to_wet_linkage": page_result["transition_support"],
    }
    return {
        "candidate_id": candidate_id,
        "page_id": page_id,
        "source_url": source_url,
        "body_sha256": body_sha256,
        "judgment_sha256": stable_json_sha256(judgment),
        "guard_input_sha256": stable_json_sha256(guard_input),
        "guard_output_sha256": stable_json_sha256(guard),
        "page_result_sha256": stable_json_sha256(dict(page_result)),
        "page_axis_values": page_axis_values,
    }


def _identity_summary(
    artifacts: _LoadedFrozenArtifacts,
    identities: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    sorted_identities = sorted(identities, key=lambda item: str(item["page_id"]))
    candidate_counts: dict[str, int] = defaultdict(int)
    for item in sorted_identities:
        candidate_counts[str(item["candidate_id"])] += 1
    for row in artifacts.candidate_manifest:
        candidate_counts.setdefault(str(row["candidate_id"]), 0)
    return {
        "identity_schema_version": "ce_agent_frozen_webpage_identity_v1",
        "scientific_contract_version": VersionLineage().scientific_contract_version,
        "schema_version": VersionLineage().schema_version,
        "adapter_version": FROZEN_WEBPAGE_ADAPTER_VERSION,
        "snapshot_package_id": artifacts.snapshot_dir.name,
        "logical_page_count": len(sorted_identities),
        "candidate_count": len(artifacts.candidate_manifest),
        "zero_page_candidate_id": ZERO_PAGE_CANDIDATE_ID,
        "zero_page_candidate_page_count": candidate_counts[ZERO_PAGE_CANDIDATE_ID],
        "source_file_sha256": dict(sorted(artifacts.source_file_hashes.items())),
        "page_identity_bundle_sha256": stable_json_sha256(sorted_identities),
        "candidate_page_count_bundle_sha256": stable_json_sha256(dict(sorted(candidate_counts.items()))),
        "frozen_manifest_sha256": artifacts.source_file_hashes["candidate_manifest.csv"],
        "page_result_bundle_sha256": artifacts.source_file_hashes[
            "final_110_candidate_page_results.csv"
        ],
        "body_bundle_sha256": artifacts.source_file_hashes[
            "replay_inputs/logical_frozen_pages.jsonl"
        ],
        "judgment_bundle_sha256": artifacts.source_file_hashes[
            "replay_inputs/final_semantic_judgments.jsonl"
        ],
        "guard_bundle_sha256": artifacts.source_file_hashes[
            "final_110_page_guard_details.jsonl"
        ],
        "identity_fields": [
            "candidate_id",
            "page_id",
            "source_url",
            "body_sha256",
            "judgment_sha256",
            "guard_input_sha256",
            "guard_output_sha256",
            "page_result_sha256",
            "page_axis_values"
        ]
    }


def _verify_identity_baseline(actual: Mapping[str, Any], baseline_path: Path) -> None:
    baseline = _load_json(baseline_path)
    required = (
        "identity_schema_version",
        "logical_page_count",
        "candidate_count",
        "zero_page_candidate_id",
        "zero_page_candidate_page_count",
        "source_file_sha256",
        "page_identity_bundle_sha256",
        "candidate_page_count_bundle_sha256",
        "frozen_manifest_sha256",
        "page_result_bundle_sha256",
        "body_bundle_sha256",
        "judgment_bundle_sha256",
        "guard_bundle_sha256",
    )
    for key in required:
        if baseline.get(key) != actual.get(key):
            raise FrozenWebpageAdapterError(f"webpage_identity_baseline_mismatch:{key}")


def _evidence_record_from_page(
    artifacts: _LoadedFrozenArtifacts,
    page_result: Mapping[str, str],
    identity: Mapping[str, Any],
) -> StructuredEvidenceRecord:
    page_id = str(page_result["page_id"])
    request = artifacts.requests[page_id]
    logical_page = artifacts.logical_pages[page_id]
    judgment = artifacts.judgments[page_id]
    guard = artifacts.guards[page_id]
    result = str(page_result["page_result"])
    disposition_by_result = {
        "supports": EvidenceDisposition.ACCEPTED,
        "does_not_support": EvidenceDisposition.REJECTED_NON_SUPPORT,
        "unresolved": EvidenceDisposition.UNRESOLVED,
        "insufficient_source_content": EvidenceDisposition.UNRESOLVED,
    }
    try:
        disposition = disposition_by_result[result]
    except KeyError as exc:
        raise FrozenWebpageAdapterError(f"unsupported_frozen_page_result:{page_id}:{result}") from exc

    assessments: dict[AxisName, AxisValue] = {}
    for axis_name, column in PAGE_AXIS_COLUMNS.items():
        if column is None:
            continue
        assessments[axis_name] = _axis_value(
            page_result.get(column, "unresolved"),
            page_id=page_id,
            field_name=column,
        )

    reason_codes = [
        ReasonCode.LEGACY_PAGE_AXIS_NOT_PRESENT,
        ReasonCode.LEGACY_ALIGNMENT_GRADE_NOT_PRESENT,
    ]
    if disposition is EvidenceDisposition.ACCEPTED:
        reason_codes.append(ReasonCode.ACCEPTED_MINIMUM_SCIENTIFIC_STANDARD)
    elif result == "does_not_support":
        reason_codes.append(ReasonCode.NO_SUPPORT_IN_SUCCESSFULLY_EVALUATED_SCOPE)
    elif result == "insufficient_source_content":
        reason_codes.append(ReasonCode.SOURCE_CONTENT_INSUFFICIENT)
    else:
        reason_codes.append(ReasonCode.CANDIDATE_RELEVANT_UNRESOLVED)

    raw_value_provenance = {
        "adapter_action": "identity_preserving_mapping_only",
        "original_page_result": dict(page_result),
        "frozen_body_identity": {
            "logical_page_id": logical_page["logical_page_id"],
            "logical_body_sha256": identity["body_sha256"],
            "body_text_source": logical_page.get("body_text_source", ""),
            "fetch_status": logical_page["fetch_status"],
        },
        "gpt55_judgment": judgment,
        "guard_input_sha256": identity["guard_input_sha256"],
        "guard_output": guard,
        "request_identity": {
            "request_sha256": request.get("request_sha256", ""),
            "readability_usable": request["readability_usable"],
            "readability_reason": request.get("readability_reason", ""),
        },
        "identity": dict(identity),
    }
    record_hash_payload = {
        "candidate_id": page_result["candidate_id"],
        "page_id": page_id,
        "source_url": page_result["source_url"],
        "disposition": disposition.value,
        "axis_assessments": {key.value: value.value for key, value in assessments.items()},
        "identity": dict(identity),
    }
    return StructuredEvidenceRecord(
        candidate_id=str(page_result["candidate_id"]),
        lane=FROZEN_WEBPAGE_LANE,
        package_id=artifacts.snapshot_dir.name,
        source_role=SourceRole.FROZEN_WEBPAGE_RECORD,
        disposition=disposition,
        source_record_id=page_id,
        source_event_id="",
        source_file="replay_inputs/logical_frozen_pages.jsonl",
        source_row=str(logical_page.get("logical_ordinal") or request.get("ordinal") or ""),
        source_url=str(page_result["source_url"]),
        source_sha256=str(identity["body_sha256"]),
        source_version=str(page_result["semantic_provenance"]),
        source_date="",
        raw_value_provenance=raw_value_provenance,
        spatial_grade=SpatialAlignmentGrade.UNRESOLVED,
        temporal_grade=TemporalAlignmentGrade.UNRESOLVED,
        axis_assessments=assessments,
        rule_ids=(RuleId.FROZEN_WEB_PRESERVE,),
        reason_codes=tuple(reason_codes),
        record_hash=stable_json_sha256(record_hash_payload),
        lineage=VersionLineage(adapter_version=FROZEN_WEBPAGE_ADAPTER_VERSION),
    )


def _execution_records(
    artifacts: _LoadedFrozenArtifacts,
    evidence: Iterable[StructuredEvidenceRecord],
    identity_summary: Mapping[str, Any],
) -> tuple[LaneExecutionRecord, ...]:
    evidence_by_candidate: dict[str, list[StructuredEvidenceRecord]] = defaultdict(list)
    for record in evidence:
        evidence_by_candidate[record.candidate_id].append(record)

    provenance = artifacts.provenance
    started_at = str(provenance.get("generation_started_at_utc") or "")
    completed_at = str(provenance.get("final_replay_completed_at_utc") or "")
    package_sha256 = stable_json_sha256(identity_summary["source_file_sha256"])
    records: list[LaneExecutionRecord] = []

    for manifest_row in artifacts.candidate_manifest:
        candidate_id = str(manifest_row["candidate_id"])
        candidate_evidence = evidence_by_candidate[candidate_id]
        page_ids = {record.source_record_id for record in candidate_evidence}
        requests = [artifacts.requests[page_id] for page_id in page_ids]
        judgments = [artifacts.judgments[page_id] for page_id in page_ids]
        guards = [artifacts.guards[page_id] for page_id in page_ids]
        logical_count = len(candidate_evidence)
        readable_count = sum(bool(row["readability_usable"]) for row in requests)
        insufficient_count = sum(
            record.raw_value_provenance["original_page_result"]["page_result"]
            == "insufficient_source_content"
            for record in candidate_evidence
        )
        completed_judgment_count = sum(row["terminal_state"] == "ok" for row in judgments)
        guarded_count = sum(row["terminal_state"] == "ok" for row in guards)
        judgment_or_guard_failed = any(
            row["terminal_state"] not in {"ok", "not_requested_unusable_source"}
            for row in (*judgments, *guards)
        )
        unresolved_count = sum(
            record.raw_value_provenance["original_page_result"]["page_result"]
            == "unresolved"
            for record in candidate_evidence
        )
        zero_pages = logical_count == 0
        retrieval_performed = True

        if zero_pages:
            if candidate_id != ZERO_PAGE_CANDIDATE_ID:
                raise FrozenWebpageAdapterError(f"unexpected_zero_page_candidate:{candidate_id}")
            retrieval_coverage = RetrievalCoverageState.ZERO_PAGES_RETRIEVED
            judgment_coverage = JudgmentCoverageState.COMPLETE
            webpage_coverage = WebpageCoverage.ZERO_PAGES_RETRIEVED
            status = LaneExecutionStatus.UNAVAILABLE
            coverage = AxisCoverageState.PARTIAL
            derivation_rule = RuleId.WEB_AXIS_ZERO_PAGES
        elif judgment_or_guard_failed:
            retrieval_coverage = RetrievalCoverageState.COMPLETE_FOR_FROZEN_SCOPE
            judgment_coverage = JudgmentCoverageState.INCOMPLETE_OR_FAILED
            webpage_coverage = WebpageCoverage.JUDGMENT_INCOMPLETE_OR_FAILED
            status = LaneExecutionStatus.TERMINAL_FAILURE
            coverage = AxisCoverageState.FAILED
            derivation_rule = RuleId.WEB_AXIS_UNRESOLVED_DEPENDENCY
        elif insufficient_count or readable_count < logical_count:
            retrieval_coverage = RetrievalCoverageState.COMPLETE_FOR_FROZEN_SCOPE
            judgment_coverage = JudgmentCoverageState.SOURCE_CONTENT_INSUFFICIENT
            webpage_coverage = WebpageCoverage.SOURCE_CONTENT_INSUFFICIENT
            status = LaneExecutionStatus.COMPLETED_WITH_MATCHES
            coverage = AxisCoverageState.PARTIAL
            derivation_rule = RuleId.WEB_AXIS_UNRESOLVED_EVIDENCE
        else:
            retrieval_coverage = RetrievalCoverageState.COMPLETE_FOR_FROZEN_SCOPE
            judgment_coverage = JudgmentCoverageState.COMPLETE
            webpage_coverage = WebpageCoverage.BOUNDED_COMPLETE_FOR_FROZEN_SCOPE
            status = LaneExecutionStatus.COMPLETED_WITH_MATCHES
            coverage = AxisCoverageState.COMPLETE
            derivation_rule = RuleId.FROZEN_WEB_PRESERVE

        webpage = FrozenWebpageExecutionDetails(
            retrieval_performed=retrieval_performed,
            logical_page_count=logical_count,
            successfully_fetched_or_readable_count=readable_count,
            source_content_insufficient_count=insufficient_count,
            completed_judgment_count=completed_judgment_count,
            candidate_relevant_unresolved_count=unresolved_count,
            guarded_page_count=guarded_count,
            zero_pages_retrieved=zero_pages,
            retrieval_coverage_state=retrieval_coverage,
            judgment_coverage_state=judgment_coverage,
            webpage_coverage=webpage_coverage,
            frozen_manifest_hash=str(identity_summary["frozen_manifest_sha256"]),
            page_result_bundle_hash=str(identity_summary["page_result_bundle_sha256"]),
            body_bundle_hash=str(identity_summary["body_bundle_sha256"]),
            judgment_bundle_hash=str(identity_summary["judgment_bundle_sha256"]),
            guard_bundle_hash=str(identity_summary["guard_bundle_sha256"]),
            derivation_rule_id=derivation_rule,
        )
        records.append(
            LaneExecutionRecord(
                candidate_id=candidate_id,
                lane=FROZEN_WEBPAGE_LANE,
                enabled=True,
                required=True,
                source_mode="frozen_local_read_only",
                package_id=artifacts.snapshot_dir.name,
                package_sha256=package_sha256,
                readiness=LaneReadiness.READY,
                execution_status=status,
                coverage=coverage,
                records_examined=logical_count,
                matches_found=logical_count,
                accepted_count=sum(
                    record.disposition is EvidenceDisposition.ACCEPTED
                    for record in candidate_evidence
                ),
                contextual_count=sum(
                    record.disposition is EvidenceDisposition.CONTEXTUAL
                    for record in candidate_evidence
                ),
                unresolved_count=sum(
                    record.disposition is EvidenceDisposition.UNRESOLVED
                    for record in candidate_evidence
                ),
                started_at=started_at,
                completed_at=completed_at if status.is_successful_completion else "",
                checkpoint_id="",
                retry_count=0,
                attempt=1,
                webpage=webpage,
                lineage=VersionLineage(adapter_version=FROZEN_WEBPAGE_ADAPTER_VERSION),
            )
        )
    validate_unique_lane_execution(records)
    if len(records) != EXPECTED_CASE_COUNT:
        raise FrozenWebpageAdapterError(f"expected_40_execution_records:{len(records)}")
    return tuple(records)


def adapt_frozen_webpage_snapshot(
    snapshot_dir: Path,
    *,
    identity_baseline_path: Path | None = None,
    lineage_provenance_path: Path | None = None,
    crosswalk_path: Path | None = None,
) -> FrozenWebpageAdapterResult:
    """Validate and map the frozen snapshot without any external operation."""

    artifacts = _load_frozen_artifacts(
        snapshot_dir,
        lineage_provenance_path=lineage_provenance_path,
        crosswalk_path=crosswalk_path,
    )
    identities: list[dict[str, Any]] = []
    evidence: list[StructuredEvidenceRecord] = []
    for page_result in artifacts.page_results:
        page_id = str(page_result["page_id"])
        identity = _page_identity(
            page_result,
            artifacts.requests[page_id],
            artifacts.logical_pages[page_id],
            artifacts.judgments[page_id],
            artifacts.guards[page_id],
        )
        identities.append(identity)
        evidence.append(_evidence_record_from_page(artifacts, page_result, identity))

    identity_summary = _identity_summary(artifacts, identities)
    if identity_baseline_path is not None:
        _verify_identity_baseline(identity_summary, identity_baseline_path.resolve())
    execution = _execution_records(artifacts, evidence, identity_summary)
    if len(evidence) != EXPECTED_PAGE_COUNT:
        raise FrozenWebpageAdapterError(f"expected_110_evidence_records:{len(evidence)}")
    return FrozenWebpageAdapterResult(tuple(evidence), execution, identity_summary)


def evaluate_bounded_web_axis(
    *,
    axis_name: AxisName,
    evidence_records: Iterable[StructuredEvidenceRecord],
    execution_record: LaneExecutionRecord,
    scope_policy_version: str = ACTIVE_WEB_AXIS_SCOPE_POLICY_VERSION,
) -> CaseAxisRecord:
    """Apply only the approved bounded-web truth table for one candidate/axis."""

    if scope_policy_version not in WEB_AXIS_SCOPE_POLICY_VERSIONS:
        raise FrozenWebpageAdapterError(
            f"unsupported_web_axis_scope_policy:{scope_policy_version}"
        )
    shadow_axis_scope = scope_policy_version == SHADOW_WEB_AXIS_SCOPE_POLICY_VERSION
    records = tuple(evidence_records)
    if execution_record.lane != FROZEN_WEBPAGE_LANE or execution_record.webpage is None:
        raise FrozenWebpageAdapterError("bounded_web_requires_frozen_webpage_execution")
    if any(record.candidate_id != execution_record.candidate_id for record in records):
        raise FrozenWebpageAdapterError("bounded_web_candidate_identity_mismatch")
    accepted = tuple(
        record.source_record_id
        for record in records
        if record.disposition is EvidenceDisposition.ACCEPTED
        and record.axis_assessments.get(axis_name) is AxisValue.YES
    )
    contextual = tuple(
        record.source_record_id
        for record in records
        if record.disposition is EvidenceDisposition.CONTEXTUAL
    )
    unresolved = tuple(
        record.source_record_id
        for record in records
        if record.axis_assessments.get(axis_name) is AxisValue.UNRESOLVED
        or (
            not shadow_axis_scope
            and record.disposition is EvidenceDisposition.UNRESOLVED
        )
    )
    details = execution_record.webpage
    coverage = details.webpage_coverage
    axis_scope_complete = bool(
        shadow_axis_scope
        and details.logical_page_count > 0
        and details.retrieval_coverage_state
        is RetrievalCoverageState.COMPLETE_FOR_FROZEN_SCOPE
        and details.completed_judgment_count == details.logical_page_count
        and details.guarded_page_count == details.logical_page_count
        and details.source_content_insufficient_count == 0
        and not unresolved
    )
    reasons: tuple[ReasonCode, ...]
    negative_scope = ""

    if accepted:
        value = AxisValue.YES
        axis_coverage = (
            AxisCoverageState.COMPLETE
            if coverage is WebpageCoverage.BOUNDED_COMPLETE_FOR_FROZEN_SCOPE
            or axis_scope_complete
            else AxisCoverageState.PARTIAL
        )
        rule = RuleId.WEB_AXIS_YES
        reasons = ()
    elif details.zero_pages_retrieved:
        value = AxisValue.UNRESOLVED
        axis_coverage = AxisCoverageState.PARTIAL
        rule = RuleId.WEB_AXIS_ZERO_PAGES
        reasons = (ReasonCode.ZERO_PAGES_RETRIEVED,)
    elif coverage in {WebpageCoverage.NOT_RUN, WebpageCoverage.UNAVAILABLE}:
        value = AxisValue.UNRESOLVED
        axis_coverage = (
            AxisCoverageState.NOT_RUN
            if coverage is WebpageCoverage.NOT_RUN
            else AxisCoverageState.UNAVAILABLE
        )
        rule = RuleId.WEB_AXIS_NOT_RUN
        reasons = (
            ReasonCode.LANE_NOT_RUN
            if coverage is WebpageCoverage.NOT_RUN
            else ReasonCode.EXECUTION_NOT_PROVEN,
        )
    elif coverage in {
        WebpageCoverage.JUDGMENT_INCOMPLETE_OR_FAILED,
        WebpageCoverage.BOUNDED_PARTIAL,
    }:
        value = AxisValue.UNRESOLVED
        axis_coverage = AxisCoverageState.FAILED
        rule = RuleId.WEB_AXIS_UNRESOLVED_DEPENDENCY
        reasons = (ReasonCode.JUDGMENT_INCOMPLETE_OR_FAILED,)
    elif unresolved or details.source_content_insufficient_count:
        value = AxisValue.UNRESOLVED
        axis_coverage = (
            AxisCoverageState.COMPLETE
            if coverage is WebpageCoverage.BOUNDED_COMPLETE_FOR_FROZEN_SCOPE
            else AxisCoverageState.PARTIAL
        )
        rule = RuleId.WEB_AXIS_UNRESOLVED_EVIDENCE
        reasons = (
            ReasonCode.SOURCE_CONTENT_INSUFFICIENT
            if details.source_content_insufficient_count
            else ReasonCode.CANDIDATE_RELEVANT_UNRESOLVED,
        )
    elif coverage is WebpageCoverage.BOUNDED_COMPLETE_FOR_FROZEN_SCOPE or axis_scope_complete:
        value = AxisValue.NO
        axis_coverage = AxisCoverageState.COMPLETE
        rule = RuleId.WEB_AXIS_NO
        reasons = ()
        negative_scope = "bounded_evaluated_scope_only"
    else:
        value = AxisValue.UNRESOLVED
        axis_coverage = AxisCoverageState.PARTIAL
        rule = RuleId.WEB_AXIS_UNRESOLVED_DEPENDENCY
        reasons = (ReasonCode.EXECUTION_NOT_PROVEN,)

    return CaseAxisRecord(
        candidate_id=execution_record.candidate_id,
        axis_name=axis_name,
        axis_value=value,
        coverage_state=axis_coverage,
        official_supporting_record_ids=(),
        official_contextual_record_ids=(),
        webpage_supporting_record_ids=accepted,
        webpage_contextual_record_ids=contextual,
        strongest_spatial_grade=None,
        all_spatial_grades=tuple(
            dict.fromkeys(record.spatial_grade for record in records)
        ),
        strongest_temporal_grade=None,
        all_temporal_grades=tuple(
            dict.fromkeys(record.temporal_grade for record in records)
        ),
        decision_rule_id=rule,
        unresolved_reason_codes=reasons,
        review_reason_codes=(
            (ReasonCode.COVERAGE_WARNING,)
            if value is AxisValue.YES
            and coverage is not WebpageCoverage.BOUNDED_COMPLETE_FOR_FROZEN_SCOPE
            else ()
        ),
        source_package_ids=(execution_record.package_id,),
        negative_scope=negative_scope,
        lineage=VersionLineage(
            adapter_version=FROZEN_WEBPAGE_ADAPTER_VERSION,
            aggregator_version=AGGREGATOR_VERSION_ENVELOPE,
        ),
    )


__all__ = [
    "EXPECTED_CASE_COUNT",
    "EXPECTED_PAGE_COUNT",
    "FROZEN_WEBPAGE_LANE",
    "ACTIVE_WEB_AXIS_SCOPE_POLICY_VERSION",
    "SHADOW_WEB_AXIS_SCOPE_POLICY_VERSION",
    "ZERO_PAGE_CANDIDATE_ID",
    "FrozenWebpageAdapterError",
    "FrozenWebpageAdapterResult",
    "adapt_frozen_webpage_snapshot",
    "evaluate_bounded_web_axis",
    "sha256_file",
    "stable_json_sha256",
]
