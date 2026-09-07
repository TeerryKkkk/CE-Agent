"""Canonical repaired-v2 schemas for the CE-Agent evidence pipeline.

This module is the sole Python schema authority introduced by repair Step 1
(W007-W011).  It deliberately does not change runner or aggregation behavior.
The scientific meanings are owned by ``contracts/ce_agent_repair_v2.json``;
the enums and validators below are the executable encoding of that contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
import re
from typing import Any, Iterable, Mapping, TypeVar


SCIENTIFIC_CONTRACT_VERSION = "ce_agent_scientific_contract_v2.0.0"
SCHEMA_VERSION = "ce_agent_repaired_v2_schema_v1.0.0"
ADAPTER_VERSION_ENVELOPE = "ce_agent_repaired_v2_adapter_v1"
AGGREGATOR_VERSION_ENVELOPE = "ce_agent_repaired_v2_aggregator_v1"
FROZEN_WEBPAGE_ADAPTER_VERSION = f"{ADAPTER_VERSION_ENVELOPE}.frozen_webpage_readonly.1"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CanonicalSchemaError(ValueError):
    """Raised when a record would violate the repaired-v2 contract."""


class CanonicalEnum(str, Enum):
    """String enum with stable JSON serialization."""


class AxisName(CanonicalEnum):
    DROUGHT_CORROBORATION = "drought_corroboration"
    WET_HAZARD_CORROBORATION = "wet_hazard_corroboration"
    REALIZED_IMPACT = "realized_impact"
    HAZARD_TO_IMPACT_ATTRIBUTION = "hazard_to_impact_attribution"
    DROUGHT_TO_WET_LINKAGE = "drought_to_wet_linkage"


class AxisValue(CanonicalEnum):
    YES = "yes"
    NO = "no"
    UNRESOLVED = "unresolved"


class AxisCoverageState(CanonicalEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    NOT_RUN = "not_run"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class LaneExecutionStatus(CanonicalEnum):
    COMPLETED_WITH_MATCHES = "completed_with_matches"
    COMPLETED_ZERO_MATCH = "completed_zero_match"
    COMPLETED_CONTEXT_ONLY = "completed_context_only"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"
    RETRYABLE_FAILURE = "retryable_failure"
    TERMINAL_FAILURE = "terminal_failure"

    @property
    def is_successful_completion(self) -> bool:
        return self in {
            self.COMPLETED_WITH_MATCHES,
            self.COMPLETED_ZERO_MATCH,
            self.COMPLETED_CONTEXT_ONLY,
        }


class LaneReadiness(CanonicalEnum):
    READY = "ready"
    NOT_READY = "not_ready"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"


class EvidenceDisposition(CanonicalEnum):
    ACCEPTED = "accepted"
    REJECTED_NON_SUPPORT = "rejected_non_support"
    REJECTED_MISMATCH = "rejected_mismatch"
    CONTEXTUAL = "contextual"
    UNRESOLVED = "unresolved"


class SourceRole(CanonicalEnum):
    USDM_COUNTY_WEEK_DROUGHT_CONTEXT = "usdm_county_week_drought_context"
    DIRECT_PRECIPITATION_EVENT = "direct_precipitation_event"
    RAIN_INDUCED_HYDROLOGIC_EVENT = "rain_induced_hydrologic_event"
    CONDITIONAL_WET_CONSEQUENCE = "conditional_wet_consequence"
    COASTAL_OR_ZONE_EVENT = "coastal_or_zone_event"
    BROADER_STORM_CONTEXT = "broader_storm_context"
    IRRELEVANT_EVENT = "irrelevant_event"
    FROZEN_WEBPAGE_RECORD = "frozen_webpage_record"
    ADMINISTRATIVE_RESPONSE = "administrative_response"


class SpatialAlignmentGrade(CanonicalEnum):
    LOCAL = "local"
    COUNTY = "county"
    ZONE = "zone"
    COUNTY_CONTEXT = "county_context"
    CONTEXT = "context"
    MISMATCH = "mismatch"
    UNRESOLVED = "unresolved"


class TemporalAlignmentGrade(CanonicalEnum):
    CANDIDATE_WINDOW_EXACT_OVERLAP = "candidate_window_exact_overlap"
    EXACT_TIME_SAME_EVENT_EPISODE = "exact_time_same_event_episode"
    PREVIOUS_OR_SAME_TUESDAY = "previous_or_same_tuesday"
    NEAR_WINDOW_BROADER_STORM_SEQUENCE = "near_window_broader_storm_sequence"
    MISMATCH = "mismatch"
    UNRESOLVED = "unresolved"


class PhysicalCandidateStatus(CanonicalEnum):
    AUTHORITATIVE_PHYSICAL_CANDIDATE = "authoritative_physical_candidate"
    NOT_IN_AUTHORITATIVE_INVENTORY = "not_in_authoritative_inventory"
    UNRESOLVED = "unresolved"


class ComponentCorroborationView(CanonicalEnum):
    BOTH_COMPONENTS_CORROBORATED = "both_components_corroborated"
    DROUGHT_ONLY_CORROBORATED = "drought_only_corroborated"
    WET_ONLY_CORROBORATED = "wet_only_corroborated"
    NO_COMPONENT_CORROBORATION_IN_EVALUATED_SCOPE = (
        "no_component_corroboration_in_evaluated_scope"
    )
    COMPONENT_STATUS_UNRESOLVED = "component_status_unresolved"


class ImpactAttributionView(CanonicalEnum):
    NO_DOCUMENTED_IMPACT_IN_EVALUATED_SCOPE = "no_documented_impact_in_evaluated_scope"
    DOCUMENTED_IMPACT_WITHOUT_ATTRIBUTION = "documented_impact_without_attribution"
    DOCUMENTED_IMPACT_ATTRIBUTION_UNRESOLVED = "documented_impact_attribution_unresolved"
    EXPLICITLY_ATTRIBUTED_IMPACT = "explicitly_attributed_impact"
    IMPACT_STATUS_UNRESOLVED = "impact_status_unresolved"


class LinkageView(CanonicalEnum):
    EXPLICIT_LINKAGE_SUPPORTED = "explicit_linkage_supported"
    NO_EXPLICIT_LINKAGE_IN_EVALUATED_SCOPE = "no_explicit_linkage_in_evaluated_scope"
    LINKAGE_UNRESOLVED = "linkage_unresolved"


class NavigationSummary(CanonicalEnum):
    CANDIDATE_ONLY = "candidate_only"
    ONE_COMPONENT_CORROBORATED = "one_component_corroborated"
    BOTH_COMPONENTS_CORROBORATED = "both_components_corroborated"
    BOTH_COMPONENTS_WITH_DOCUMENTED_IMPACT = "both_components_with_documented_impact"
    REVIEW_REQUIRED = "review_required"


class WebpageCoverage(CanonicalEnum):
    BOUNDED_COMPLETE_FOR_FROZEN_SCOPE = "bounded_complete_for_frozen_scope"
    BOUNDED_PARTIAL = "bounded_partial"
    ZERO_PAGES_RETRIEVED = "zero_pages_retrieved"
    SOURCE_CONTENT_INSUFFICIENT = "source_content_insufficient"
    JUDGMENT_INCOMPLETE_OR_FAILED = "judgment_incomplete_or_failed"
    NOT_RUN = "not_run"
    UNAVAILABLE = "unavailable"


class RetrievalCoverageState(CanonicalEnum):
    COMPLETE_FOR_FROZEN_SCOPE = "complete_for_frozen_scope"
    PARTIAL = "partial"
    ZERO_PAGES_RETRIEVED = "zero_pages_retrieved"
    NOT_RUN = "not_run"
    UNAVAILABLE = "unavailable"


class JudgmentCoverageState(CanonicalEnum):
    COMPLETE = "complete"
    SOURCE_CONTENT_INSUFFICIENT = "source_content_insufficient"
    INCOMPLETE_OR_FAILED = "incomplete_or_failed"
    NOT_RUN = "not_run"
    UNAVAILABLE = "unavailable"


class ReasonCode(CanonicalEnum):
    ACCEPTED_MINIMUM_SCIENTIFIC_STANDARD = "accepted_minimum_scientific_standard"
    NO_SUPPORT_IN_SUCCESSFULLY_EVALUATED_SCOPE = "no_support_in_successfully_evaluated_scope"
    CANDIDATE_RELEVANT_UNRESOLVED = "candidate_relevant_unresolved"
    SOURCE_CONTENT_INSUFFICIENT = "source_content_insufficient"
    RETRIEVAL_INCOMPLETE_OR_FAILED = "retrieval_incomplete_or_failed"
    JUDGMENT_INCOMPLETE_OR_FAILED = "judgment_incomplete_or_failed"
    GUARD_INCOMPLETE_OR_FAILED = "guard_incomplete_or_failed"
    ZERO_PAGES_RETRIEVED = "zero_pages_retrieved"
    EXECUTION_NOT_PROVEN = "execution_not_proven"
    LANE_UNAVAILABLE = "lane_unavailable"
    LANE_NOT_RUN = "lane_not_run"
    SOURCE_MISSING = "source_missing"
    SOURCE_MALFORMED = "source_malformed"
    SOURCE_FAILURE = "source_failure"
    PLACE_TIME_ALIGNMENT_INSUFFICIENT = "place_time_alignment_insufficient"
    SPATIAL_MISMATCH = "spatial_mismatch"
    TEMPORAL_MISMATCH = "temporal_mismatch"
    CONTEXT_ONLY = "context_only"
    ADMINISTRATIVE_RESPONSE_ONLY = "administrative_response_only"
    EXPLICIT_RELATION_ABSENT = "explicit_relation_absent"
    LEGACY_PAGE_AXIS_NOT_PRESENT = "legacy_page_axis_not_present"
    LEGACY_ALIGNMENT_GRADE_NOT_PRESENT = "legacy_alignment_grade_not_present"
    REQUIRED_FIELD_MISSING = "required_field_missing"
    DERIVED_REVIEW_ROW = "derived_review_row"
    SYNTHETIC_EXECUTION_STATE = "synthetic_execution_state"
    COVERAGE_WARNING = "coverage_warning"


class RuleId(CanonicalEnum):
    PHYSICAL_CANDIDATE = "PHYS-CAND-001"
    AXIS_VALUE = "AXIS-VALUE-001"
    AXIS_NO = "AXIS-NO-001"
    AXIS_UNRESOLVED = "AXIS-UNRES-001"
    GRADE_SUPPORT_SEPARATION = "GRADE-SUPPORT-001"
    CONTEXT_NON_SUPPORT = "CONTEXT-NON-SUPPORT-001"
    USDM_REFERENCE_WEEK = "USDM-DATE-001"
    USDM_D1_POSITIVE = "USDM-D1POS-001"
    USDM_D1_ZERO = "USDM-D1ZERO-001"
    USDM_MISSING = "USDM-MISSING-001"
    USDM_AXIS_SCOPE = "USDM-SCOPE-001"
    NOAA_EVENT_ROLE = "NOAA-ROLE-001"
    NOAA_SPATIAL = "NOAA-SPATIAL-001"
    NOAA_NO_UNIVERSAL_25KM = "NOAA-NO25KM-001"
    NOAA_STRUCTURED_IMPACT = "NOAA-IMPACT-STRUCT-001"
    NOAA_STRUCTURED_ATTRIBUTION = "NOAA-ATTR-STRUCT-001"
    NOAA_NARRATIVE = "NOAA-NARRATIVE-001"
    NOAA_AXIS_SCOPE = "NOAA-SCOPE-001"
    LINKAGE_EXPLICIT = "LINK-EXPLICIT-001"
    LINKAGE_NON_INFERENCE = "LINK-NONINFER-001"
    OPENFEMA_DISABLED = "OPENFEMA-DISABLED-001"
    OPENFEMA_ADMIN_ONLY = "OPENFEMA-ADMIN-001"
    LABEL_PRIMARY = "LABEL-PRIMARY-001"
    LABEL_VIEWS = "LABEL-VIEWS-001"
    LABEL_LEGACY = "LABEL-LEGACY-001"
    SUMMARY_NEUTRAL = "SUMMARY-NEUTRAL-001"
    FROZEN_WEB_PRESERVE = "FROZEN-WEB-PRESERVE-001"
    WEB_AXIS_YES = "WEB-AXIS-YES-001"
    WEB_AXIS_NO = "WEB-AXIS-NO-001"
    WEB_AXIS_UNRESOLVED_EVIDENCE = "WEB-AXIS-UNRES-001"
    WEB_AXIS_UNRESOLVED_DEPENDENCY = "WEB-AXIS-UNRES-002"
    WEB_AXIS_ZERO_PAGES = "WEB-AXIS-UNRES-003"
    WEB_AXIS_NOT_RUN = "WEB-AXIS-UNRES-004"
    PROHIBIT_USDM_NOAA_LINKAGE = "PROHIBIT-USDM-NOAA-LINK-001"
    PROHIBIT_WET_IMPLIES_IMPACT = "PROHIBIT-WET-IMPACT-001"
    PROHIBIT_IMPACT_IMPLIES_ATTRIBUTION = "PROHIBIT-IMPACT-ATTR-001"
    PROHIBIT_ADMIN_SUPPORT = "PROHIBIT-ADMIN-AXIS-001"
    PROHIBIT_PHYSICAL_CONFIRMATION = "PROHIBIT-PHYSICAL-CONFIRM-001"
    PROHIBIT_OFFICIAL_WEB_MERGE = "PROHIBIT-OFFICIAL-WEB-MERGE-001"
    PROHIBIT_CONTEXT_ACCUMULATION = "PROHIBIT-CONTEXT-ACCUM-001"


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CanonicalSchemaError(f"required_nonempty:{field_name}")


def _require_sha256(value: str, field_name: str, *, optional: bool = False) -> None:
    if optional and not value:
        return
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise CanonicalSchemaError(f"invalid_sha256:{field_name}")


def _reject_unknown(data: Mapping[str, Any], cls: type[Any]) -> None:
    expected = {item.name for item in fields(cls)}
    unknown = set(data) - expected
    if unknown:
        raise CanonicalSchemaError(f"unknown_fields:{cls.__name__}:{sorted(unknown)}")


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _serialize(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {
            (key.value if isinstance(key, Enum) else str(key)): _serialize(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


@dataclass(frozen=True)
class VersionLineage:
    scientific_contract_version: str = SCIENTIFIC_CONTRACT_VERSION
    schema_version: str = SCHEMA_VERSION
    adapter_version: str = ADAPTER_VERSION_ENVELOPE
    aggregator_version: str = AGGREGATOR_VERSION_ENVELOPE

    def __post_init__(self) -> None:
        if self.scientific_contract_version != SCIENTIFIC_CONTRACT_VERSION:
            raise CanonicalSchemaError("incompatible_scientific_contract_version")
        if self.schema_version != SCHEMA_VERSION:
            raise CanonicalSchemaError("incompatible_schema_version")
        if not self.adapter_version.startswith(ADAPTER_VERSION_ENVELOPE):
            raise CanonicalSchemaError("adapter_version_outside_envelope")
        if not self.aggregator_version.startswith(AGGREGATOR_VERSION_ENVELOPE):
            raise CanonicalSchemaError("aggregator_version_outside_envelope")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VersionLineage":
        _reject_unknown(data, cls)
        return cls(**data)


@dataclass(frozen=True)
class FrozenWebpageExecutionDetails:
    retrieval_performed: bool
    logical_page_count: int
    successfully_fetched_or_readable_count: int
    source_content_insufficient_count: int
    completed_judgment_count: int
    candidate_relevant_unresolved_count: int
    guarded_page_count: int
    zero_pages_retrieved: bool
    retrieval_coverage_state: RetrievalCoverageState
    judgment_coverage_state: JudgmentCoverageState
    webpage_coverage: WebpageCoverage
    frozen_manifest_hash: str
    page_result_bundle_hash: str
    body_bundle_hash: str
    judgment_bundle_hash: str
    guard_bundle_hash: str
    derivation_rule_id: RuleId
    adapter_version: str = FROZEN_WEBPAGE_ADAPTER_VERSION

    def __post_init__(self) -> None:
        counts = {
            "logical_page_count": self.logical_page_count,
            "successfully_fetched_or_readable_count": self.successfully_fetched_or_readable_count,
            "source_content_insufficient_count": self.source_content_insufficient_count,
            "completed_judgment_count": self.completed_judgment_count,
            "candidate_relevant_unresolved_count": self.candidate_relevant_unresolved_count,
            "guarded_page_count": self.guarded_page_count,
        }
        for name, value in counts.items():
            if not isinstance(value, int) or value < 0:
                raise CanonicalSchemaError(f"invalid_nonnegative_count:{name}")
        logical = self.logical_page_count
        for name, value in counts.items():
            if name != "logical_page_count" and value > logical:
                raise CanonicalSchemaError(f"count_exceeds_logical_pages:{name}")
        for name in (
            "frozen_manifest_hash",
            "page_result_bundle_hash",
            "body_bundle_hash",
            "judgment_bundle_hash",
            "guard_bundle_hash",
        ):
            _require_sha256(getattr(self, name), name)
        if not self.adapter_version.startswith(ADAPTER_VERSION_ENVELOPE):
            raise CanonicalSchemaError("webpage_adapter_version_outside_envelope")
        if self.zero_pages_retrieved:
            if logical != 0 or any(value != 0 for name, value in counts.items() if name != "logical_page_count"):
                raise CanonicalSchemaError("zero_pages_requires_all_zero_counts")
            if not self.retrieval_performed:
                raise CanonicalSchemaError("zero_pages_requires_proven_retrieval")
            if self.webpage_coverage is not WebpageCoverage.ZERO_PAGES_RETRIEVED:
                raise CanonicalSchemaError("zero_pages_requires_zero_page_coverage")
        if self.webpage_coverage is WebpageCoverage.BOUNDED_COMPLETE_FOR_FROZEN_SCOPE:
            if logical == 0:
                raise CanonicalSchemaError("bounded_complete_requires_pages")
            if self.successfully_fetched_or_readable_count != logical:
                raise CanonicalSchemaError("bounded_complete_requires_all_readable")
            if self.completed_judgment_count != logical or self.guarded_page_count != logical:
                raise CanonicalSchemaError("bounded_complete_requires_all_judged_and_guarded")
            if self.source_content_insufficient_count:
                raise CanonicalSchemaError("bounded_complete_forbids_insufficient_content")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FrozenWebpageExecutionDetails":
        _reject_unknown(data, cls)
        converted = dict(data)
        converted["retrieval_coverage_state"] = RetrievalCoverageState(
            converted["retrieval_coverage_state"]
        )
        converted["judgment_coverage_state"] = JudgmentCoverageState(
            converted["judgment_coverage_state"]
        )
        converted["webpage_coverage"] = WebpageCoverage(converted["webpage_coverage"])
        converted["derivation_rule_id"] = RuleId(converted["derivation_rule_id"])
        return cls(**converted)


@dataclass(frozen=True)
class LaneExecutionRecord:
    candidate_id: str
    lane: str
    enabled: bool
    required: bool
    source_mode: str
    package_id: str
    package_sha256: str
    readiness: LaneReadiness
    execution_status: LaneExecutionStatus
    coverage: AxisCoverageState
    records_examined: int
    matches_found: int
    accepted_count: int
    contextual_count: int
    unresolved_count: int
    started_at: str = ""
    completed_at: str = ""
    checkpoint_id: str = ""
    retry_count: int = 0
    attempt: int = 1
    webpage: FrozenWebpageExecutionDetails | None = None
    lineage: VersionLineage = field(default_factory=VersionLineage)

    def __post_init__(self) -> None:
        _require_text(self.candidate_id, "candidate_id")
        _require_text(self.lane, "lane")
        _require_text(self.source_mode, "source_mode")
        _require_text(self.package_id, "package_id")
        _require_sha256(self.package_sha256, "package_sha256", optional=True)
        for name in (
            "records_examined",
            "matches_found",
            "accepted_count",
            "contextual_count",
            "unresolved_count",
            "retry_count",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise CanonicalSchemaError(f"invalid_nonnegative_count:{name}")
        if not isinstance(self.attempt, int) or self.attempt < 1:
            raise CanonicalSchemaError("invalid_attempt")
        if self.matches_found > self.records_examined:
            raise CanonicalSchemaError("matches_exceed_records_examined")
        if self.accepted_count + self.contextual_count + self.unresolved_count > self.matches_found:
            raise CanonicalSchemaError("evidence_counts_exceed_matches")
        if self.execution_status is LaneExecutionStatus.DISABLED:
            if self.enabled:
                raise CanonicalSchemaError("disabled_status_requires_enabled_false")
            if self.readiness is not LaneReadiness.NOT_APPLICABLE:
                raise CanonicalSchemaError("disabled_status_requires_not_applicable_readiness")
        elif not self.enabled:
            raise CanonicalSchemaError("enabled_false_requires_disabled_status")
        if self.execution_status.is_successful_completion:
            if self.readiness is not LaneReadiness.READY:
                raise CanonicalSchemaError("successful_completion_requires_ready")
            if not self.completed_at:
                raise CanonicalSchemaError("successful_completion_requires_completed_at")
        if self.execution_status is LaneExecutionStatus.COMPLETED_ZERO_MATCH:
            if any(
                (
                    self.matches_found,
                    self.accepted_count,
                    self.contextual_count,
                    self.unresolved_count,
                )
            ):
                raise CanonicalSchemaError("completed_zero_match_requires_zero_match_counts")
            if self.lane == "frozen_webpage":
                raise CanonicalSchemaError("frozen_webpage_forbids_completed_zero_match")
        if self.execution_status is LaneExecutionStatus.COMPLETED_CONTEXT_ONLY:
            if self.matches_found < 1 or self.contextual_count < 1 or self.accepted_count:
                raise CanonicalSchemaError("completed_context_only_count_mismatch")
        if self.execution_status is LaneExecutionStatus.COMPLETED_WITH_MATCHES:
            if self.matches_found < 1:
                raise CanonicalSchemaError("completed_with_matches_requires_match")
        if self.execution_status in {
            LaneExecutionStatus.UNAVAILABLE,
            LaneExecutionStatus.RETRYABLE_FAILURE,
            LaneExecutionStatus.TERMINAL_FAILURE,
        } and self.coverage is AxisCoverageState.COMPLETE:
            raise CanonicalSchemaError("unsuccessful_execution_forbids_complete_coverage")
        if self.lane == "frozen_webpage" and self.webpage is None:
            raise CanonicalSchemaError("frozen_webpage_requires_execution_details")
        if self.lane != "frozen_webpage" and self.webpage is not None:
            raise CanonicalSchemaError("webpage_details_forbidden_for_other_lanes")

    @property
    def key(self) -> tuple[str, str]:
        return self.candidate_id, self.lane

    @property
    def successful_completion(self) -> bool:
        return self.execution_status.is_successful_completion

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LaneExecutionRecord":
        _reject_unknown(data, cls)
        converted = dict(data)
        converted["readiness"] = LaneReadiness(converted["readiness"])
        converted["execution_status"] = LaneExecutionStatus(converted["execution_status"])
        converted["coverage"] = AxisCoverageState(converted["coverage"])
        if converted.get("webpage") is not None:
            converted["webpage"] = FrozenWebpageExecutionDetails.from_dict(converted["webpage"])
        converted["lineage"] = VersionLineage.from_dict(converted.get("lineage", {}))
        return cls(**converted)


EXECUTION_ONLY_EVIDENCE_FIELDS = frozenset(
    {
        "execution_status",
        "enabled",
        "required",
        "readiness",
        "coverage",
        "zero_match",
        "retryable_failure",
        "terminal_failure",
    }
)


@dataclass(frozen=True)
class StructuredEvidenceRecord:
    candidate_id: str
    lane: str
    package_id: str
    source_role: SourceRole
    disposition: EvidenceDisposition
    source_record_id: str
    source_event_id: str
    source_file: str
    source_row: str
    source_url: str
    source_sha256: str
    source_version: str
    source_date: str
    raw_value_provenance: Mapping[str, Any]
    spatial_grade: SpatialAlignmentGrade
    temporal_grade: TemporalAlignmentGrade
    axis_assessments: Mapping[AxisName, AxisValue]
    rule_ids: tuple[RuleId, ...]
    reason_codes: tuple[ReasonCode, ...]
    record_hash: str
    record_kind: str = "actual_source_record_or_match"
    derived_review_row: bool = False
    lineage: VersionLineage = field(default_factory=VersionLineage)

    def __post_init__(self) -> None:
        _require_text(self.candidate_id, "candidate_id")
        _require_text(self.lane, "lane")
        _require_text(self.package_id, "package_id")
        _require_text(self.source_record_id, "source_record_id")
        if not self.source_file and not self.source_url:
            raise CanonicalSchemaError("actual_evidence_requires_source_file_or_url")
        _require_sha256(self.source_sha256, "source_sha256")
        _require_sha256(self.record_hash, "record_hash")
        if self.record_kind != "actual_source_record_or_match":
            raise CanonicalSchemaError("synthetic_evidence_record_kind_rejected")
        if self.derived_review_row:
            raise CanonicalSchemaError("derived_display_row_rejected")
        if not isinstance(self.raw_value_provenance, Mapping) or not self.raw_value_provenance:
            raise CanonicalSchemaError("raw_value_provenance_required")
        if not self.axis_assessments:
            raise CanonicalSchemaError("axis_assessments_required")
        if not all(isinstance(key, AxisName) and isinstance(value, AxisValue) for key, value in self.axis_assessments.items()):
            raise CanonicalSchemaError("invalid_axis_assessments")
        if not self.rule_ids:
            raise CanonicalSchemaError("rule_ids_required")
        if self.disposition is EvidenceDisposition.ACCEPTED and AxisValue.YES not in self.axis_assessments.values():
            raise CanonicalSchemaError("accepted_record_requires_yes_assessment")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StructuredEvidenceRecord":
        if EXECUTION_ONLY_EVIDENCE_FIELDS & set(data):
            raise CanonicalSchemaError("execution_only_state_in_structured_evidence")
        _reject_unknown(data, cls)
        converted = dict(data)
        converted["source_role"] = SourceRole(converted["source_role"])
        converted["disposition"] = EvidenceDisposition(converted["disposition"])
        converted["spatial_grade"] = SpatialAlignmentGrade(converted["spatial_grade"])
        converted["temporal_grade"] = TemporalAlignmentGrade(converted["temporal_grade"])
        converted["axis_assessments"] = {
            AxisName(key): AxisValue(value)
            for key, value in converted["axis_assessments"].items()
        }
        converted["rule_ids"] = tuple(RuleId(value) for value in converted["rule_ids"])
        converted["reason_codes"] = tuple(
            ReasonCode(value) for value in converted["reason_codes"]
        )
        converted["lineage"] = VersionLineage.from_dict(converted.get("lineage", {}))
        return cls(**converted)


@dataclass(frozen=True)
class CaseAxisRecord:
    candidate_id: str
    axis_name: AxisName
    axis_value: AxisValue
    coverage_state: AxisCoverageState
    official_supporting_record_ids: tuple[str, ...]
    official_contextual_record_ids: tuple[str, ...]
    webpage_supporting_record_ids: tuple[str, ...]
    webpage_contextual_record_ids: tuple[str, ...]
    strongest_spatial_grade: SpatialAlignmentGrade | None
    all_spatial_grades: tuple[SpatialAlignmentGrade, ...]
    strongest_temporal_grade: TemporalAlignmentGrade | None
    all_temporal_grades: tuple[TemporalAlignmentGrade, ...]
    decision_rule_id: RuleId
    unresolved_reason_codes: tuple[ReasonCode, ...]
    review_reason_codes: tuple[ReasonCode, ...]
    source_package_ids: tuple[str, ...]
    negative_scope: str = ""
    lineage: VersionLineage = field(default_factory=VersionLineage)

    def __post_init__(self) -> None:
        _require_text(self.candidate_id, "candidate_id")
        supporting = self.official_supporting_record_ids + self.webpage_supporting_record_ids
        if self.axis_value is AxisValue.YES:
            if not supporting:
                raise CanonicalSchemaError("axis_yes_requires_supporting_record")
            if self.coverage_state in {
                AxisCoverageState.NOT_RUN,
                AxisCoverageState.UNAVAILABLE,
                AxisCoverageState.FAILED,
            }:
                raise CanonicalSchemaError("axis_yes_requires_complete_or_partial_coverage")
        elif self.axis_value is AxisValue.NO:
            if supporting:
                raise CanonicalSchemaError("axis_no_forbids_supporting_record")
            if self.coverage_state is not AxisCoverageState.COMPLETE:
                raise CanonicalSchemaError("axis_no_requires_complete_coverage")
            if self.unresolved_reason_codes:
                raise CanonicalSchemaError("axis_no_forbids_unresolved_reason")
            if self.negative_scope != "bounded_evaluated_scope_only":
                raise CanonicalSchemaError("axis_no_requires_bounded_scope_statement")
        else:
            if not self.unresolved_reason_codes:
                raise CanonicalSchemaError("axis_unresolved_requires_reason")
        if self.coverage_state in {
            AxisCoverageState.NOT_RUN,
            AxisCoverageState.UNAVAILABLE,
            AxisCoverageState.FAILED,
        } and self.axis_value is not AxisValue.UNRESOLVED:
            raise CanonicalSchemaError("unsuccessful_coverage_requires_unresolved_axis")
        if self.strongest_spatial_grade and self.strongest_spatial_grade not in self.all_spatial_grades:
            raise CanonicalSchemaError("strongest_spatial_grade_not_in_all_grades")
        if self.strongest_temporal_grade and self.strongest_temporal_grade not in self.all_temporal_grades:
            raise CanonicalSchemaError("strongest_temporal_grade_not_in_all_grades")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CaseAxisRecord":
        _reject_unknown(data, cls)
        converted = dict(data)
        converted["axis_name"] = AxisName(converted["axis_name"])
        converted["axis_value"] = AxisValue(converted["axis_value"])
        converted["coverage_state"] = AxisCoverageState(converted["coverage_state"])
        converted["strongest_spatial_grade"] = (
            SpatialAlignmentGrade(converted["strongest_spatial_grade"])
            if converted.get("strongest_spatial_grade")
            else None
        )
        converted["all_spatial_grades"] = tuple(
            SpatialAlignmentGrade(value) for value in converted["all_spatial_grades"]
        )
        converted["strongest_temporal_grade"] = (
            TemporalAlignmentGrade(converted["strongest_temporal_grade"])
            if converted.get("strongest_temporal_grade")
            else None
        )
        converted["all_temporal_grades"] = tuple(
            TemporalAlignmentGrade(value) for value in converted["all_temporal_grades"]
        )
        converted["decision_rule_id"] = RuleId(converted["decision_rule_id"])
        converted["unresolved_reason_codes"] = tuple(
            ReasonCode(value) for value in converted["unresolved_reason_codes"]
        )
        converted["review_reason_codes"] = tuple(
            ReasonCode(value) for value in converted["review_reason_codes"]
        )
        for name in (
            "official_supporting_record_ids",
            "official_contextual_record_ids",
            "webpage_supporting_record_ids",
            "webpage_contextual_record_ids",
            "source_package_ids",
        ):
            converted[name] = tuple(converted[name])
        converted["lineage"] = VersionLineage.from_dict(converted.get("lineage", {}))
        return cls(**converted)


@dataclass(frozen=True)
class PhysicalStatusRecord:
    candidate_id: str
    status: PhysicalCandidateStatus
    authoritative_inventory_id: str
    physical_input_hashes: tuple[str, ...]
    decision_rule_id: RuleId = RuleId.PHYSICAL_CANDIDATE
    lineage: VersionLineage = field(default_factory=VersionLineage)

    def __post_init__(self) -> None:
        _require_text(self.candidate_id, "candidate_id")
        _require_text(self.authoritative_inventory_id, "authoritative_inventory_id")
        for index, digest in enumerate(self.physical_input_hashes):
            _require_sha256(digest, f"physical_input_hashes[{index}]")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PhysicalStatusRecord":
        _reject_unknown(data, cls)
        converted = dict(data)
        converted["status"] = PhysicalCandidateStatus(converted["status"])
        converted["physical_input_hashes"] = tuple(converted["physical_input_hashes"])
        converted["decision_rule_id"] = RuleId(converted["decision_rule_id"])
        converted["lineage"] = VersionLineage.from_dict(converted.get("lineage", {}))
        return cls(**converted)


@dataclass(frozen=True)
class LegacyCompatibilityView:
    legacy_tier: str
    source_artifact_id: str
    read_only: bool = True
    influences_repaired_v2_truth: bool = False

    def __post_init__(self) -> None:
        _require_text(self.source_artifact_id, "source_artifact_id")
        if not self.read_only or self.influences_repaired_v2_truth:
            raise CanonicalSchemaError("legacy_compatibility_must_be_read_only_and_non_influential")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LegacyCompatibilityView":
        _reject_unknown(data, cls)
        return cls(**data)


@dataclass(frozen=True)
class DerivedViewsRecord:
    candidate_id: str
    component_corroboration: ComponentCorroborationView
    impact_attribution: ImpactAttributionView
    linkage: LinkageView
    navigation_summary: NavigationSummary
    legacy_compatibility: LegacyCompatibilityView
    lineage: VersionLineage = field(default_factory=VersionLineage)

    def __post_init__(self) -> None:
        _require_text(self.candidate_id, "candidate_id")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DerivedViewsRecord":
        _reject_unknown(data, cls)
        converted = dict(data)
        converted["component_corroboration"] = ComponentCorroborationView(
            converted["component_corroboration"]
        )
        converted["impact_attribution"] = ImpactAttributionView(
            converted["impact_attribution"]
        )
        converted["linkage"] = LinkageView(converted["linkage"])
        converted["navigation_summary"] = NavigationSummary(
            converted["navigation_summary"]
        )
        converted["legacy_compatibility"] = LegacyCompatibilityView.from_dict(
            converted["legacy_compatibility"]
        )
        converted["lineage"] = VersionLineage.from_dict(converted.get("lineage", {}))
        return cls(**converted)


@dataclass(frozen=True)
class CaseOutputRecord:
    candidate_id: str
    physical_status: PhysicalStatusRecord
    axes: Mapping[AxisName, CaseAxisRecord]
    derived_views: DerivedViewsRecord
    lineage: VersionLineage = field(default_factory=VersionLineage)

    def __post_init__(self) -> None:
        _require_text(self.candidate_id, "candidate_id")
        if self.physical_status.candidate_id != self.candidate_id:
            raise CanonicalSchemaError("physical_status_candidate_mismatch")
        if self.derived_views.candidate_id != self.candidate_id:
            raise CanonicalSchemaError("derived_views_candidate_mismatch")
        if set(self.axes) != set(AxisName):
            raise CanonicalSchemaError("case_output_requires_exactly_five_axes")
        for axis_name, record in self.axes.items():
            if record.candidate_id != self.candidate_id or record.axis_name is not axis_name:
                raise CanonicalSchemaError("case_axis_identity_mismatch")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CaseOutputRecord":
        _reject_unknown(data, cls)
        converted = dict(data)
        converted["physical_status"] = PhysicalStatusRecord.from_dict(
            converted["physical_status"]
        )
        converted["axes"] = {
            AxisName(key): CaseAxisRecord.from_dict(value)
            for key, value in converted["axes"].items()
        }
        converted["derived_views"] = DerivedViewsRecord.from_dict(
            converted["derived_views"]
        )
        converted["lineage"] = VersionLineage.from_dict(converted.get("lineage", {}))
        return cls(**converted)


def validate_unique_lane_execution(records: Iterable[LaneExecutionRecord]) -> None:
    seen: set[tuple[str, str]] = set()
    for record in records:
        if record.key in seen:
            raise CanonicalSchemaError(f"duplicate_lane_execution:{record.key[0]}:{record.key[1]}")
        seen.add(record.key)


_ALLOWED_TRANSITIONS: Mapping[LaneExecutionStatus, frozenset[LaneExecutionStatus]] = {
    LaneExecutionStatus.DISABLED: frozenset({LaneExecutionStatus.DISABLED}),
    LaneExecutionStatus.UNAVAILABLE: frozenset(
        {
            LaneExecutionStatus.UNAVAILABLE,
            LaneExecutionStatus.RETRYABLE_FAILURE,
            LaneExecutionStatus.TERMINAL_FAILURE,
            LaneExecutionStatus.COMPLETED_WITH_MATCHES,
            LaneExecutionStatus.COMPLETED_ZERO_MATCH,
            LaneExecutionStatus.COMPLETED_CONTEXT_ONLY,
        }
    ),
    LaneExecutionStatus.RETRYABLE_FAILURE: frozenset(
        {
            LaneExecutionStatus.RETRYABLE_FAILURE,
            LaneExecutionStatus.TERMINAL_FAILURE,
            LaneExecutionStatus.COMPLETED_WITH_MATCHES,
            LaneExecutionStatus.COMPLETED_ZERO_MATCH,
            LaneExecutionStatus.COMPLETED_CONTEXT_ONLY,
        }
    ),
    LaneExecutionStatus.TERMINAL_FAILURE: frozenset({LaneExecutionStatus.TERMINAL_FAILURE}),
    LaneExecutionStatus.COMPLETED_WITH_MATCHES: frozenset(
        {LaneExecutionStatus.COMPLETED_WITH_MATCHES}
    ),
    LaneExecutionStatus.COMPLETED_ZERO_MATCH: frozenset(
        {LaneExecutionStatus.COMPLETED_ZERO_MATCH}
    ),
    LaneExecutionStatus.COMPLETED_CONTEXT_ONLY: frozenset(
        {LaneExecutionStatus.COMPLETED_CONTEXT_ONLY}
    ),
}


def validate_lane_transition(
    previous: LaneExecutionRecord,
    current: LaneExecutionRecord,
) -> None:
    if previous.key != current.key:
        raise CanonicalSchemaError("lane_transition_key_mismatch")
    if current.execution_status not in _ALLOWED_TRANSITIONS[previous.execution_status]:
        raise CanonicalSchemaError(
            f"invalid_lane_transition:{previous.execution_status.value}:{current.execution_status.value}"
        )
    if current.attempt < previous.attempt:
        raise CanonicalSchemaError("lane_attempt_cannot_decrease")


__all__ = [
    "ADAPTER_VERSION_ENVELOPE",
    "AGGREGATOR_VERSION_ENVELOPE",
    "SCIENTIFIC_CONTRACT_VERSION",
    "SCHEMA_VERSION",
    "FROZEN_WEBPAGE_ADAPTER_VERSION",
    "AxisCoverageState",
    "AxisName",
    "AxisValue",
    "CanonicalSchemaError",
    "CaseAxisRecord",
    "CaseOutputRecord",
    "ComponentCorroborationView",
    "DerivedViewsRecord",
    "EvidenceDisposition",
    "FrozenWebpageExecutionDetails",
    "ImpactAttributionView",
    "JudgmentCoverageState",
    "LaneExecutionRecord",
    "LaneExecutionStatus",
    "LaneReadiness",
    "LegacyCompatibilityView",
    "LinkageView",
    "NavigationSummary",
    "PhysicalCandidateStatus",
    "PhysicalStatusRecord",
    "ReasonCode",
    "RetrievalCoverageState",
    "RuleId",
    "SourceRole",
    "SpatialAlignmentGrade",
    "StructuredEvidenceRecord",
    "TemporalAlignmentGrade",
    "VersionLineage",
    "WebpageCoverage",
    "validate_lane_transition",
    "validate_unique_lane_execution",
]
