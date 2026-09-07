from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from climate_pipeline.case_aggregation import recompute_case_results
from climate_pipeline.full_body_direct_judge import derive_local_page_result


DEFAULT_SNAPSHOT = ROOT / "examples" / "california40"
PAGE_FIELDS = [
    "page_id",
    "candidate_id",
    "source_url",
    "semantic_provenance",
    "terminal_state",
    "model_support_recommendation",
    "page_result",
    "candidate_hazard_support",
    "realized_impact_support",
    "attribution_support",
    "transition_support",
    "previous_page_result",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fields = fields or (list(rows[0]) if rows else [])
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_replay_input_hashes(snapshot: Path) -> None:
    manifest = load_json(snapshot / "provenance_manifest.json")
    expected = manifest.get("replay_input_sha256") or {}
    for relative, digest in expected.items():
        path = snapshot / relative
        if not path.is_file():
            raise RuntimeError(f"missing_replay_input:{relative}")
        actual = sha256_file(path)
        if actual != digest:
            raise RuntimeError(f"replay_input_hash_mismatch:{relative}:{actual}")


def replay(snapshot: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    inputs = snapshot / "replay_inputs"
    requests = load_jsonl(inputs / "candidate_page_manifest.jsonl")
    frozen_pages = {
        row["logical_page_id"]: row
        for row in load_jsonl(inputs / "logical_frozen_pages.jsonl")
    }
    final_semantics = {
        row["page_id"]: row
        for row in load_jsonl(inputs / "final_semantic_judgments.jsonl")
    }
    prior_page_results = {
        row["page_id"]: row
        for row in load_csv(inputs / "prior_page_results.csv")
    }
    results: list[dict[str, Any]] = []
    guards: list[dict[str, Any]] = []
    for request in requests:
        page_id = str(request["page_id"])
        frozen_semantic = final_semantics[page_id]
        semantic = frozen_semantic.get("parsed_semantic_result")
        terminal_state = str(frozen_semantic.get("terminal_state") or "")
        semantic_provenance = str(frozen_semantic.get("semantic_provenance") or "")
        body = str(frozen_pages[page_id].get("body_text_or_archived_body_text") or "")
        if hashlib.sha256(body.encode("utf-8")).hexdigest() != request["body_sha256"]:
            raise RuntimeError(f"frozen_body_hash_mismatch:{page_id}")
        local = derive_local_page_result(
            semantic=semantic,
            candidate=request["candidate"],
            body_text=body,
            call_status=terminal_state,
        )
        row = {
            "page_id": page_id,
            "candidate_id": request["candidate_id"],
            "source_url": request["source_url"],
            "semantic_provenance": semantic_provenance,
            "terminal_state": terminal_state,
            "model_support_recommendation": (semantic or {}).get(
                "candidate_support_recommendation", ""
            ),
            "page_result": local["page_result"],
            "candidate_hazard_support": local["candidate_hazard_support"],
            "realized_impact_support": local["realized_impact_support"],
            "attribution_support": local["explicit_attribution_support"],
            "transition_support": local[
                "explicit_drought_to_wet_transition_support"
            ],
            "previous_page_result": prior_page_results[page_id]["page_result"],
        }
        results.append(row)
        guards.append(
            {
                **row,
                "source_event_axes": local["source_event_axes"],
                "guard_actions": local["guard_actions"],
            }
        )

    if len(results) != 110 or len({row["page_id"] for row in results}) != 110:
        raise RuntimeError("expected_110_unique_pages")
    manifest_rows = load_csv(snapshot / "candidate_manifest.csv")
    structured_rows = load_csv(inputs / "structured_evidence_results.csv")
    cases = recompute_case_results(
        page_rows=results,
        structured_rows=structured_rows,
        manifest_rows=manifest_rows,
    )
    if len(cases) != 40:
        raise RuntimeError("expected_40_unique_cases")
    return results, guards, cases


def exact_rows(rows: list[dict[str, Any]], fields: list[str]) -> list[tuple[str, ...]]:
    return [tuple(str(row.get(field) if row.get(field) is not None else "") for field in fields) for row in rows]


def verify_expected(snapshot: Path, pages: list[dict[str, Any]], cases: list[dict[str, Any]]) -> None:
    expected_pages = load_csv(snapshot / "final_110_candidate_page_results.csv")
    expected_cases = load_csv(snapshot / "final_40_case_results.csv")
    case_fields = list(expected_cases[0])
    if exact_rows(pages, PAGE_FIELDS) != exact_rows(expected_pages, PAGE_FIELDS):
        raise RuntimeError("frozen_page_results_do_not_match_canonical_snapshot")
    if exact_rows(cases, case_fields) != exact_rows(expected_cases, case_fields):
        raise RuntimeError("frozen_case_results_do_not_match_canonical_snapshot")


def summary(pages: list[dict[str, Any]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "external_api_calls": 0,
        "page_count": len(pages),
        "case_count": len(cases),
        "page_result_counts": dict(
            sorted(Counter(row["page_result"] for row in pages).items())
        ),
        "case_support_tier_counts": dict(
            sorted(Counter(row["support_tier"] for row in cases).items())
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Zero-API replay of the frozen California 40-case snapshot.")
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    snapshot = args.snapshot_dir.resolve()
    verify_replay_input_hashes(snapshot)
    pages, guards, cases = replay(snapshot)
    if args.verify:
        verify_expected(snapshot, pages, cases)
    if args.output_dir:
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
        write_csv(output / "final_110_candidate_page_results.csv", pages, PAGE_FIELDS)
        write_jsonl(output / "final_110_page_guard_details.jsonl", guards)
        write_csv(output / "final_40_case_results.csv", cases)
        write_csv(
            output / "unresolved_pages.csv",
            [
                row
                for row in pages
                if row["page_result"]
                in {"unresolved", "insufficient_source_content"}
            ],
            PAGE_FIELDS,
        )
        write_csv(
            output / "unresolved_cases.csv",
            [row for row in cases if row["support_tier"] == "needs_review"],
        )
        write_json(output / "replay_summary.json", summary(pages, cases))
    print(json.dumps(summary(pages, cases), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
