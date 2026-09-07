# CE-Agent

**Evidence retrieval and validation for compound climate events.**

CE-Agent connects physically identified drought–rainfall events to documented hazards,
realized impacts, and explicit causal or temporal links. It combines structured official
records, bounded web retrieval, language-model judgments, and deterministic evidence checks.

The included California example reproduces **40 cases and 110 source pages without API calls**.

```mermaid
flowchart LR
    A[Event candidates] --> B[Official records and web evidence]
    B --> C[Semantic judgments]
    C --> D[Date, location, and quotation checks]
    D --> E[Separate evidence axes]
    E --> F[Case-level support]
```

## What it does

- Keeps hazard evidence, observed impacts, attribution, and drought-to-wet linkage separate.
- Checks whether quoted evidence occurs in the source text and matches the candidate event.
- Bounds search, page fetching, query expansion, and model requests.
- Preserves uncertainty when evidence is incomplete or contradictory.
- Recomputes case results from archived inputs with file and body integrity checks.

## Quick start

Use Python 3.11 or later; Python 3.12 is used for testing.

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` on macOS/Linux, then install the project:

```bash
python -m pip install .
python scripts/replay.py --verify
```

The replay checks its input hashes, recomputes page and case decisions, and compares
them with the supplied reference tables. It reads saved semantic judgments; it does
not rerun language-model inference or repeat live source retrieval.

To save a new set of outputs:

```bash
python scripts/replay.py --verify --output-dir examples/california40/outputs/replay
```

The output directory must not already exist. Generated files under `outputs/`
are ignored by Git.

## Included example

| Page result | Count |
| --- | ---: |
| Supports | 6 |
| Does not support | 65 |
| Unresolved | 32 |
| Insufficient source content | 7 |

These are page-level evidence decisions, not counts of climate events. Case-level
aggregation also uses the structured evidence in the example.

| Case result | Count |
| --- | ---: |
| Hazards and impact supported; explicit linkage absent | 31 |
| Partial or no public support | 6 |
| Additional evidence assessment needed | 3 |

Inputs, source URLs, field definitions, and reference outputs are in
[`examples/california40/`](examples/california40/). The accompanying inventory contains
408 candidates; the replay evaluates the 40-case subset in `candidate_manifest.csv`.

`replay_inputs/` contains archived pages, model judgments, structured evidence,
and their provenance. The `final_*` tables, guard records, and acceptance baseline
are fixed validation fixtures. Regenerated summaries and unresolved-case/page
exports belong in `outputs/`, not alongside these fixtures.

## Retrieval and validation

[`scripts/retrieve.py`](scripts/retrieve.py) exposes the manifest-based retrieval runner:

```bash
python scripts/retrieve.py --help
```

Live retrieval requires your own provider configuration and any official source caches
required by the selected lanes. Set `OPENAI_API_KEY` and `TAVILY_API_KEY` in your environment
when enabling those providers. The offline example does not require either variable.
Large raw climate datasets and bulk official-record caches are obtained separately.

## Code map

| Location | Purpose |
| --- | --- |
| `src/climate_pipeline/` | Retrieval, semantic validation, evidence checks, and aggregation |
| `src/climate_pipeline/pipeline/` | Typed evidence records and case aggregation |
| `configs/` | Event and evidence policies |
| `scripts/replay.py` | Deterministic California example |
| `scripts/retrieve.py` | Manifest-based retrieval |
| `tests/` | Evidence, uncertainty, budget, and replay checks |

## Tests

```bash
python -m pip install ".[test]"
python -m pytest
```

CI runs the full suite and the offline replay on Linux and Windows, checks imports
from the installed package, and verifies that the example leaves Git status clean.

## Data and interpretation

Archived source text is retained for evidence verification. Website sharing tokens and
HTTP client metadata have been removed; packaged hashes describe these distributed files.
The replay decisions remain identical to the source snapshot.

Evidence coverage is incomplete and source availability changes over time. A lack of
documented support is not proof that an event or impact did not occur. This implementation
assesses evidence alignment; it does not establish causal effects.
