from __future__ import annotations

import os
from pathlib import Path

RUN_MODE = "raw_ce_active"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_HOME = Path(os.getenv("CE_AGENT_PROJECT_ROOT", PROJECT_ROOT.parent)).resolve()
DATA_ROOT = Path(os.getenv("CE_AGENT_DATA_ROOT", PROJECT_HOME / "data")).resolve()
RUN_ROOT = Path(os.getenv("CE_AGENT_RUN_ROOT", PROJECT_HOME / "runs")).resolve()
DATA_DIR = DATA_ROOT
DATASET_DIR = DATA_ROOT / "dataset"
OUTPUT_DIR = RUN_ROOT
SNAPSHOT_DIR = OUTPUT_DIR / "snapshots"
LOG_DIR = OUTPUT_DIR / "logs"

RAW_CE_PILOT_CONFIG = PROJECT_ROOT / "configs" / "raw_ce_pilot_texas_2021_2025.yaml"
RAW_CE_PILOT_OUTPUT_DIR = OUTPUT_DIR / "raw_ce_pilot" / "texas_2021_2025"

SPEI3_DROUGHT_US_CSV = DATASET_DIR / "spei3_drought_us_2000_2025.csv"
P99_RAINFALL_US_CSV = DATASET_DIR / "events_extreme_rain_pr3_p99_1985_2025.csv"
DTER_US_CSV = DATASET_DIR / "dter_eca_events_us_2000_2025.csv"
COUNTY_BOUNDARY_ZIP = DATA_DIR / "raw" / "tl_2024_us_county.zip"

MAX_EVENTS = 8
MIN_EVENTS = 4
MAX_SELECTED_GRID_ROWS_PER_CASE = 12
MAX_ITERATIONS_PER_EVENT = int(os.getenv("MAX_ITERATIONS_PER_EVENT", "14"))
MAX_SEARCH_QUERIES_PER_EVENT = int(os.getenv("MAX_SEARCH_QUERIES_PER_EVENT", "24"))
MAX_FETCHES_PER_EVENT = int(os.getenv("MAX_FETCHES_PER_EVENT", "80"))
MAX_ACCEPTED_PAGES_PER_EVENT = int(os.getenv("MAX_ACCEPTED_PAGES_PER_EVENT", "15"))
MAX_LLM_CALLS_PER_EVENT = int(os.getenv("MAX_LLM_CALLS_PER_EVENT", "60"))
MAX_SEARCH_RESULTS_PER_QUERY = int(os.getenv("MAX_SEARCH_RESULTS_PER_QUERY", "8"))
MAX_FETCHES_PER_QUERY = int(os.getenv("MAX_FETCHES_PER_QUERY", "4"))

SEARCH_CLIENT = os.getenv("SEARCH_CLIENT", os.getenv("SEARCH_PROVIDER", "fixture")).lower()
US_RETRIEVAL_PROVIDER = os.getenv("US_RETRIEVAL_PROVIDER", "tavily").lower()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock").lower()
PIPELINE_INPUT_MODE = os.getenv("PIPELINE_INPUT_MODE", "real").lower()
TARGET_CASE_IDS = [value.strip() for value in os.getenv("TARGET_CASE_IDS", "").split(",") if value.strip()]
ALLOW_SYNTHETIC_EVENT_FIXTURES = os.getenv("ALLOW_SYNTHETIC_EVENT_FIXTURES", "false").lower() == "true"

DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BOCHA_API_KEY = os.getenv("BOCHA_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5-2026-04-23")
OPENAI_RESPONSES_URL = os.getenv("OPENAI_RESPONSES_URL", "https://api.openai.com/v1/responses")
OPENAI_TIMEOUT_SECONDS = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_SEARCH_URL = os.getenv("TAVILY_SEARCH_URL", "https://api.tavily.com/search")
TAVILY_TIMEOUT_SECONDS = int(os.getenv("TAVILY_TIMEOUT_SECONDS", "30"))
BOCHA_SEARCH_URL = os.getenv("BOCHA_SEARCH_URL", "https://api.bochaai.com/v1/web-search")
BOCHA_TIMEOUT_SECONDS = int(os.getenv("BOCHA_TIMEOUT_SECONDS", "30"))
DEEPSEEK_TIMEOUT_SECONDS = int(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "45"))
WEB_READER_MIN_CLEANED_TEXT_CHARS = int(os.getenv("WEB_READER_MIN_CLEANED_TEXT_CHARS", "200"))
WEB_READER_TIMEOUT_SECONDS = int(os.getenv("WEB_READER_TIMEOUT_SECONDS", "20"))
WEB_READER_HTTP_RETRIES = int(os.getenv("WEB_READER_HTTP_RETRIES", "2"))
REASONING_EFFORT = os.getenv("DEEPSEEK_REASONING_EFFORT", "high")
THINKING_ENABLED = os.getenv("DEEPSEEK_THINKING_ENABLED", "true").lower() == "true"

SCORING_VERSION = "link_score_v1"
NORMALIZER_VERSION = "normalizer_v1"
DETECTION_RULE_VERSION = "mvp_csv_selection_v1"
EVENT_ADAPTER_VERSION = "event_adapter_v1"
RETRIEVAL_POLICY_VERSION = "retrieval_policy_v1"
PLANNER_PROMPT_VERSION = "planner_v1"
PAGE_CLASSIFIER_VERSION = "page_classifier_v1"
EXTRACTOR_PROMPT_VERSION = "extractor_v1"
DOSSIER_VERSION = "dossier_v1"
CHAIN_VERSION = "chain_v1"

OUTPUT_FILES = {
    "events": OUTPUT_DIR / "events.jsonl",
    "official_sources": OUTPUT_DIR / "official_sources.jsonl",
    "retrieval_actions": OUTPUT_DIR / "retrieval_actions.jsonl",
    "page_audits": OUTPUT_DIR / "page_audits.jsonl",
    "evidence_units": OUTPUT_DIR / "evidence_units.jsonl",
    "evidence_records": OUTPUT_DIR / "evidence_records.jsonl",
    "normalized_evidence": OUTPUT_DIR / "normalized_evidence.jsonl",
    "event_evidence_links": OUTPUT_DIR / "event_evidence_links.jsonl",
    "candidate_chains": OUTPUT_DIR / "candidate_chains.jsonl",
    "qa_report": OUTPUT_DIR / "qa_report.md",
}

LIVE_ATTEMPT_DIR = OUTPUT_DIR / os.getenv("LIVE_ATTEMPT_NAME", "live_attempt")
LIVE_SNAPSHOT_DIR = LIVE_ATTEMPT_DIR / "snapshots"
LIVE_SEARCH_RESULTS_FILE = LIVE_ATTEMPT_DIR / "search_results.jsonl"

__all__ = [
    "RUN_MODE",
    "PROJECT_ROOT",
    "PROJECT_HOME",
    "DATA_ROOT",
    "RUN_ROOT",
    "DATA_DIR",
    "DATASET_DIR",
    "OUTPUT_DIR",
    "SNAPSHOT_DIR",
    "LOG_DIR",
    "RAW_CE_PILOT_CONFIG",
    "RAW_CE_PILOT_OUTPUT_DIR",
    "SPEI3_DROUGHT_US_CSV",
    "P99_RAINFALL_US_CSV",
    "DTER_US_CSV",
    "COUNTY_BOUNDARY_ZIP",
    "MAX_EVENTS",
    "MIN_EVENTS",
    "MAX_SELECTED_GRID_ROWS_PER_CASE",
    "MAX_ITERATIONS_PER_EVENT",
    "MAX_SEARCH_QUERIES_PER_EVENT",
    "MAX_FETCHES_PER_EVENT",
    "MAX_ACCEPTED_PAGES_PER_EVENT",
    "MAX_LLM_CALLS_PER_EVENT",
    "MAX_SEARCH_RESULTS_PER_QUERY",
    "MAX_FETCHES_PER_QUERY",
    "SEARCH_CLIENT",
    "US_RETRIEVAL_PROVIDER",
    "LLM_PROVIDER",
    "PIPELINE_INPUT_MODE",
    "TARGET_CASE_IDS",
    "ALLOW_SYNTHETIC_EVENT_FIXTURES",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_API_KEY",
    "BOCHA_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_RESPONSES_URL",
    "OPENAI_TIMEOUT_SECONDS",
    "TAVILY_API_KEY",
    "TAVILY_SEARCH_URL",
    "TAVILY_TIMEOUT_SECONDS",
    "BOCHA_SEARCH_URL",
    "BOCHA_TIMEOUT_SECONDS",
    "DEEPSEEK_TIMEOUT_SECONDS",
    "WEB_READER_MIN_CLEANED_TEXT_CHARS",
    "WEB_READER_TIMEOUT_SECONDS",
    "WEB_READER_HTTP_RETRIES",
    "REASONING_EFFORT",
    "THINKING_ENABLED",
    "SCORING_VERSION",
    "NORMALIZER_VERSION",
    "DETECTION_RULE_VERSION",
    "EVENT_ADAPTER_VERSION",
    "RETRIEVAL_POLICY_VERSION",
    "PLANNER_PROMPT_VERSION",
    "PAGE_CLASSIFIER_VERSION",
    "EXTRACTOR_PROMPT_VERSION",
    "DOSSIER_VERSION",
    "CHAIN_VERSION",
    "OUTPUT_FILES",
    "LIVE_ATTEMPT_DIR",
    "LIVE_SNAPSHOT_DIR",
    "LIVE_SEARCH_RESULTS_FILE",
]
