"""Exa-specific configuration. Extends config.py with Exa API settings.

.env loading mirrors ``shared/config.py``: the user-data ``.env``
(written by the Cloris first-launch surface) takes priority when a
frozen .app is in play, with the project-root ``.env`` falling
through. See ``shared/user_data_dir.py`` for the resolution rules.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from shared.user_data_dir import (
    cloris_user_data_dir,
    should_use_user_data_dir,
)

_PROJECT_ROOT_ENV = Path(__file__).parent.parent / ".env"
if should_use_user_data_dir():
    _user_env_path = cloris_user_data_dir() / ".env"
    if _user_env_path.exists():
        load_dotenv(_user_env_path, override=False)
load_dotenv(_PROJECT_ROOT_ENV, override=False)


def _optional(key: str, default: str = "") -> str:
    return os.getenv(key, default)


# --- Exa API ---
# Resolved through ``shared.config`` so the ``AGENT_KEY_PREFIX`` override
# (``_agent_key``) applies exactly once, in one place.
from shared.config import EXA_API_KEY as _EXA_API_KEY

EXA_API_KEY: str = _EXA_API_KEY
EXA_API_BASE: str = "https://api.exa.ai"

# --- Rate limits (Exa published, 2026-09-03) ---
SEARCH_QPS: int = 10
CONTENTS_QPS: int = 100

# --- Request shape ---
DEFAULT_NUM_RESULTS: int = 10
MAX_NUM_RESULTS: int = 100
REQUEST_TIMEOUT_SECONDS: int = 30

# --- Request enums (rejected client-side before any network call) ---
SEARCH_TYPES: frozenset[str] = frozenset(
    {"instant", "fast", "auto", "deep-lite", "deep", "deep-reasoning"}
)
CATEGORIES: frozenset[str] = frozenset(
    {
        "company",
        "publication",
        "news",
        "personal site",
        "financial report",
        "people",
    }
)
MAX_DOMAIN_FILTER_ENTRIES: int = 1200
MIN_MAX_AGE_HOURS: int = -1
MAX_MAX_AGE_HOURS: int = 720

# --- Pricing (Exa published list, 2026-09-03) ---
# Used ONLY for worst-case pre-dispatch estimates in
# :class:`exa.cost_meter.ExaCostMeter`. Actual charges come from the
# response's ``costDollars`` and replace the estimate at settle time.
#
#   $7  / 1k search requests (first 10 results included)
#   +$1 / 1k results beyond the tenth
#   $1  / 1k pages, per content type (text | highlights | summary)
#   $12 / 1k deep searches, $15 / 1k deep-reasoning searches
PRICE_TABLE: dict = {
    "search_request_usd": {
        "instant": 0.007,
        "fast": 0.007,
        "auto": 0.007,
        "deep-lite": 0.007,
        "deep": 0.012,
        "deep-reasoning": 0.015,
    },
    "search_results_included": 10,
    "search_extra_result_usd": 0.001,
    "content_page_usd": 0.001,
    "content_types": ("text", "highlights", "summary"),
}

# --- Paths ---
PROJECT_ROOT: Path = Path(__file__).parent.parent
# Exa state lives under the shared ``OUTPUT_DIR`` seam so it relocates
# with the rest of Cloris's writable state when running as a frozen
# .app. See ``shared/config.py:OUTPUT_DIR`` for the resolution rules.
from shared.config import OUTPUT_DIR as _OUTPUT_DIR

# Declared, NOT created. Importing this module must not touch the disk:
# ``exa/run.py`` imports it for `.env` resolution on every invocation,
# including ``--check``, whose whole contract is that it writes nothing —
# an import-time mkdir made a preflight leave an empty state root behind
# and report a different posture on its second run. The directory is
# created where it is actually used: ``shared.output_paths
# .resolve_exa_state_dir`` (via ``source_state_root``) and the
# orchestrator's own ``run()``.
EXA_STATE_ROOT: Path = _OUTPUT_DIR / "state" / "exa"
