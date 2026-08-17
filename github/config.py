"""GitHub-specific configuration. Extends config.py with GitHub API settings.

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


# --- GitHub API ---
GITHUB_TOKEN: str = _optional("GITHUB_TOKEN", "")
GITHUB_API_BASE: str = "https://api.github.com"

# --- Rate Limits ---
# REST API: 5,000 requests/hour with authentication
REST_RATE_LIMIT: int = 5000
REST_RATE_WINDOW: int = 3600  # seconds

# Code search: 10 requests/minute (most restrictive endpoint)
CODE_SEARCH_RATE_LIMIT: int = 10
CODE_SEARCH_RATE_WINDOW: int = 60

# Search API (users, repos): 30 requests/minute
SEARCH_RATE_LIMIT: int = 30
SEARCH_RATE_WINDOW: int = 60

# --- Pagination ---
RESULTS_PER_PAGE: int = 100  # GitHub max
MAX_RESULTS_PER_QUERY: int = 1000  # GitHub hard cap

# --- Enrichment ---
# Max repos to fetch per candidate (sorted by stars)
MAX_REPOS_PER_USER: int = 20
# Max commits to scan for email discovery per repo
MAX_COMMITS_FOR_EMAIL: int = 10
# Minimum data to attempt evaluation (at least this many non-fork repos)
MIN_REPOS_FOR_EVALUATION: int = 1

# --- Session Limits (GitHub governor) ---
# No anti-automation caps needed — pure API, no browser.
# Only real constraint is GitHub's own rate limits (handled by client).
MAX_SESSION_DURATION_SECONDS: int = 999 * 3600  # effectively uncapped
MAX_ENRICHMENTS_PER_SESSION: int = 999_999  # effectively uncapped
MAX_SESSIONS_PER_DAY: int = 999  # effectively uncapped

# --- Paths ---
PROJECT_ROOT: Path = Path(__file__).parent.parent
# GitHub state lives under the shared ``OUTPUT_DIR`` seam so it
# relocates with the rest of Cloris's writable state when running as
# a frozen .app. See ``shared/config.py:OUTPUT_DIR`` for the
# resolution rules.
from shared.config import OUTPUT_DIR as _OUTPUT_DIR
GITHUB_STATE_ROOT: Path = _OUTPUT_DIR / "state" / "github"
GITHUB_STATE_ROOT.mkdir(parents=True, exist_ok=True)
# Deprecated compatibility alias. New runs should resolve a brief-scoped state dir
# beneath GITHUB_STATE_ROOT rather than writing directly to this root.
GITHUB_OUTPUT_DIR: Path = GITHUB_STATE_ROOT
GITHUB_STATE_DIR: Path = Path.home() / ".sourcing-governor" / "github"
GITHUB_STATE_DIR.mkdir(parents=True, exist_ok=True)

# --- Enrichment Extensions ---
MAX_READMES_PER_CANDIDATE: int = 5
MAX_WEBSITE_FETCH_SIZE: int = 10000  # bytes

# --- Social Graph Expansion ---
MAX_STARGAZERS_PER_REPO: int = 500
MAX_FOLLOWERS_PER_SEED: int = 200
# Minimum confidence to trigger graph expansion from a save.
# Calibrated against post-Fix-2 score distribution where SIGNAL_SAVE = 0.40-0.55.
# Target: top ~30% of SIGNAL_SAVE scores qualify for expansion.
GRAPH_EXPANSION_MIN_CONFIDENCE: float = 0.50

# --- Geography filters (GitHub pipeline) ---
BAY_AREA_GEO_KEYWORDS: list[str] = [
    "san francisco",
    "sf",
    "bay area",
    "silicon valley",
    "palo alto",
    "mountain view",
    "menlo park",
    "berkeley",
    "oakland",
    "san jose",
    "sunnyvale",
    "santa clara",
    "redwood city",
    "cupertino",
    "stanford",
    "san mateo",
    "fremont",
    ", ca",
    "california",
]

# In-state-but-not-Bay markers — must reject even when location also contains
# "california" or ", ca".
NON_BAY_US_MARKERS: list[str] = [
    "los angeles",
    "san diego",
    "sacramento",
    "irvine",
    "long beach",
    "fresno",
]
