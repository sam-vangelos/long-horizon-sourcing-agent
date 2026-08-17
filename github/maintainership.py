"""Maintainership-level classifier for OSS Maintainers Slice 4.

Per OSS Maintainers Module Spec §13.1, this is the single highest-
leverage correctness surface in the module: given a candidate
``username`` and a list of recruiter-named ``target_projects``,
classify the candidate as ``contributor`` / ``maintainer`` /
``project_lead``. The output rides on
:class:`MaintainershipClassification` and feeds the evidence layer
(Slice 6) and the workspace card.

Signals (each scored independently, then aggregated):

1. **Merge authority** — count + recency of PRs in ``target_projects``
   where ``merged_by.login == username``. A maintainer-tier signal:
   only project members typically have merge rights.
2. **Release tag authorship** — count + cadence of releases where
   ``author.login == username``. project_lead-tier when sustained.
3. **Commit cadence** — sustained activity over the past 12 months.
   contributor-tier baseline; high cadence corroborates other signals.
4. **Reviewer activity** — sample of recent merged PRs; count
   reviews authored by ``username``. maintainer-tier.
5. **CONTRIBUTORS / MAINTAINERS text-mine** — exact username match
   in ``CONTRIBUTORS.md`` / ``MAINTAINERS.md`` / ``MAINTAINERS``.
   maintainer-tier when present.
6. **GOVERNANCE.md text-mine** — username appears in
   ``GOVERNANCE.md`` (typically lists project leads / BDFLs).
   project_lead-tier signal.
7. **README lead mention** — username appears in repo README's
   leads/maintainers section.

All signals are KEYED to ``target_projects``: someone can't be a
"maintainer" without the recruiter-named project attesting it. This
is the §12 mitigation against false-positives.

Per-candidate API budget: hard cap of 40 calls (spec §9). The
classifier counts calls and aborts gracefully when the cap is hit,
recording ``budget_exhausted=True`` in :attr:`MaintainershipClassification.signals`.

Confidence is the aggregate weighted sum normalized to ``[0, 1]``.
The level decision is rule-based, not learned: thresholds tuned
against the calibration fixture at
:file:`tests/fixtures/github_maintainership_ground_truth.json` per
the spec §13.1 agreement gate (exact-level ≥ 0.80, within-one-level
≥ 0.95, non-adjacent confusion ≤ 0.02).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

from github import maintainer_signal_cache as mcache
from github.client import GitHubClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class MaintainershipClassification:
    """Per-candidate output of :func:`classify`.

    ``level`` is the final classification; ``confidence`` is the
    aggregate score in [0, 1]; ``evidence_sources`` is a recruiter-
    readable list of cited signals (e.g.
    ``"merge_authority:kubernetes/kubernetes:23PRs"``); ``signals``
    is the raw per-signal payload for debugging + per-signal
    contribution inspection (spec §13.1).

    ``role_certainty`` distinguishes inferred classifier output
    (``"inferred"``) from roster/registry-declared roles merged at
    the channel seam (``"declared"`` via
    :func:`merge_declared_maintainership`).
    """

    level: str
    confidence: float
    evidence_sources: list[str] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)
    role_certainty: str = "inferred"

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "confidence": self.confidence,
            "evidence_sources": list(self.evidence_sources),
            "signals": dict(self.signals),
            "role_certainty": self.role_certainty,
        }


# ---------------------------------------------------------------------------
# Tunable weights + thresholds — calibrated against the fixture per §13.1
# ---------------------------------------------------------------------------


# Per-signal weights summed to determine confidence and level. Values
# are first-pass; Slice 4's agreement gate is the calibration target.
# Documented at the spec level so re-tuning has a single source of
# truth.
SIGNAL_WEIGHTS: dict[str, float] = {
    "merge_authority": 1.0,        # maintainer-tier
    "release_authorship": 1.5,     # project_lead-tier when sustained
    "commit_cadence": 0.6,         # contributor baseline
    "reviewer_activity": 0.8,      # maintainer-tier
    "contributors_file": 0.7,      # maintainer-tier (text-mine)
    "maintainers_file": 1.2,       # explicit maintainer assertion
    "governance_file": 1.5,        # project_lead-tier
    "readme_lead": 1.0,            # project_lead-tier
}


# Aggregate score thresholds. A candidate scores in [0, ~7]
# theoretically; thresholds are tuned to match the fixture's
# distribution. Values land mid-range so a well-merged contributor
# (merge_authority ≥ 5 + commit_cadence ≥ 50) can't trip the
# maintainer threshold without secondary corroboration.
LEVEL_THRESHOLDS: dict[str, float] = {
    "project_lead": 3.0,
    "maintainer": 1.6,
    # Anything below `maintainer` is `contributor`.
}


# Per-candidate budget. Defensive cap; the orchestrator may pass a
# lower value when running near a rate-limit ceiling.
DEFAULT_API_BUDGET: int = 40


# Time window for "recent" activity (commits, PRs).
RECENT_WINDOW_DAYS: int = 365


# How many merged PRs to sample for reviewer-activity scoring per
# target project. The classifier hits each PR's reviews endpoint, so
# this is the reviewer-activity API cost.
PR_SAMPLE_FOR_REVIEWS: int = 5


# How many merged PRs to scan for merge-authority scoring per target
# project. Higher than PR_SAMPLE_FOR_REVIEWS because we don't follow
# the reviews endpoint for these — just inspect ``merged_by.login``.
PR_SCAN_FOR_MERGE_AUTHORITY: int = 50


# Filenames to fetch per target project for text-mine signals.
GOVERNANCE_PATHS: tuple[str, ...] = ("GOVERNANCE.md", "GOVERNANCE")
CONTRIBUTORS_PATHS: tuple[str, ...] = (
    "CONTRIBUTORS.md",
    "CONTRIBUTORS",
    "CONTRIBUTORS.txt",
    ".github/CONTRIBUTORS.md",
)
MAINTAINERS_PATHS: tuple[str, ...] = (
    "MAINTAINERS.md",
    "MAINTAINERS",
    "MAINTAINERS.txt",
    ".github/MAINTAINERS.md",
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def classify(
    username: str,
    target_projects: list[str],
    client: GitHubClient,
    *,
    api_budget: int = DEFAULT_API_BUDGET,
) -> Optional[MaintainershipClassification]:
    """Classify a candidate's maintainership level on the named projects.

    Returns ``None`` when ``target_projects`` is empty (the spec §11
    behavior-preserving contract: classic github briefs without
    target projects skip classification entirely). Returns a
    classification dataclass otherwise; ``budget_exhausted=True`` in
    ``signals`` when the API cap was hit before all signals
    completed.
    """

    if not target_projects:
        return None

    cleaned_projects = [p.strip().lower() for p in target_projects if p.strip()]
    if not cleaned_projects:
        # All entries were blank/whitespace — treat as classic-github
        # brief per spec §11 (skip classification entirely).
        return None

    state = _ClassifyState(
        username=username,
        target_projects=cleaned_projects,
        client=client,
        budget_remaining=max(api_budget, 0),
    )

    # If the caller passed a budget of zero, mark exhausted up-front
    # so downstream consumers can detect "we never even started" via
    # the signals dict.
    if state.budget_remaining == 0:
        state.budget_exhausted = True

    # Per-signal scoring. Each signal updates ``state.signals`` and
    # ``state.evidence_sources`` and decrements ``budget_remaining``.
    # Signals run cheapest-first; budget exhaustion aborts early but
    # always returns a partial result.
    await _score_governance_files(state)
    await _score_contributors_file(state)
    await _score_maintainers_file(state)
    await _score_readme_lead(state)
    await _score_release_authorship(state)
    await _score_merge_authority(state)
    await _score_reviewer_activity(state)
    # commit_cadence reuses signals already on hand from
    # release_authorship + merge_authority; no extra API calls.
    _score_commit_cadence(state)

    # Aggregate.
    weighted_score = 0.0
    for signal_name, weight in SIGNAL_WEIGHTS.items():
        signal_value = state.signals.get(signal_name, 0.0)
        if isinstance(signal_value, (int, float)):
            weighted_score += weight * float(signal_value)

    level = _level_from_score(weighted_score)
    confidence = _confidence_from_score(weighted_score)

    if state.budget_exhausted:
        state.signals["budget_exhausted"] = True

    return MaintainershipClassification(
        level=level,
        confidence=confidence,
        evidence_sources=list(state.evidence_sources),
        signals=dict(state.signals),
    )


# ---------------------------------------------------------------------------
# Internal scoring state
# ---------------------------------------------------------------------------


@dataclass
class _ClassifyState:
    username: str
    target_projects: list[str]
    client: GitHubClient
    budget_remaining: int
    signals: dict[str, Any] = field(default_factory=dict)
    evidence_sources: list[str] = field(default_factory=list)
    budget_exhausted: bool = False

    def can_spend(self, calls: int = 1) -> bool:
        return self.budget_remaining >= calls

    def spend(self, calls: int = 1) -> None:
        self.budget_remaining = max(self.budget_remaining - calls, 0)
        if self.budget_remaining == 0:
            self.budget_exhausted = True


def _split_owner_repo(target: str) -> Optional[tuple[str, str]]:
    if "/" not in target:
        return None
    owner, repo = target.split("/", 1)
    if not owner or not repo:
        return None
    return owner, repo


# ---------------------------------------------------------------------------
# Signal scorers
# ---------------------------------------------------------------------------


async def _score_governance_files(state: _ClassifyState) -> None:
    """Text-mine ``GOVERNANCE.md`` for the candidate's username."""

    score = 0.0
    for target in state.target_projects:
        if not state.can_spend():
            break
        parsed = _split_owner_repo(target)
        if parsed is None:
            continue
        owner, repo = parsed
        text = await _fetch_governance(state, owner, repo, GOVERNANCE_PATHS, "governance")
        if text is None:
            continue
        if _username_in_text(state.username, text):
            score += 1.0
            state.evidence_sources.append(f"governance:{target}")
    state.signals["governance_file"] = score


async def _score_contributors_file(state: _ClassifyState) -> None:
    score = 0.0
    for target in state.target_projects:
        if not state.can_spend():
            break
        parsed = _split_owner_repo(target)
        if parsed is None:
            continue
        owner, repo = parsed
        text = await _fetch_governance(
            state, owner, repo, CONTRIBUTORS_PATHS, "contributors_file"
        )
        if text is None:
            continue
        if _username_in_text(state.username, text):
            score += 1.0
            state.evidence_sources.append(f"contributors_file:{target}")
    state.signals["contributors_file"] = score


async def _score_maintainers_file(state: _ClassifyState) -> None:
    score = 0.0
    for target in state.target_projects:
        if not state.can_spend():
            break
        parsed = _split_owner_repo(target)
        if parsed is None:
            continue
        owner, repo = parsed
        text = await _fetch_governance(
            state, owner, repo, MAINTAINERS_PATHS, "maintainers_file"
        )
        if text is None:
            continue
        if _username_in_text(state.username, text):
            score += 1.0
            state.evidence_sources.append(f"maintainers_file:{target}")
    state.signals["maintainers_file"] = score


async def _score_readme_lead(state: _ClassifyState) -> None:
    """Username mention in repo README's lead/maintainer section."""

    score = 0.0
    for target in state.target_projects:
        if not state.can_spend():
            break
        parsed = _split_owner_repo(target)
        if parsed is None:
            continue
        owner, repo = parsed
        # Try cache first.
        cached = mcache.get(owner, repo, "readme")
        text = cached.data if cached is not None and isinstance(cached.data, str) else None
        if text is None:
            try:
                text = await state.client.get_repo_readme(target)
                state.spend()
            except Exception as exc:  # noqa: BLE001 — fail-soft per spec §12
                logger.warning("readme fetch failed for %s: %s", target, exc)
                text = None
            if text is not None:
                mcache.put(owner, repo, "readme", text)
        if text is None:
            continue
        # Look for a lead/maintainer mention near the username.
        if _username_in_lead_section(state.username, text):
            score += 1.0
            state.evidence_sources.append(f"readme_lead:{target}")
    state.signals["readme_lead"] = score


async def _score_release_authorship(state: _ClassifyState) -> None:
    """Count releases authored by the candidate; score by recency + count."""

    total_authored = 0
    total_releases_seen = 0
    recent_authored = 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=RECENT_WINDOW_DAYS)
    for target in state.target_projects:
        if not state.can_spend():
            break
        parsed = _split_owner_repo(target)
        if parsed is None:
            continue
        owner, repo = parsed
        releases = await _fetch_releases(state, owner, repo)
        if releases is None:
            continue
        for release in releases:
            total_releases_seen += 1
            author = release.get("author") or {}
            if not isinstance(author, dict):
                continue
            if str(author.get("login") or "").lower() == state.username.lower():
                total_authored += 1
                if _within_window(release.get("published_at"), cutoff):
                    recent_authored += 1
        if total_authored > 0:
            state.evidence_sources.append(
                f"release_authorship:{target}:{total_authored}"
            )

    state.signals["release_authorship_count"] = total_authored
    state.signals["release_authorship_recent"] = recent_authored
    state.signals["release_authorship_total_seen"] = total_releases_seen
    # Score: 1.0 per recent release, 0.3 per older release, capped.
    score = min(recent_authored * 1.0 + (total_authored - recent_authored) * 0.3, 3.0)
    state.signals["release_authorship"] = score


async def _score_merge_authority(state: _ClassifyState) -> None:
    """Count PRs merged by the candidate within target projects."""

    total_merged_by_user = 0
    recent_merged_by_user = 0
    total_prs_scanned = 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=RECENT_WINDOW_DAYS)
    for target in state.target_projects:
        if not state.can_spend():
            break
        parsed = _split_owner_repo(target)
        if parsed is None:
            continue
        owner, repo = parsed
        pulls = await _fetch_pulls(state, owner, repo)
        if pulls is None:
            continue
        for pr in pulls:
            total_prs_scanned += 1
            merged_by = pr.get("merged_by") or {}
            if not isinstance(merged_by, dict):
                continue
            if str(merged_by.get("login") or "").lower() == state.username.lower():
                total_merged_by_user += 1
                if _within_window(pr.get("merged_at"), cutoff):
                    recent_merged_by_user += 1
        if total_merged_by_user > 0:
            state.evidence_sources.append(
                f"merge_authority:{target}:{total_merged_by_user}PRs"
            )

    state.signals["merge_authority_count"] = total_merged_by_user
    state.signals["merge_authority_recent"] = recent_merged_by_user
    state.signals["merge_authority_total_scanned"] = total_prs_scanned
    # Score curve: rewards sustained merge authority more than burst.
    # 0 PRs → 0.0; 1-3 → 0.5; 4-10 → 1.0; 11+ → 1.5; recent gets +0.5
    # bonus weight when ≥3.
    if total_merged_by_user >= 11:
        base = 1.5
    elif total_merged_by_user >= 4:
        base = 1.0
    elif total_merged_by_user >= 1:
        base = 0.5
    else:
        base = 0.0
    if recent_merged_by_user >= 3:
        base += 0.5
    state.signals["merge_authority"] = min(base, 2.0)


async def _score_reviewer_activity(state: _ClassifyState) -> None:
    """Sample recent merged PRs and count reviews authored by candidate."""

    reviews_authored = 0
    prs_inspected = 0
    for target in state.target_projects:
        if not state.can_spend():
            break
        parsed = _split_owner_repo(target)
        if parsed is None:
            continue
        owner, repo = parsed
        pulls = await _fetch_pulls(state, owner, repo)
        if pulls is None:
            continue
        # Sample the most-recent N PRs (already sorted desc by
        # `list_repo_pulls` on `updated`); cap by what's fresh.
        sample = list(pulls)[:PR_SAMPLE_FOR_REVIEWS]
        for pr in sample:
            number = pr.get("number")
            if not isinstance(number, int):
                continue
            if not state.can_spend():
                break
            prs_inspected += 1
            try:
                reviews = await state.client.get_pull_reviews(target, number)
                state.spend()
            except Exception as exc:  # noqa: BLE001 — fail-soft per spec §12
                logger.warning(
                    "pull reviews fetch failed for %s/#%d: %s", target, number, exc
                )
                continue
            for review in reviews:
                user = review.get("user") or {}
                if not isinstance(user, dict):
                    continue
                if (
                    str(user.get("login") or "").lower()
                    == state.username.lower()
                ):
                    reviews_authored += 1
        if reviews_authored > 0:
            state.evidence_sources.append(
                f"reviewer_activity:{target}:{reviews_authored}"
            )
    state.signals["reviewer_activity_count"] = reviews_authored
    state.signals["reviewer_activity_prs_inspected"] = prs_inspected
    # 0 → 0; 1-2 → 0.5; 3-5 → 1.0; 6+ → 1.5
    if reviews_authored >= 6:
        score = 1.5
    elif reviews_authored >= 3:
        score = 1.0
    elif reviews_authored >= 1:
        score = 0.5
    else:
        score = 0.0
    state.signals["reviewer_activity"] = score


def _score_commit_cadence(state: _ClassifyState) -> None:
    """Derive a cadence score from prior signals (no API calls).

    Uses the merge-authority + release-authorship signals already
    collected as a proxy for "this person ships consistently to the
    target projects." A first-pass; later slices may add a dedicated
    /commits scan if calibration shows the proxy is too coarse.
    """

    merge_recent = float(state.signals.get("merge_authority_recent", 0) or 0)
    release_recent = float(state.signals.get("release_authorship_recent", 0) or 0)
    cadence = merge_recent * 0.05 + release_recent * 0.1
    state.signals["commit_cadence"] = round(min(cadence, 1.0), 3)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _level_from_score(weighted_score: float) -> str:
    if weighted_score >= LEVEL_THRESHOLDS["project_lead"]:
        return "project_lead"
    if weighted_score >= LEVEL_THRESHOLDS["maintainer"]:
        return "maintainer"
    return "contributor"


def _confidence_from_score(weighted_score: float) -> float:
    """Map an aggregate score in [0, ~7] to a confidence in [0, 1].

    Saturating curve: low scores stay near 0; project_lead-tier
    scores land near 0.85+ but never hit 1.0 (calibration epistemic
    humility — we never claim perfect certainty on one classifier
    pass).
    """

    # Sigmoid-ish, manually tuned.
    if weighted_score <= 0.0:
        return 0.0
    if weighted_score >= 5.0:
        return 0.95
    return min(round(weighted_score / 5.0, 3), 0.95)


# ---------------------------------------------------------------------------
# Cached fetch helpers
# ---------------------------------------------------------------------------


async def _fetch_governance(
    state: _ClassifyState,
    owner: str,
    repo: str,
    paths: tuple[str, ...],
    cache_kind: str,
) -> Optional[str]:
    cached = mcache.get(owner, repo, cache_kind)
    if cached is not None and isinstance(cached.data, str):
        return cached.data
    for path in paths:
        if not state.can_spend():
            break
        try:
            text = await state.client.get_repo_contents(f"{owner}/{repo}", path)
            state.spend()
        except Exception as exc:  # noqa: BLE001 — fail-soft per spec §12
            logger.warning("contents fetch failed for %s/%s/%s: %s", owner, repo, path, exc)
            continue
        if text:
            mcache.put(owner, repo, cache_kind, text)
            return text
    # Cache the absence as empty string so we don't keep paging through
    # all the path variants on the next call.
    mcache.put(owner, repo, cache_kind, "")
    return None


async def _fetch_releases(
    state: _ClassifyState,
    owner: str,
    repo: str,
) -> Optional[list[dict]]:
    cached = mcache.get(owner, repo, "releases")
    if cached is not None and isinstance(cached.data, list):
        return cached.data
    if not state.can_spend():
        return None
    try:
        releases = await state.client.list_repo_releases(f"{owner}/{repo}")
        state.spend()
    except Exception as exc:  # noqa: BLE001 — fail-soft per spec §12
        logger.warning("releases fetch failed for %s/%s: %s", owner, repo, exc)
        return None
    mcache.put(owner, repo, "releases", releases)
    return releases


async def _fetch_pulls(
    state: _ClassifyState,
    owner: str,
    repo: str,
) -> Optional[list[dict]]:
    cached = mcache.get(owner, repo, "pr_merges")
    if cached is not None and isinstance(cached.data, list):
        return cached.data
    if not state.can_spend():
        return None
    try:
        pulls = await state.client.list_repo_pulls(
            f"{owner}/{repo}",
            state="closed",
            sort="updated",
            max_results=PR_SCAN_FOR_MERGE_AUTHORITY,
        )
        state.spend()
    except Exception as exc:  # noqa: BLE001 — fail-soft per spec §12
        logger.warning("pulls fetch failed for %s/%s: %s", owner, repo, exc)
        return None
    mcache.put(owner, repo, "pr_merges", pulls)
    return pulls


# ---------------------------------------------------------------------------
# Text-mine helpers
# ---------------------------------------------------------------------------


def _username_in_text(username: str, text: str) -> bool:
    """Match a github username verbatim in text content.

    Spec §12 contract: requires GitHub-username match (not display-
    name match); confidence is capped externally for display-name-
    only matches (a future signal). Matches ``@username`` and bare
    ``username`` with word-boundary discipline so ``alice`` doesn't
    match within ``aliceland``.
    """

    if not username or not text:
        return False
    pattern = re.compile(
        r"(?:^|[\s\@\(\[<])(" + re.escape(username) + r")(?:$|[\s\,\.\)\]\>:])",
        re.IGNORECASE,
    )
    return pattern.search(text) is not None


def _username_in_lead_section(username: str, readme_text: str) -> bool:
    """Heuristic: username appears within ~500 chars of a 'lead'/'maintainer' header."""

    if not username or not readme_text:
        return False
    lowered = readme_text.lower()
    lead_keywords = ("maintainer", "lead", "core team", "owners", "authors")
    for keyword in lead_keywords:
        idx = lowered.find(keyword)
        if idx == -1:
            continue
        section = readme_text[idx : idx + 500]
        if _username_in_text(username, section):
            return True
    return False


def _within_window(ts_str: Any, cutoff: datetime) -> bool:
    if not isinstance(ts_str, str) or not ts_str:
        return False
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except ValueError:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts >= cutoff


# ---------------------------------------------------------------------------
# Declared-role merge — roster/registry precedence over classifier
# ---------------------------------------------------------------------------

# Map roster/registry role labels to the classifier's level vocabulary.
# code_owner / maintainer / recipe_maintainer → maintainer;
# governance_listed → project_lead.
_DECLARED_ROLE_TO_LEVEL: dict[str, str] = {
    "code_owner": "maintainer",
    "maintainer": "maintainer",
    "recipe_maintainer": "maintainer",
    "governance_listed": "project_lead",
}

_LEVEL_ORDER: dict[str, int] = {
    "contributor": 0,
    "maintainer": 1,
    "project_lead": 2,
}


def _declared_level_from_entries(declared_entries: list[dict[str, Any]]) -> str:
    """Pick the highest maintainership level implied by declared roles."""

    best = "contributor"
    for entry in declared_entries:
        if not isinstance(entry, dict):
            continue
        mapped = _DECLARED_ROLE_TO_LEVEL.get(str(entry.get("role", "")))
        if mapped is None:
            continue
        if _LEVEL_ORDER.get(mapped, -1) > _LEVEL_ORDER.get(best, -1):
            best = mapped
    return best


def _declared_entry_target_key(entry: dict[str, Any]) -> str:
    return str(entry.get("repo") or entry.get("package") or "").strip().lower()


def declared_entries_for_target_projects(
    declared_entries: list[dict[str, Any]],
    target_projects: list[str],
) -> list[dict[str, Any]]:
    """Return declared roles scoped to recruiter-named target projects."""

    targets = {
        p.strip().lower()
        for p in target_projects
        if isinstance(p, str) and p.strip()
    }
    if not targets:
        return []
    scoped: list[dict[str, Any]] = []
    for entry in declared_entries:
        if not isinstance(entry, dict):
            continue
        key = _declared_entry_target_key(entry)
        if key and key in targets:
            scoped.append(entry)
    return scoped


def _declared_evidence_token(entry: dict[str, Any]) -> str:
    repo = str(entry.get("repo") or entry.get("package") or entry.get("hub") or "").strip()
    if not repo:
        return ""
    source_file = str(entry.get("source_file") or "").strip()
    return f"declared:{repo}:{source_file}"


def _inferred_dict(
    inferred: MaintainershipClassification | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(inferred, MaintainershipClassification):
        return inferred.to_dict()
    return dict(inferred)


def merge_declared_maintainership(
    declared_entries: list[dict[str, Any]],
    inferred: MaintainershipClassification | dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge roster/registry-declared roles ahead of classifier output.

    When ``declared_entries`` is non-empty, the returned dict's
    ``level`` comes from the declared evidence, ``role_certainty`` is
    ``"declared"``, ``evidence_sources`` leads with
    ``declared:<repo>:<source_file>`` entries, and the classifier's
    own level/confidence are preserved under ``corroboration`` only when
    a real classification was passed. With no declared entries, returns
    the inferred dict unchanged with ``role_certainty: "inferred"``.
    """

    if not declared_entries:
        if inferred is None:
            raise ValueError("merge_declared_maintainership requires inferred when declared_entries is empty")
        result = _inferred_dict(inferred)
        result.setdefault("role_certainty", "inferred")
        return result

    inferred_dict = _inferred_dict(inferred) if inferred is not None else None

    declared_sources = [
        token
        for entry in declared_entries
        if isinstance(entry, dict)
        for token in [_declared_evidence_token(entry)]
        if token
    ]
    inferred_sources = (
        list(inferred_dict.get("evidence_sources", []) or []) if inferred_dict else []
    )
    result: dict[str, Any] = {
        "level": _declared_level_from_entries(declared_entries),
        "confidence": inferred_dict.get("confidence") if inferred_dict else None,
        "role_certainty": "declared",
        "evidence_sources": declared_sources + inferred_sources,
        "signals": dict(inferred_dict.get("signals", {})) if inferred_dict else {},
    }
    if inferred_dict is not None:
        result["corroboration"] = {
            "level": inferred_dict.get("level", "contributor"),
            "confidence": inferred_dict.get("confidence", 0.0),
        }
    return result


# ---------------------------------------------------------------------------
# Calibration helpers — used by the test harness.
# ---------------------------------------------------------------------------


def level_distance(predicted: str, ground_truth: str) -> int:
    """Return ordinal distance between two levels.

    ``contributor`` → 0, ``maintainer`` → 1, ``project_lead`` → 2.
    Used by the test harness to compute within-one-level and non-
    adjacent confusion rates per spec §13.1.
    """

    order = {"contributor": 0, "maintainer": 1, "project_lead": 2}
    pa = order.get(predicted, -1)
    gb = order.get(ground_truth, -1)
    if pa < 0 or gb < 0:
        return 99
    return abs(pa - gb)


def evaluate_agreement(
    predictions: Iterable[tuple[str, str]],
) -> dict[str, float]:
    """Compute the spec §13.1 agreement metrics from (pred, truth) pairs.

    Returns a dict with keys:

    - ``"exact_rate"`` — fraction where pred == truth.
    - ``"within_one_rate"`` — fraction where ``level_distance ≤ 1``.
    - ``"non_adjacent_rate"`` — fraction where ``level_distance == 2``
      (project_lead ↔ contributor confusion; the spec §13.1 ≤ 0.02
      ceiling).
    - ``"n"`` — pair count.

    Empty input returns zeros (the caller treats this as a fixture
    error, not a passing gate).
    """

    pairs = list(predictions)
    n = len(pairs)
    if n == 0:
        return {"exact_rate": 0.0, "within_one_rate": 0.0, "non_adjacent_rate": 0.0, "n": 0}
    exact = sum(1 for p, g in pairs if p == g)
    within_one = sum(1 for p, g in pairs if level_distance(p, g) <= 1)
    non_adj = sum(1 for p, g in pairs if level_distance(p, g) == 2)
    return {
        "exact_rate": exact / n,
        "within_one_rate": within_one / n,
        "non_adjacent_rate": non_adj / n,
        "n": float(n),
    }
