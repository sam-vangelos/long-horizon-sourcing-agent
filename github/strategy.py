"""GitHub strategy formation and adaptation — Opus plans and adjusts search execution.

Mirrors strategy.py but generates GitHub search queries instead of LinkedIn Booleans.
The brief is the primary input. Opus synthesizes GitHub API queries across four channels:
user search, code search, repo mining, and org exploration.

Two main functions:
1. form_github_strategy() — ONE Opus call to generate GitHub search queries from brief
2. adapt_after_batch() — ONE Opus call after each batch to adapt based on results
"""

from __future__ import annotations

import json
import sys
from typing import Optional

from github.schemas import GitHubSearchQuery, GitHubBatchReport
from github.query_validator import validate_batch
from github.hubs.derive import derive_registry_targets
from github.rosters import derive_feedstock_repos
import github.health as _health_module
from shared.adaptive import (
    AdaptiveAction,
    AdaptationDecision,
    ChannelExhaustion,
    NoiseMarker,
    ScoutMetrics,
    SignalMarker,
)
from shared.llm_clients import opus_llm_cached
from shared.brief_loader import Brief
from shared.resolvers.ecosystems import REGISTRY_ALIASES
from shared.schemas import ExecutionPlan
import shared.config as _config


_REGISTRY_PACKAGE_CAP = 25
_ROSTER_REPO_CAP = _REGISTRY_PACKAGE_CAP
# python/pypi/conda tokens gate conda-forge feedstock roster queries.
_ROSTER_PYTHON_STACK_ALIASES: dict[str, str] = {
    "python": "python",
    "pypi": "python",
    "pypi.org": "python",
    "conda": "python",
    "conda-forge": "python",
}
# Attribute names resolved at call time via ``_health_module`` — import-time
# binding of function objects defeated ``unittest.mock.patch`` on the probe seam.
_REGISTRY_PROBE_ATTR_BY_ECOSYSTEM = {
    "npmjs.org": "probe_npm_registry",
    "crates.io": "probe_crates_registry",
}


def _probe_registry_hub(ecosystem: str) -> bool:
    attr = _REGISTRY_PROBE_ATTR_BY_ECOSYSTEM.get(ecosystem)
    if attr is None:
        return False
    probe_fn = getattr(_health_module, attr)
    return bool(probe_fn())


# ---------------------------------------------------------------------------
# Strategy formation (run start)
# ---------------------------------------------------------------------------

def form_github_strategy(
    brief: Brief,
    prior_run_data: Optional[dict] = None,
) -> tuple[list[GitHubSearchQuery], str]:
    """Ask Opus to generate GitHub search queries from the sourcing brief.

    Returns (queries, rationale) tuple.
    """
    system = _build_strategy_system(brief)
    user_prompt = _build_strategy_user(brief, prior_run_data)

    try:
        usage_context = {
            "stage": "github_strategy",
            "source": "github",
            "brief_id": getattr(brief, "id", None),
            "role_title": getattr(brief, "role_title", ""),
        }
        result = opus_llm_cached(
            system,
            user_prompt,
            expect_json=True,
            max_tokens=16384,
            usage_context=usage_context,
            model_name=_config.STRATEGY_MODEL_NAME,
        )
    except Exception as e:
        queries = _default_queries(brief)
        queries, _validation_results = validate_batch(queries, brief, set())
        if not queries:
            raise RuntimeError(
                "strategy formation failed and no fallback queries are derivable from the brief"
            ) from e
        return queries, f"Fallback: {e}"

    rationale = result.get("strategy_rationale", "")
    queries = _parse_strategy_response(result)

    # Validate and repair LLM-generated queries
    queries, _validation_results = validate_batch(queries, brief, set())

    # OSS Maintainers Slice 7: target-project seeding. When the
    # recruiter named explicit ``target_projects`` (per OSS
    # Maintainers Module Spec §8), seed acquisition queries from
    # them — those are recruiter-authoritative.
    # Behavior-preserving for classic github briefs: empty
    # ``target_projects`` ⇒ no extra queries.
    next_id = max((q.id for q in queries), default=0) + 1
    next_id = _append_target_project_queries(brief, queries, next_id)
    next_id = _append_registry_queries(brief, queries, next_id)
    next_id = _append_roster_queries(brief, queries, next_id)

    return queries, rationale


# ---------------------------------------------------------------------------
# Registry adapter (Multi-Agent Execution Plan Slice 1.6)
# ---------------------------------------------------------------------------

def form_strategy_for_registry(
    brief: Brief,
    prior_run_data: Optional[dict] = None,
) -> ExecutionPlan:
    """Uniform-signature adapter wrapping :func:`form_github_strategy`.

    The native callable returns ``(list[GitHubSearchQuery], str)`` —
    a tuple shape that predates the multi-module ExecutionPlan
    contract. The launcher-registry seam expects
    :class:`shared.schemas.ExecutionPlan`, so the adapter normalizes:
    rationale → ``strategy_rationale``; queries serialized via
    :meth:`GitHubSearchQuery.to_dict` → ``generated_strings``.

    Defaults match the orchestrator's call site
    (``github/orchestrator.py:229``): brief-driven seeding only via
    ``target_projects``. Native callers (orchestrator) keep
    calling :func:`form_github_strategy` directly and are unaffected.
    """

    queries, rationale = form_github_strategy(brief, prior_run_data)
    plan = ExecutionPlan(
        strategy_rationale=rationale,
        generated_strings=[q.to_dict() for q in queries],
    )
    from shared.role_strategy import apply_role_strategy_to_plan

    apply_role_strategy_to_plan(brief, plan, merge_lane_templates=False)
    return plan


def _brief_example_location(brief: Brief) -> str | None:
    permanent_filters = getattr(brief, "permanent_filters", None)
    if isinstance(permanent_filters, dict):
        location = str(permanent_filters.get("Location") or "").strip()
        if location:
            return location
    raw = getattr(brief, "raw", None) or {}
    if isinstance(raw, dict):
        geography = str(raw.get("geography") or "").strip()
        if geography:
            return geography
    return None


def _brief_example_key_term(brief: Brief) -> str:
    new_brief = getattr(brief, "_new_brief", None)
    if new_brief is not None:
        capability_areas = getattr(new_brief, "capability_areas", None) or []
        for area in capability_areas:
            name = getattr(area, "name", None)
            if name and str(name).strip():
                return str(name).strip()
    role_title = str(getattr(brief, "role_title", "") or "").strip()
    if role_title:
        tokens = [token for token in role_title.split() if len(token) >= 3]
        if tokens:
            return tokens[0]
        return role_title
    return "platform"


def _brief_example_code_signal(brief: Brief) -> str | None:
    new_brief = getattr(brief, "_new_brief", None)
    if new_brief is not None:
        for area in getattr(new_brief, "capability_areas", None) or []:
            signals = getattr(area, "github_code_signals", None) or []
            for signal in signals:
                if isinstance(signal, str) and signal.strip():
                    return signal.strip()
    return None


def _tokenize_role_title_for_search(role_title: str) -> str:
    tokens = [token for token in role_title.split() if len(token) >= 3]
    return " ".join(tokens[:4])


def _build_strategy_system(brief: Brief) -> str:
    noise_section = ""
    if brief.noise_archetypes:
        noise_section = f"\n## Noise Archetypes (people to avoid)\n{json.dumps(brief.noise_archetypes, indent=2)}"

    toolchain_section = ""
    new_brief = getattr(brief, "_new_brief", None)
    if new_brief is not None and getattr(new_brief, "capability_areas", None):
        code_signal_lines: list[str] = []
        for area in new_brief.capability_areas:
            signals = getattr(area, "github_code_signals", None) or []
            if signals:
                code_signal_lines.append(
                    f"  {area.name}: {', '.join(signals)}"
                )
        if code_signal_lines:
            toolchain_section = (
                "\n## Code Signals (practitioner fingerprints)\n"
                + "\n".join(code_signal_lines)
                + "\n"
            )

    example_location = _brief_example_location(brief)
    example_key_term = _brief_example_key_term(brief)
    example_code_signal = _brief_example_code_signal(brief)
    location_filter_doc = (
        f"- `location:{example_location}` or `location:\"Major City\"` — profile location (free text, ~30% fill rate)"
        if example_location
        else "- `location:Country` or `location:\"Major City\"` — profile location (free text, ~30% fill rate; omit when the brief has no geography)"
    )
    location_segment_high = (
        f"`language:python location:{example_location} followers:>100` (narrower, likely <1000)"
        if example_location
        else "`language:python followers:>100` (narrower, likely <1000)"
    )
    location_segment_mid = (
        f"`language:python location:{example_location} followers:50..100` (narrower)"
        if example_location
        else "`language:python followers:50..100` (narrower)"
    )
    location_segment_low = (
        f"`language:python location:{example_location} followers:20..50 created:>2020-01-01` (very narrow)"
        if example_location
        else "`language:python followers:20..50 created:>2020-01-01` (very narrow)"
    )
    user_search_example_query = (
        f"language:python location:{example_location} followers:>50 {example_key_term}"
        if example_location
        else f"language:python followers:>50 {example_key_term}"
    )
    if example_code_signal:
        code_search_example_query = f'\\"{example_code_signal}\\" language:python'
        code_search_example_note = (
            "Why this code pattern signals domain practitioners"
        )
    else:
        code_search_example_query = '\\"import <package>\\" language:python'
        code_search_example_note = (
            "Substitute a real code signal from the brief — never emit placeholder tokens"
        )
    topic_slug = example_key_term.lower().replace(" ", "-")
    topic_search_example_query = f"topic:{topic_slug} language:python stars:>20"

    return f"""You are a senior technical sourcing strategist planning a GitHub-based candidate search.

Role: {brief.role_title}
{brief.role_description}

## Minimum Bar
{brief.minimum_bar}

## Archetypes (target candidate profiles)
{json.dumps(brief.archetypes, indent=2)}
{noise_section}
{toolchain_section}
## Geography
{json.dumps(brief.permanent_filters, indent=2)}

## Your Task

Generate GitHub search queries across FIVE channels to find candidates matching this role.
GitHub is NOT LinkedIn — the signal comes from what people BUILD, not their job titles.

### Channel 1: User Search Queries
GitHub user search syntax: `GET /search/users?q=...`

Available filters:
- `language:python` — primary programming language
{location_filter_doc}
- `followers:>50` or `followers:10..100` — follower count range
- `repos:>10` — minimum public repo count
- `created:>2015-01-01` or `created:2018..2022` — account creation date
- `type:user` — exclude orgs
- Free text terms match against username, name, bio, and README

CRITICAL CONSTRAINT: GitHub search returns max 1,000 results per query.
For broad queries that would return >1,000, you MUST segment by follower ranges or date ranges.

Example segmentation:
- {location_segment_high}
- {location_segment_mid}
- {location_segment_low}

Generate 15-30 user search queries covering:
- Geographic segments (by location variants: country, major cities)
- Skill segments (by language: Python, Rust, C++, Julia)
- Seniority segments (by follower/repo count bands)
- Topic segments (free text terms in bio drawn from the brief's capability areas and key terms)

### Channel 2: Code Search Queries
GitHub code search syntax: `GET /search/code?q=...`

Searches code content inside repositories. Returns repos, not users — we extract contributors afterward.
VERY rate-limited: 10 requests/minute. Generate 5-10 high-value queries only.

Available filters:
- `language:python` — file language
- `extension:py` — file extension
- `repo:owner/name` — limit to specific repo
- `path:src/` — limit to path
- Free text matches code content

Best for: finding practitioners by what they've actually built. Search for specific imports,
function calls, or config patterns that only real practitioners would have.

Examples:
- `"from <package> import" language:python` — people using a brief-named library
- `"<TrainerClass>" extension:py` — people implementing a training loop
- `"<eval_harness>" "train" language:python` — evaluation harness authors
- `"<config_key>" language:python` — practitioners with domain-specific configs

Generate code search queries that surface repos with domain-specific code.
Each repo's top contributors become candidates.

### Channel 3: Repo Topic Search Queries
GitHub repo search syntax: `GET /search/repositories?q=...`

Search for repos tagged with capability-area topics. Extract owner usernames as candidates.
High signal — people who tag repos with brief-relevant topics are self-identifying as practitioners.

Available filters:
- `topic:reinforcement-learning` — repo topics
- `language:python` — primary language
- `stars:>20` — minimum stars (filters noise)
- `pushed:>2024-01-01` — recently active
- Free text matches repo name and description

Generate 5-10 topic search queries targeting repos in capability area domains.

### Tapped-Market / Edge-Case Opening

If the brief says the obvious pool is tapped, exhausted, or already heavily worked, then your opening GitHub queries should prioritize non-obvious adjacent builders rather than the canonical framework crowd.

Use this mental loop:
1. First ask what a strong but standard technical sourcer would search on GitHub for this role.
2. Then ask which same-caliber builders that standard pass would miss because their repos emphasize product/problem language, delivery tooling, internal platforms, or adjacent systems work instead of canonical framework names.
3. Generate your opening queries primarily for those missed populations.

For the initial query slate:
- Prefer profiles and repos that signal delivery accelerators, reference architectures, eval harnesses, tracing/observability, internal tools, product/problem-language AI systems, or consultancy/vertical-SaaS builders
- Do NOT over-concentrate the opening set on exact canonical framework imports or frontier-brand clusters alone
- Treat direct framework-name and obvious frontier-repo mining as cleanup/completion passes, not the only opening move

NOVELTY ACCOUNTING:
- In a tapped market, direct framework-import hits and frontier-brand repo hits are useful confirmation but low-novelty signal.
- Do not treat a strong yield from those obvious pools as proof the opening query mix is correct.
- Aim for the opening slate to surface adjacent but same-caliber builders whose repos emphasize delivery tooling, product/problem language, internal platforms, or reusable implementation patterns.

### Channel 4: Stargazer Mining — UNAVAILABLE
GitHub restricted the public starring API (2026-06): stargazer lists for repositories
you do not collaborate on now return 404 even with valid authentication (probed live
2026-07-31). Do NOT emit stargazer_repos — return it as an empty list. Redistribute
that coverage into the other channels.

### Channel 5: Seed Experts for Graph Expansion
Nominate 3-5 known experts in the target domain whose followers/following should be mined.
These should be well-known figures in the specific capability areas named in the brief. Their
social graph will be expanded to discover practitioners who are invisible to keyword search.

### Output Format

Return JSON:
{{
    "strategy_rationale": "Overall approach explanation",
    "user_search_queries": [
        {{
            "query": "{user_search_example_query}",
            "name": "{example_key_term} practitioners ({'geo-scoped' if example_location else 'global'})",
            "rationale": "Why this query targets the right population"
        }}
    ],
    "code_search_queries": [
        {{
            "query": "{code_search_example_query}",
            "name": "Brief code-signal users",
            "rationale": "{code_search_example_note}"
        }}
    ],
    "topic_search_queries": [
        {{
            "query": "{topic_search_example_query}",
            "name": "{example_key_term} topic repos",
            "rationale": "Why this topic combination targets practitioners"
        }}
    ],
    "stargazer_repos": [
        {{
            "repo": "owner/repo-from-brief",
            "rationale": "Why stargazers of this repo are high-signal candidates"
        }}
    ],
    "seed_experts": [
        {{
            "username": "expert_username",
            "rationale": "Why this person's network contains candidates"
        }}
    ],
    "coverage_gaps": ["Populations this strategy might miss"],
    "noise_predictions": ["Expected false positive patterns"]
}}

Guidelines:
- Bio text search is fuzzy — use distinctive brief-supplied terms, not generic ones ("AI", "software")
- Location field is free text — include country name AND major cities as separate queries
- Portuguese/Spanish bios are common in LATAM — include both English and local-language terms where relevant
- Follower count correlates with seniority but not perfectly — don't over-filter
- Account age filters help target experienced engineers (created before 2020) vs newer accounts

Return valid JSON only."""


def _build_strategy_user(
    brief: Brief,
    prior_run_data: Optional[dict] = None,
) -> str:
    prompt = ""

    if brief.jd_text:
        prompt += f"## Job Description\n{brief.jd_text}\n\n"

    if brief.intake_notes:
        prompt += f"## Intake Notes\n{brief.intake_notes}\n\n"

    if prior_run_data:
        prompt += f"## Prior Run Data\n{json.dumps(prior_run_data, indent=2)}\n\n"

    if brief.search_priorities:
        prompt += f"## Search Priorities\n{', '.join(brief.search_priorities)}\n\n"

    if brief.additional_search_terms:
        prompt += (
            "## Additional Search Terms\n"
            "These terms should be used for search query generation but are NOT evaluation criteria:\n"
            f"{', '.join(brief.additional_search_terms)}\n\n"
        )

    if brief.instructions:
        prompt += "## Sourcing Instructions\n" + "\n".join(f"- {i}" for i in brief.instructions) + "\n\n"

    prompt += "Generate GitHub search queries for this role."
    return prompt


def _parse_strategy_response(result: dict) -> list[GitHubSearchQuery]:
    """Parse Opus strategy response into GitHubSearchQuery objects."""
    queries = []
    query_id = 1

    for uq in result.get("user_search_queries", []):
        queries.append(GitHubSearchQuery(
            id=query_id,
            name=uq.get("name", f"User search #{query_id}"),
            query=uq.get("query", ""),
            channel="user_search",
        ))
        query_id += 1

    for cq in result.get("code_search_queries", []):
        queries.append(GitHubSearchQuery(
            id=query_id,
            name=cq.get("name", f"Code search #{query_id}"),
            query=cq.get("query", ""),
            channel="code_search",
        ))
        query_id += 1

    for tq in result.get("topic_search_queries", []):
        queries.append(GitHubSearchQuery(
            id=query_id,
            name=tq.get("name", f"Topic search #{query_id}"),
            query=tq.get("query", ""),
            channel="topic_search",
        ))
        query_id += 1

    # Stargazer lane disabled fail-closed (2026-07-31): the public starring
    # API returns 404 for non-collaborator tokens (GitHub restriction,
    # 2026-06; probed live with a valid authenticated token). Queuing these
    # would record zero-result queries and teach the adaptation layer a false
    # channel-exhaustion signal. Drop with a loud log; the dispatch handler
    # stays for legacy resumes.
    dropped_stargazers = [
        sr.get("repo", "") for sr in result.get("stargazer_repos", [])
    ]
    if dropped_stargazers:
        print(
            "    [strategy] stargazer lane unavailable (GitHub API restriction) — "
            f"dropped {len(dropped_stargazers)} suggested repo(s): "
            f"{', '.join(dropped_stargazers[:5])}",
            file=sys.stderr,
        )

    for se in result.get("seed_experts", []):
        queries.append(GitHubSearchQuery(
            id=query_id,
            name=f"Graph expansion: {se.get('username', '')}",
            query=se.get("username", ""),
            channel="graph_expansion",
        ))
        query_id += 1

    return queries


def _default_queries(brief: Brief) -> list[GitHubSearchQuery]:
    """Fallback queries when strategy formation fails."""
    queries: list[GitHubSearchQuery] = []
    query_id = 1

    query_id = _append_target_project_queries(brief, queries, query_id)

    location = ""
    permanent_filters = getattr(brief, "permanent_filters", None)
    if isinstance(permanent_filters, dict):
        location = str(permanent_filters.get("Location") or "").strip()
    if not location:
        raw = getattr(brief, "raw", None) or {}
        if isinstance(raw, dict):
            location = str(raw.get("geography") or "").strip()

    role_title = str(getattr(brief, "role_title", "") or "").strip()
    if role_title:
        title_tokens = _tokenize_role_title_for_search(role_title)
        if title_tokens:
            query_parts = [title_tokens, "type:user"]
            if location:
                query_parts.insert(1, f"location:{location}")
            queries.append(
                GitHubSearchQuery(
                    id=query_id,
                    name=f"Fallback: {title_tokens}" + (f" in {location}" if location else ""),
                    query=" ".join(query_parts),
                    channel="user_search",
                )
            )

    return queries


def _append_target_project_queries(
    brief: Brief,
    queries: list[GitHubSearchQuery],
    next_id: int,
) -> int:
    """Seed acquisition queries from ``brief.target_projects`` (OSS Maintainers Slice 7).

    For each ``owner/repo`` in ``target_projects``:

    - One ``repo_mining`` query (contributor channel on the named
      repo). Surfaces candidates with attributable activity on the
      project — the maintainership classifier (Slice 4) then keys
      its merge-authority / release / governance signals to this
      project.

    Stargazer mining for target projects is disabled fail-closed
    (same rationale as :func:`_parse_strategy_response` at
    strategy.py:465-471): queuing would record zero-result queries
    and teach adaptation a false channel-exhaustion signal.

    Behavior-preserving for classic briefs: empty ``target_projects``
    ⇒ no queries appended; returns ``next_id`` unchanged.
    Deduplicates against existing ``target_repo`` entries so
    LLM-emitted queries can't collide with target-project seeds (the
    maintainership classifier re-keys signals per project so duplicates
    are wasteful, not harmful).
    """

    target_projects = list(getattr(brief, "target_projects", []) or [])
    if not target_projects:
        return next_id

    # Build the existing-target set so we don't double-seed when
    # the LLM strategy already named the same repo. Compare
    # case-insensitively to match GitHub's resolution.
    existing_targets = {
        (q.target_repo or "").strip().lower()
        for q in queries
        if (q.target_repo or "").strip()
    }

    dropped_stargazer_targets: list[str] = []

    for raw in target_projects:
        if not isinstance(raw, str):
            continue
        target = raw.strip()
        if not target or target.lower() in existing_targets:
            continue
        existing_targets.add(target.lower())

        # Repo mining (contributor channel).
        queries.append(
            GitHubSearchQuery(
                id=next_id,
                name=f"Target project — mine contributors: {target}",
                query="",
                channel="repo_mining",
                target_repo=target,
            )
        )
        next_id += 1
        dropped_stargazer_targets.append(target)

    if dropped_stargazer_targets:
        print(
            "    [strategy] stargazer lane unavailable (GitHub API restriction) — "
            f"dropped {len(dropped_stargazer_targets)} target-project repo(s): "
            f"{', '.join(dropped_stargazer_targets[:5])}",
            file=sys.stderr,
        )

    return next_id


def _append_registry_queries(
    brief: Brief,
    queries: list[GitHubSearchQuery],
    next_id: int,
) -> int:
    """Seed registry-maintainer discovery queries at the front of the query list.

    Declared maintainership is the module's highest-precision channel; running
    it first means its people are judged WITH their declared evidence instead
    of being re-found terminal (architecture doc §7 precision-over-volume).
    """

    registry_queries: list[GitHubSearchQuery] = []

    for target in derive_registry_targets(brief):
        seeds = [
            package
            for package in target.seed_packages
            if package and not package.startswith((".", "/"))
        ]

        if not seeds:
            print(
                "    [strategy] registry maintainers — "
                f"{target.ecosystem}: ecosystem selected by target_stacks but "
                "no seed packages derivable from github_code_signals",
                file=sys.stderr,
            )
            continue

        if not _probe_registry_hub(target.ecosystem):
            print(
                "    [strategy] registry hub unreachable — "
                f"dropped {target.ecosystem} maintainers query (probe failed)",
                file=sys.stderr,
            )
            continue

        packages = list(seeds)
        if len(packages) > _REGISTRY_PACKAGE_CAP:
            dropped_tail = packages[_REGISTRY_PACKAGE_CAP:]
            packages = packages[:_REGISTRY_PACKAGE_CAP]
            print(
                "    [strategy] registry maintainers — "
                f"capped {target.ecosystem} seed packages at "
                f"{_REGISTRY_PACKAGE_CAP} (dropped tail: "
                f"{', '.join(dropped_tail[:5])}"
                f"{'…' if len(dropped_tail) > 5 else ''})",
                file=sys.stderr,
            )

        preview = ", ".join(packages[:5])
        if len(packages) > 5:
            preview = f"{preview}, …"

        registry_queries.append(
            GitHubSearchQuery(
                id=next_id,
                name=f"Registry maintainers — {target.ecosystem}: {preview}",
                query="",
                channel="registry_maintainer_discovery",
                target_ecosystem=target.ecosystem,
                target_packages=packages,
            )
        )

    if not registry_queries:
        return next_id

    queries[0:0] = registry_queries
    for idx, query in enumerate(queries, start=1):
        query.id = idx
    return len(queries) + 1


def _brief_has_python_family_stack(brief: Brief) -> bool:
    """True when ``brief.target_stacks`` names a python-family ecosystem."""
    for stack in getattr(brief, "target_stacks", None) or []:
        if not isinstance(stack, str):
            continue
        key = stack.strip().lower()
        if not key:
            continue
        if _ROSTER_PYTHON_STACK_ALIASES.get(key) == "python":
            return True
        if REGISTRY_ALIASES.get(key) == "pypi.org":
            return True
    return False


def _cap_roster_repos(repos: list[str], *, label: str) -> list[str]:
    if len(repos) <= _ROSTER_REPO_CAP:
        return repos
    dropped_tail = repos[_ROSTER_REPO_CAP:]
    capped = repos[:_ROSTER_REPO_CAP]
    print(
        "    [strategy] roster ingest — "
        f"capped {label} repos at {_ROSTER_REPO_CAP} (dropped tail: "
        f"{', '.join(dropped_tail[:5])}"
        f"{'…' if len(dropped_tail) > 5 else ''})",
        file=sys.stderr,
    )
    return capped


def _append_roster_queries(
    brief: Brief,
    queries: list[GitHubSearchQuery],
    next_id: int,
) -> int:
    """Seed roster ingest queries after registry queries at the front of the queue.

    Brief-derived sources only: ``brief.target_projects`` and
    :func:`derive_feedstock_repos` (gated on python-family ``target_stacks``).
    Emits at most two batched queries — target projects and feedstocks — with
    repos carried in ``target_packages``. No hub probe — roster fetch rides the
    GitHub API, whose availability the existing GitHub readiness probe already
    gates.
    """

    seen: set[str] = set()
    project_repos: list[str] = []

    for raw in getattr(brief, "target_projects", []) or []:
        if not isinstance(raw, str):
            continue
        target = raw.strip()
        if not target:
            continue
        key = target.lower()
        if key in seen:
            continue
        seen.add(key)
        project_repos.append(target)

    feedstock_repos: list[str] = []
    if _brief_has_python_family_stack(brief):
        for feedstock in derive_feedstock_repos(brief):
            key = feedstock.lower()
            if key in seen:
                continue
            seen.add(key)
            feedstock_repos.append(feedstock)

    roster_queries: list[GitHubSearchQuery] = []

    if project_repos:
        repos = _cap_roster_repos(project_repos, label="target projects")
        preview = ", ".join(repos[:5])
        if len(repos) > 5:
            preview = f"{preview}, …"
        roster_queries.append(
            GitHubSearchQuery(
                id=next_id,
                name=f"Roster ingest — target projects: {preview}",
                query="",
                channel="roster_ingest",
                # structured-channel payload list — repos for roster_ingest
                target_packages=repos,
            )
        )
        next_id += 1

    if feedstock_repos:
        repos = _cap_roster_repos(feedstock_repos, label="feedstock")
        preview = ", ".join(repos[:5])
        if len(repos) > 5:
            preview = f"{preview}, …"
        roster_queries.append(
            GitHubSearchQuery(
                id=next_id,
                name=f"Roster ingest — feedstocks: {preview}",
                query="",
                channel="roster_ingest",
                # structured-channel payload list — repos for roster_ingest
                target_packages=repos,
            )
        )
        next_id += 1

    if not roster_queries:
        return next_id

    registry_count = 0
    for query in queries:
        if query.channel == "registry_maintainer_discovery":
            registry_count += 1
        else:
            break

    queries[registry_count:registry_count] = roster_queries
    for idx, query in enumerate(queries, start=1):
        query.id = idx
    return len(queries) + 1


# ---------------------------------------------------------------------------
# Adaptation after batch completion
# ---------------------------------------------------------------------------

def adapt_after_batch(
    brief: Brief,
    batch_report: GitHubBatchReport,
    remaining_queries: list[GitHubSearchQuery],
    executed_queries: set[str] | None = None,
    exhaustion_context: str = "",
) -> tuple[list[GitHubSearchQuery], str, list[int]]:
    """Ask Opus to adapt after a batch of queries — generate new queries from results.

    Returns (new_queries, rationale, skipped_ids) tuple.
    """
    geo = brief.permanent_filters.get("Location", "")
    geo_constraint = ""
    if geo:
        geo_constraint = f"""
GEOGRAPHY CONSTRAINT: This search is restricted to {geo}. ALL new user_search queries MUST include 'location:{geo}' in the query string. Do NOT generate queries targeting other geographies (India, Europe, etc.). Code search, repo mining, stargazer mining, and topic search are global by design — the geo filter is applied post-hoc — but user_search queries MUST be geo-scoped."""

    example_location = geo or _brief_example_location(brief)
    example_key_term = _brief_example_key_term(brief)
    example_code_signal = _brief_example_code_signal(brief)
    role_title_tokens = _tokenize_role_title_for_search(
        str(getattr(brief, "role_title", "") or "")
    )
    user_search_valid_1 = (
        f"language:python location:{example_location} followers:>50"
        if example_location
        else "language:python followers:>50"
    )
    user_search_valid_2 = (
        f'"{example_key_term}" location:{example_location} language:python'
        if example_location
        else f'"{example_key_term}" language:python'
    )
    user_search_valid_4 = (
        f'"{role_title_tokens or example_key_term}" location:{example_location} followers:20..100'
        if example_location
        else f'"{role_title_tokens or example_key_term}" followers:20..100'
    )
    if example_code_signal:
        code_search_valid_1 = f'"{example_code_signal}" language:python'
    else:
        code_search_valid_1 = '\\"import <package>\\" language:python'
    topic_slug = example_key_term.lower().replace(" ", "-")
    topic_search_valid_1 = f"topic:{topic_slug} language:python stars:>20"
    topic_search_valid_2 = f"topic:{topic_slug} pushed:>2024-01-01"

    exhaustion_section = ""
    if exhaustion_context:
        exhaustion_section = f"""

## Channel Exhaustion Status
{exhaustion_context}

Do NOT generate new queries for EXHAUSTED channels. For DEGRADED channels,
only generate queries that meaningfully differ from previous attempts."""

    system = f"""You are a sourcing strategist adapting a GitHub search plan mid-run.

Role: {brief.role_title}
{brief.role_description}
{geo_constraint}

You've received a report on a batch of completed GitHub searches. Based on the results:
1. Generate NEW GitHub search queries that target signal patterns you observed
2. Identify remaining queued queries to skip (redundant or similar to zero-save queries)

## CRITICAL: GitHub Search API Syntax

Every query you generate must be a valid GitHub search API `q` parameter string,
NOT a natural language description.

### Valid user_search examples:
- `{user_search_valid_1}`
- `{user_search_valid_2}`
- `language:rust location:"São Paulo" repos:>10`
- `{user_search_valid_4}`

### INVALID user_search examples (DO NOT generate these):
- `ML ultra-low followers highly active` ← natural language, not API syntax
- `experienced Python developers in São Paulo` ← no qualifiers
- `ML exact pattern that saved v3` ← meta-description, not a query

### Valid code_search examples:
- `{code_search_valid_1}`
- `"TrainerClass" extension:py`
- `"eval-harness" "train" language:python`

### Valid topic_search examples:
- `{topic_search_valid_1}`
- `{topic_search_valid_2}`

Focus on:
- Queries that produced saves → generate adjacent/deeper queries in the same vein
- Languages and repos common in saved candidates → mine those repos, search those language+topic combos
- Queries that hit the 1,000 result cap → suggest narrower segmentations
- Zero-save queries → avoid similar patterns
- If the brief says the obvious pool is tapped, prefer extending productive edge-case populations before adding more canonical framework-name or frontier-brand cleanup queries
- Keep asking which same-caliber builders a standard technical sourcer would still miss, and bias new queries toward those adjacent populations first
{exhaustion_section}

Return JSON:
{{
    "new_user_queries": [
        {{"query": "...", "name": "...", "rationale": "..."}}
    ],
    "new_code_queries": [
        {{"query": "...", "name": "...", "rationale": "..."}}
    ],
    "new_topic_queries": [
        {{"query": "...", "name": "...", "rationale": "..."}}
    ],
    "new_stargazer_repos": [],  # ALWAYS empty — stargazer API restricted (2026-06), lane disabled
    "new_repos_to_mine": ["owner/repo"],
    "skip_query_ids": [1, 2],
    "rationale": "Overall adaptation reasoning"
}}

Return valid JSON only."""

    remaining_text = "\n".join(
        f"  #{q.id} [{q.channel}]: {q.name}" for q in remaining_queries
    )

    user_prompt = f"""{batch_report.to_summary_text()}

## Remaining Queued Queries ({len(remaining_queries)})
{remaining_text}

Suggest adaptations."""

    try:
        usage_context = {
            "stage": "github_batch_adaptation",
            "source": "github",
            "brief_id": getattr(brief, "id", None),
            "remaining_query_count": len(remaining_queries),
        }
        result = opus_llm_cached(
            system,
            user_prompt,
            expect_json=True,
            usage_context=usage_context,
            model_name=_config.STRATEGY_MODEL_NAME,
        )
    except Exception as e:
        return [], f"Adaptation failed: {e}", []

    # Parse new queries. Registry maintainer discovery and roster ingest are
    # v1 no-ops for adaptation — neither create nor mutate those queries here.
    new_queries = []
    next_id = max((q.id for q in remaining_queries), default=100) + 1

    for uq in result.get("new_user_queries", []):
        new_queries.append(GitHubSearchQuery(
            id=next_id,
            name=uq.get("name", f"Adapted user search #{next_id}"),
            query=uq.get("query", ""),
            channel="user_search",
        ))
        next_id += 1

    for cq in result.get("new_code_queries", []):
        new_queries.append(GitHubSearchQuery(
            id=next_id,
            name=cq.get("name", f"Adapted code search #{next_id}"),
            query=cq.get("query", ""),
            channel="code_search",
        ))
        next_id += 1

    for repo in result.get("new_repos_to_mine", []):
        new_queries.append(GitHubSearchQuery(
            id=next_id,
            name=f"Mine contributors: {repo}",
            query="",
            channel="repo_mining",
            target_repo=repo,
        ))
        next_id += 1

    for tq in result.get("new_topic_queries", []):
        new_queries.append(GitHubSearchQuery(
            id=next_id,
            name=tq.get("name", f"Adapted topic search #{next_id}"),
            query=tq.get("query", ""),
            channel="topic_search",
        ))
        next_id += 1

    # Stargazer lane disabled fail-closed (2026-07-31) — same rationale as
    # the strategy-formation drop above.
    dropped = [r for r in result.get("new_stargazer_repos", []) if r]
    if dropped:
        print(
            "    [adaptation] stargazer lane unavailable — dropped "
            f"{len(dropped)} suggested repo(s)",
            file=sys.stderr,
        )

    # Cap adaptation output at 10 queries per cycle
    if len(new_queries) > 10:
        new_queries = new_queries[:10]

    # Validate adapted queries
    new_queries, _validation_results = validate_batch(
        new_queries, brief, executed_queries or set()
    )

    # Mark skipped queries
    skip_ids = list(result.get("skip_query_ids", []))
    for q in remaining_queries:
        if q.id in set(skip_ids):
            q.status = "skipped"
            q.notes = "Skipped by adaptation"

    rationale = result.get("rationale", "")
    return new_queries, rationale, skip_ids


def build_github_adaptation_decision(
    *,
    batch_report: GitHubBatchReport,
    new_queries: list[GitHubSearchQuery],
    rationale: str,
    skipped_ids: list[int],
) -> AdaptationDecision:
    """Translate GitHub-native adaptation output into the shared event contract."""

    action = _classify_github_adaptive_action(batch_report, new_queries, skipped_ids)
    channels = sorted(
        {
            str(detail.get("channel"))
            for detail in batch_report.query_details
            if detail.get("channel")
        }
    )
    dominant_channel = _dominant_channel(batch_report, channels)
    inserted_ids = [str(query.id) for query in new_queries]
    metrics = ScoutMetrics(
        work_units_run=batch_report.queries_run,
        candidates_discovered=batch_report.total_candidates_discovered,
        saves=batch_report.total_saves,
        rejects=batch_report.total_rejects,
        insufficient=batch_report.total_insufficient,
        signal_markers=[
            SignalMarker.from_dict(marker)
            for marker in batch_report.signal_markers
            if isinstance(marker, dict)
        ],
        noise_markers=[
            NoiseMarker.from_dict(marker)
            for marker in batch_report.noise_markers
            if isinstance(marker, dict)
        ],
        exhaustion=[
            ChannelExhaustion.from_dict(marker)
            for marker in batch_report.exhaustion_markers
            if isinstance(marker, dict)
        ],
    )
    source_payload = {
        "new_queries": [query.to_dict() for query in new_queries],
        "skip_query_ids": skipped_ids,
        "channel_metrics": batch_report.channel_metrics,
        "top_performing_queries": batch_report.top_performing_queries,
        "zero_save_query_ids": batch_report.zero_save_query_ids,
        "queries_hitting_result_cap": batch_report.queries_hitting_result_cap,
        # Keep the full channel set in source_payload so recruiters can still
        # see every channel touched in a batch — the contract surfaces only
        # the dominant channel as the lane for cross-source rollups.
        "channels_in_batch": channels,
    }
    return AdaptationDecision(
        source="github",
        action=action,
        lane=dominant_channel,
        rationale=rationale,
        metrics=metrics,
        work_unit_kind="github_query",
        work_unit_family=dominant_channel,
        inserted_work_units=inserted_ids,
        skipped_work_units=[str(query_id) for query_id in skipped_ids],
        source_payload=source_payload,
    )


def _dominant_channel(batch_report: GitHubBatchReport, channels: list[str]) -> str:
    """Pick a single representative channel for the shared lane vocabulary.

    Cross-source rollups expect ``lane`` to be one token, not a CSV. Use
    the channel with the highest save count; fall back to the first
    alphabetically-sorted channel; fall back to ``"github"`` if no channels
    were touched.
    """

    if not channels:
        return "github"
    save_by_channel: dict[str, int] = {}
    for rollup in batch_report.channel_metrics:
        if isinstance(rollup, dict):
            name = str(rollup.get("channel") or "")
            save_by_channel[name] = int(rollup.get("saves") or 0)
    best = max(
        channels,
        key=lambda channel: (save_by_channel.get(channel, 0), -channels.index(channel)),
    )
    return best


def _classify_github_adaptive_action(
    batch_report: GitHubBatchReport,
    new_queries: list[GitHubSearchQuery],
    skipped_ids: list[int],
) -> AdaptiveAction:
    if skipped_ids and not new_queries:
        return AdaptiveAction.SKIP
    if batch_report.total_saves > 0 and new_queries:
        return AdaptiveAction.EXPERIMENT
    if batch_report.total_saves > 0:
        return AdaptiveAction.COMMIT
    if batch_report.total_candidates_discovered == 0 and new_queries:
        return AdaptiveAction.BROADEN
    if skipped_ids:
        return AdaptiveAction.NARROW
    if new_queries:
        return AdaptiveAction.EXPERIMENT
    return AdaptiveAction.CONTINUE
