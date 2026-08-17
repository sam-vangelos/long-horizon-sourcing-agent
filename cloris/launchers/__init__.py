"""Per-source launcher registry — Phase F Slice F1.

The single source-of-truth that maps a `source` string (``"linkedin"``,
``"github"``, …) to the runtime callables a worker spawn needs:

- ``state_key_fn(brief_path) -> str`` — derive the canonical state-key
  from the brief content (NOT path), so the key stays stable across
  flat→nested brief migrations.
- ``state_dir_fn(brief_path) -> Path`` — resolve the on-disk state
  directory (``output/state/<source>/<state_key>``).
- ``orchestrator_argv_fn(brief_path, state_dir, *, resume, fresh=False) -> list[str]``
  — compose the argv ``cloris.worker`` execvp's into. This is the
  source-specific seam; everything else upstream is generic.

What the registry deliberately does NOT carry:

- Editorial taxonomy (display labels, deck copy). Those live in
  ``cloris/frontend/src/lib/sources.ts`` so copy iteration doesn't
  touch the spawn path.
- Per-source readiness probes — those live in
  ``linkedin/health.py`` / ``github/health.py`` and are dispatched
  separately by ``GET /api/launch-readiness/{source}/{brief_id}``.
- Per-brief save destinations (Phase F Slice F2) — those will be
  added to the registry as a fourth callable when F2 ships.

Contract notes:

- All callables are pure with respect to the registry; they may read
  the brief from disk and may compute state-key hashes, but they
  must be free of side effects.
- The registry is module-scope and immutable post-import; sources
  cannot be added at runtime.
- Sources that return ``None`` from ``state_key_fn`` (e.g., a brief
  that lacks the source-specific identifier) signal "this source
  cannot launch for this brief"; the API layer surfaces a 422 in
  that case.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

Source = Literal["linkedin", "github", "designer", "exec_search", "researcher"]
PipelineState = Literal["production", "partial", "stub"]

if TYPE_CHECKING:
    # Import-only-for-typing block. The new fields added in Phase 1
    # Slice 1.0 (the multi-agent-execution pioneer slice) reference
    # types living in linkedin/, market_intelligence/, and shared/.
    # Importing them at runtime would invert the dependency direction
    # this module deliberately preserves: the launcher registry is the
    # single source of truth that the source modules register *into*,
    # not the other way around. ``from __future__ import annotations``
    # above keeps every annotation as a string at runtime, so the
    # TYPE_CHECKING block is sufficient for tooling.
    from linkedin.health import ReadinessReport
    from market_intelligence.schema import MarketEvidenceBatch
    from shared.runtime_state.read_models import RunSummary
    from shared.schemas import ExecutionPlan


@dataclass(frozen=True)
class SaveDestinationBlocker:
    """Per-brief save-destination blocker for the launch-readiness probe.

    Phase F Slice F2. Mirrors :class:`linkedin.health.ReadinessBlocker`
    so the API handler can aggregate brief-readiness blockers alongside
    source-readiness blockers without conversion. ``kind`` is always
    ``"config"`` for save-destination blockers — the recruiter needs
    to fill in a configuration value before launch can proceed.
    """

    kind: str
    message: str
    remediation: str
    code: str = ""


@dataclass(frozen=True)
class LauncherEntry:
    """One source's runtime contract.

    See module docstring for what each callable owns.

    Phase F Slice F2 adds ``save_destination_blocker_fn``: given a
    brief on disk, returns a :class:`SaveDestinationBlocker` if the
    brief lacks the per-source destination needed to launch, else
    ``None``. The API handler aggregates this with the source-level
    readiness probe so a brief without a configured destination
    blocks launch before the worker spawns.

    Multi-agent-execution Phase 1 Slice 1.0 (pioneer PR) declares
    eight optional callables and two scalar/flag fields that the
    follow-on slices 1.1–1.8 populate per source. Until those slices
    land, every new field defaults to ``None`` (or its appropriate
    type-default), so the registry contract is forward-compatible
    without any behavior change. Existing callsites that construct
    ``LauncherEntry(...)`` with the four pre-1.0 fields continue to
    work unchanged. Per the plan, see
    ``plans/multi-agent-execution-plan.md`` §"Phase 1 — Foundation".
    """

    state_key_fn: Callable[[str], str]
    state_dir_fn: Callable[[str], Path]
    orchestrator_argv_fn: Callable[..., list[str]]
    save_destination_blocker_fn: Callable[
        [str], SaveDestinationBlocker | None
    ] = lambda brief_path: None

    # ------------------------------------------------------------------
    # Phase 1 Slice 1.0 fields. Each is populated by a later slice; the
    # dataclass shape is declared upfront so 1.1–1.8 don't all serialize
    # on the same dataclass edit. Until each populating slice ships, the
    # default is ``None`` (or empty-string / ``False`` for the non-
    # callable fields), and every callsite still falls through to the
    # legacy if/elif ladder the slice will eventually replace.
    # ------------------------------------------------------------------

    # Slice 1.1: per-source launch-readiness probe. Today the API
    # readiness aggregator at cloris/api.py:2805-2814 branches on source
    # to call ``probe_linkedin_readiness`` / ``probe_github_readiness``
    # / fall through to ``report = None``. Slice 1.1 routes via the
    # registry: ``LAUNCHERS[source].readiness_probe_fn()``. Phase 2.2
    # (this slice's predecessor) populated the remaining three sources
    # — researcher / designer / exec_search — via per-module
    # ``probe_<source>_readiness`` functions. Every registered source
    # now supplies a non-None callable; the ``| None`` typing remains
    # so the dispatch site at ``_readiness_blockers`` stays
    # defensively safe against an unknown-source path.
    readiness_probe_fn: Callable[[], "ReadinessReport"] | None = None

    # Slice 1.2: in-process orchestrator dispatch for the frozen .app
    # worker. Today cloris/worker.py:295-308 branches on source and
    # silently rejects ``exec_search`` ("frozen .app supports linkedin,
    # github, researcher, and designer only"). Slice 1.2 routes via the
    # registry, closing the silent regression.
    in_process_dispatch_fn: Callable[[list[str]], int] | None = None

    # Slice 1.3: reflection-time research-packet builder. Today
    # market_intelligence/engine.py:1621-1636 branches on
    # ``batch.source``. Underlying signatures vary per source —
    # LinkedIn's ``maybe_build_and_persist_research_packet`` takes
    # ``(batch, *, reconstruct_report_analysis: bool)``; GitHub's
    # ``maybe_build_and_persist_github_research_packet`` takes
    # ``(batch,)``. The registered callables are thin adapter wrappers
    # (``_linkedin_research_packet_builder`` /
    # ``_github_research_packet_builder``) that share a uniform
    # ``(batch, *, reconstruct_report_analysis: bool) -> MarketEvidenceBatch``
    # signature so the dispatch site at engine.py:1621 invokes them
    # uniformly without per-source kwarg routing. The GitHub adapter
    # accepts and discards ``reconstruct_report_analysis`` (LinkedIn-
    # specific reflection reconstruction). OSS Maintainers Slice 9
    # (Phase 3 cleanup per spec §16) unifies the two underlying
    # builders into a shared abstraction once both have hardened
    # against real customer signal; until then the adapters keep the
    # underlying signatures intact.
    research_packet_builder_fn: (
        Callable[..., "MarketEvidenceBatch"] | None
    ) = None

    # Slice 1.4: snapshot-time helpers. Closes the five per-source
    # branches in market_intelligence/run_snapshots.py.
    #
    # ``brief_id_for_snapshot_fn(brief, raw, brief_path) -> str``
    # closes run_snapshots.py:286-289 (``_runtime_brief_id``).
    # ``Brief`` ships in two flavors today (shared.brief_loader vs
    # shared.brief_schema); ``Callable[..., str]`` defers the decision
    # to slice 1.4's adapter.
    brief_id_for_snapshot_fn: Callable[..., str] | None = None

    # ``progress_projection_fn(store, run_id, run_dir) -> None`` closes
    # run_snapshots.py:315 onwards (per-source progress.json /
    # search-memory / stage-projection rebuilds).
    progress_projection_fn: Callable[..., None] | None = None

    # ``snapshot_research_packet_fn(...)`` closes run_snapshots.py:409.
    # Parallel to ``research_packet_builder_fn`` (1.3) but in the
    # snapshot-time path.
    snapshot_research_packet_fn: Callable[..., Any] | None = None

    # ``snapshot_state_key_fn(...)`` closes run_snapshots.py:461.
    snapshot_state_key_fn: Callable[..., str] | None = None

    # ``reconstruct_report_analysis`` closes run_snapshots.py:535
    # (LinkedIn-only special case where reflection runs trigger
    # report-analysis reconstruction when ``run-report.json`` is
    # missing). Default ``False`` matches today's behavior for every
    # non-LinkedIn source. LinkedIn's slice 1.4 entry will set this to
    # ``True``.
    reconstruct_report_analysis: bool = False

    # Slice 1.5 (multi-agent-execution Phase 1): per-source work-unit
    # kind. The control_plane status aggregator and run-report builder
    # read ``LAUNCHERS[source].progress_kind`` directly; the legacy
    # ``_progress_kind_for_source`` ladder was removed in this slice
    # so the registry is the single source-of-truth.
    #
    # Registered values mirror the constants declared in
    # ``shared/runtime_state/store.py:37-41``: ``"linkedin_string"`` /
    # ``"github_query"`` / ``"researcher_author_query"`` /
    # ``"designer_behance_query"``. Empty string is the legitimate
    # sentinel for sources that don't aggregate work-unit progress
    # (today: ``exec_search``); the dispatch sites at
    # ``cloris/control_plane.py`` guard with ``if progress_kind:`` so
    # an empty string skips the ``read_models.work_unit_progress``
    # read entirely.
    progress_kind: str = ""

    # Designer go-live D3: multi-kind progress aggregation. Sources with
    # multiple work-unit kinds (Designer uses both designer_behance_query
    # and designer_cse_query) populate this tuple. When non-empty, the
    # control plane uses ``work_unit_progress_multi`` with a SQL IN clause
    # instead of the single-kind path. Empty tuple falls through to the
    # existing ``progress_kind`` single-kind path.
    progress_kinds: tuple[str, ...] = ()

    # Slice 1.6: per-source strategy formation, exposed via a uniform
    # ``form_strategy_for_registry(brief, prior_run_data) -> ExecutionPlan``
    # adapter that each module owns. Designer's deterministic
    # ``form_designer_strategy`` wraps the same way as LinkedIn's
    # Opus-driven ``form_strategy`` (correction 3a in the plan:
    # Designer is defensibly heterogeneous, not deviant). ``Brief``
    # is polymorphic across modules today; ``Callable[..., ExecutionPlan]``
    # defers signature normalization to the per-module adapter.
    form_strategy_fn: Callable[..., "ExecutionPlan"] | None = None

    # Slice 1.7: per-source run-summary read for cross-source synthesis
    # (Phase 2.4) and dispatch (Phase 2.5). Mirrors the read-helper
    # pattern at shared/runtime_state/read_models.py:742
    # (``extract_save_reason_and_confidence``) — read-only, opens
    # SQLite via ``mode=ro`` URI.
    summarize_run_fn: Callable[[Path], "RunSummary"] | None = None

    # Northwind trial hardening: product-facing pipeline maturity. Trial mode uses
    # this to expose only production modules; non-trial surfaces can still show
    # partial/stub modules with honest disabled copy.
    pipeline_state: PipelineState = "stub"

    # Reopen P7.1: administrative launchability gate, distinct from
    # ``pipeline_state``. ``pipeline_state`` describes product *maturity*
    # (a "partial" subagent can still be launched while it matures);
    # ``launchable=False`` means "administratively retired — refuse every
    # launch/resume attempt regardless of maturity." The single spawn
    # choke point (``_spawn_worker_for_source`` in
    # ``cloris/api/_monolith.py``) enforces this unconditionally — even
    # ``force=true`` does not bypass it, because force only exists to skip
    # readiness PROBES, not to resurrect a retired subagent. ``sunset`` is
    # the human-facing reason marker: sunset subagents are paused by
    # product decision ("paused for now"), not broken — the frontend can
    # use it to render that distinction instead of a generic disabled
    # state. Every registered source defaults to launchable; only sources
    # explicitly retired below flip both flags.
    launchable: bool = True
    sunset: bool = False


# ---------------------------------------------------------------------------
# Slice 1.2 (multi-agent-execution Phase 1): per-source ``in_process_dispatch_fn``
# adapters. The frozen .app worker at ``cloris/worker.py:_dispatch_in_process``
# routes through ``LAUNCHERS[source].in_process_dispatch_fn(orchestrator_argv)``
# instead of the pre-1.2 if/elif source ladder. The bundle ships no python
# interpreter, so ``os.execvp`` is unavailable; the adapter imports the
# orchestrator module and calls its ``main()`` in-process. PID stays this
# process's PID so the sidecar's ``pid`` field stays truthful for any
# later stop/probe operation.
#
# Side-effect of registry dispatch: ``exec_search`` becomes registerable in
# the frozen .app — closes the silent regression the legacy ladder encoded
# ("frozen .app supports linkedin, github, researcher, and designer only").
# Every registered source MUST populate this field; the dispatch site no
# longer falls through to a legacy branch and a ``None`` here would surface
# as the "no in-process dispatch" stderr path under the bundle.
# ---------------------------------------------------------------------------


def _dispatch_orchestrator_in_process(
    module_dotpath: str,
    orchestrator_argv: list[str],
) -> int:
    """Slice off the python invocation prefix, set ``sys.argv``, run ``main()``.

    The worker hands us the same argv shape ``orchestrator_argv_fn``
    produced — ``[python_executable, "-m", MODULE_DOTPATH, ...orchestrator-cli-args]``
    — so we drop the first three elements and replace them with
    ``MODULE_DOTPATH`` as ``sys.argv[0]`` (matching what the orchestrator
    would observe under ``python -m MODULE_DOTPATH``). The orchestrator
    reads its CLI args via ``argparse.parse_args()`` on ``sys.argv``,
    which is the contract that keeps it working unchanged under the
    frozen bundle.

    ``sys.argv`` is saved + restored around the call so dispatches don't
    bleed argv state across sources. Production hits this exactly once
    per worker process (the worker exits after the orchestrator returns),
    but the test suite calls this helper repeatedly across the
    parametrized 5-source sweep at
    ``tests/test_cloris_worker_sidecar.py``.

    The lazy ``importlib.import_module`` keeps ``cloris.launchers``
    free of import-time coupling to every per-source orchestrator —
    the registry stays the substrate that source modules register
    *into*, not the other way around.
    """

    import importlib

    sliced_argv = [module_dotpath, *orchestrator_argv[3:]]
    saved_argv = list(sys.argv)
    sys.argv = sliced_argv
    try:
        module = importlib.import_module(module_dotpath)
        rc = module.main()
    finally:
        sys.argv = saved_argv
    return rc if isinstance(rc, int) else 0


def _linkedin_in_process_dispatch(orchestrator_argv: list[str]) -> int:
    return _dispatch_orchestrator_in_process(
        "linkedin.session_orchestrator", orchestrator_argv
    )


def _github_in_process_dispatch(orchestrator_argv: list[str]) -> int:
    return _dispatch_orchestrator_in_process(
        "github.session_orchestrator", orchestrator_argv
    )


def _researcher_in_process_dispatch(orchestrator_argv: list[str]) -> int:
    return _dispatch_orchestrator_in_process(
        "researcher.session_orchestrator", orchestrator_argv
    )


def _designer_in_process_dispatch(orchestrator_argv: list[str]) -> int:
    return _dispatch_orchestrator_in_process(
        "designer.session_orchestrator", orchestrator_argv
    )


def _exec_search_in_process_dispatch(orchestrator_argv: list[str]) -> int:
    return _dispatch_orchestrator_in_process(
        "exec_search.session_orchestrator", orchestrator_argv
    )


def _linkedin_research_packet_builder(
    batch: "MarketEvidenceBatch",
    *,
    reconstruct_report_analysis: bool,
) -> "MarketEvidenceBatch":
    """Adapter wrapping :func:`maybe_build_and_persist_research_packet`.

    Slice 1.3 of multi-agent-execution. Replaces the legacy
    ``if batch.source == "linkedin"`` branch at
    ``market_intelligence/engine.py:1621``. Pure pass-through —
    forwards both the batch and the LinkedIn-specific
    ``reconstruct_report_analysis`` flag to the underlying builder.
    Lazy import keeps ``cloris.launchers`` importable without pulling
    ``market_intelligence`` at module-load time.
    """

    from market_intelligence.research_context import (
        maybe_build_and_persist_research_packet,
    )

    return maybe_build_and_persist_research_packet(
        batch,
        reconstruct_report_analysis=reconstruct_report_analysis,
    )


def _github_research_packet_builder(
    batch: "MarketEvidenceBatch",
    *,
    reconstruct_report_analysis: bool,
) -> "MarketEvidenceBatch":
    """Adapter wrapping :func:`maybe_build_and_persist_github_research_packet`.

    Slice 1.3 of multi-agent-execution. Replaces the legacy
    ``elif batch.source == "github"`` branch at
    ``market_intelligence/engine.py:1626``. The underlying GitHub
    builder doesn't take ``reconstruct_report_analysis`` (LinkedIn
    has a special-case where reflection runs reconstruct missing
    ``run-report.json`` analysis; GitHub doesn't). The kwarg is
    accepted and discarded so the dispatch site invokes a uniform
    signature regardless of source. OSS Maintainers Slice 9 (Phase 3
    cleanup per spec §16) is the post-trial work that unifies the
    two underlying builders into a shared abstraction; until then,
    this thin adapter keeps the GitHub builder's signature intact.
    """

    from market_intelligence.github_reflection import (
        maybe_build_and_persist_github_research_packet,
    )

    del reconstruct_report_analysis
    return maybe_build_and_persist_github_research_packet(batch)


def _researcher_research_packet_builder(
    batch: "MarketEvidenceBatch",
    *,
    reconstruct_report_analysis: bool,
) -> "MarketEvidenceBatch":
    del reconstruct_report_analysis
    from market_intelligence.researcher_reflection import (
        maybe_build_and_persist_researcher_research_packet,
    )

    return maybe_build_and_persist_researcher_research_packet(batch)


def _designer_research_packet_builder(
    batch: "MarketEvidenceBatch",
    *,
    reconstruct_report_analysis: bool,
) -> "MarketEvidenceBatch":
    del reconstruct_report_analysis
    from market_intelligence.design_market_intelligence import (
        maybe_build_and_persist_design_research_packet,
    )

    return maybe_build_and_persist_design_research_packet(batch)


def _exec_search_research_packet_builder(
    batch: "MarketEvidenceBatch",
    *,
    reconstruct_report_analysis: bool,
) -> "MarketEvidenceBatch":
    del reconstruct_report_analysis
    from market_intelligence.exec_search_reflection import (
        maybe_build_and_persist_exec_search_research_packet,
    )

    return maybe_build_and_persist_exec_search_research_packet(batch)


def _derive_brief_id(brief_path: str) -> str:
    """Compute the canonical brief id for a LinkedIn launch.

    Reads the brief content (`linkedin_project_id` / `id` / stem
    fallback) so the value stays stable across flat→nested migrations.
    """

    from shared.output_paths import derive_brief_id

    return derive_brief_id(brief_path=brief_path)


def _linkedin_readiness_probe() -> "ReadinessReport":
    """Registry adapter — Phase 1 Slice 1.1.

    Wraps :func:`linkedin.health.probe_linkedin_readiness` so the
    launch-readiness aggregator at ``cloris/api.py:_readiness_blockers``
    dispatches via ``LAUNCHERS[source].readiness_probe_fn()`` instead of
    branching on source. The inline import is load-bearing: tests at
    ``tests/test_launch_readiness_endpoint.py`` and
    ``tests/test_save_destination_config.py`` stub the probe with
    ``monkeypatch.setattr(linkedin.health, "probe_linkedin_readiness", ...)``,
    which only takes effect if the registered callable resolves the
    symbol at call time. Capturing the function object at module-import
    time would silently bypass those fixtures.
    """

    from linkedin.health import probe_linkedin_readiness

    return probe_linkedin_readiness()


def _linkedin_state_dir(brief_path: str) -> Path:
    from shared.output_paths import resolve_linkedin_state_dir

    return resolve_linkedin_state_dir(brief_path=brief_path)


def _linkedin_orchestrator_argv(
    brief_path: str,
    state_dir: str,
    *,
    resume: bool,
    fresh: bool = False,
    python_executable: str = sys.executable,
) -> list[str]:
    """LinkedIn execvp argv.

    Mirrors the existing :func:`cloris.worker.build_session_orchestrator_argv`
    contract. Kept here as a thin wrapper so the registry holds one
    callable per source rather than reaching into ``cloris.worker``.
    """

    from cloris.worker import build_session_orchestrator_argv

    return build_session_orchestrator_argv(
        brief_path=brief_path,
        state_dir=state_dir,
        resume=resume,
        fresh=fresh,
        python_executable=python_executable,
    )


def _linkedin_save_destination_blocker(
    brief_path: str,
) -> SaveDestinationBlocker | None:
    """Block LinkedIn launches when the brief lacks a project_id.

    Phase F Slice F2. Reads the V2 ``source_config.linkedin.project_id``
    field, falling back to the flat ``linkedin_project_id`` for briefs
    not yet migrated. Returns ``None`` (no blocker) when the project id
    is configured. The blocker's remediation is recruiter-actionable —
    points them at ``BriefDetail`` to fill in the destination.
    """

    from shared.brief_v2_schema import linkedin_project_id_from_brief
    from shared.storage import read_json

    try:
        raw = read_json(brief_path)
    except Exception:
        # If the brief can't be read at all, the launch will fail
        # downstream with a clearer error; don't double-report here.
        return None

    if not isinstance(raw, dict):
        return None

    project_id = linkedin_project_id_from_brief(raw)
    if project_id:
        return None

    return SaveDestinationBlocker(
        kind="config",
        message=(
            "Cloris doesn't yet know which LinkedIn project to save into "
            "for this brief."
        ),
        remediation=(
            "Paste your Recruiter project URL — either inline here, or from "
            "the brief's \"Where Cloris saves\" section."
        ),
    )


def _permanent_filter_truthy(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return bool(value)
    if isinstance(value, list):
        return bool(value)
    return True


def linkedin_permanent_filter_automation_blockers(
    brief_path: str,
) -> list[SaveDestinationBlocker]:
    """Surface permanent_filters that LinkedIn browser automation does not honor.

    Location is applied in ``linkedin.browser.LinkedInBrowser.apply_permanent_filters``;
    seniority, years of experience, company_filters, keywords, seniority_excluded,
    and any other keys are still TODO or silent no-ops there — block launch so the
    recruiter isn't misled by readiness that only checks auth/save destination.
    """

    from shared.storage import read_json

    try:
        raw = read_json(brief_path)
    except Exception:
        return []
    if not isinstance(raw, dict):
        return []
    pf = raw.get("permanent_filters")
    if not isinstance(pf, dict) or not pf:
        return []

    location_keys = frozenset({"Location", "location"})
    known_non_automated = frozenset(
        {
            "seniority",
            "years_experience",
            "company_filters",
            "keywords",
            "seniority_excluded",
        }
    )

    gap_labels: list[str] = []
    for key in sorted(known_non_automated):
        if key in pf and _permanent_filter_truthy(pf[key]):
            gap_labels.append(key.replace("_", " "))
    for key in sorted(pf.keys()):
        if key in location_keys or key in known_non_automated:
            continue
        if _permanent_filter_truthy(pf[key]):
            gap_labels.append(str(key))

    if not gap_labels:
        return []

    joined = ", ".join(gap_labels)
    return [
        SaveDestinationBlocker(
            kind="config",
            message=(
                "This brief's permanent filters include constraints LinkedIn "
                f"automation doesn't apply yet ({joined})."
            ),
            remediation=(
                "Remove those keys from permanent_filters in the brief JSON, or "
                "apply them manually in LinkedIn Recruiter before relying on results."
            ),
        )
    ]


def _github_state_key(brief_path: str) -> str:
    from shared.output_paths import github_state_key

    return github_state_key(brief_path=brief_path)


def _github_readiness_probe() -> "ReadinessReport":
    """Registry adapter — Phase 1 Slice 1.1.

    Wraps :func:`github.health.probe_github_readiness`. Inline import
    pattern matches :func:`_linkedin_readiness_probe`; see that
    docstring for the monkeypatch-compatibility rationale.
    """

    from github.health import probe_github_readiness

    return probe_github_readiness()


def _github_save_destination_blocker(
    brief_path: str,
) -> SaveDestinationBlocker | None:
    """OSS-Maintainers-posture readiness gate — P6.9.

    GitHub has no per-brief save destination (this slot's original F2
    purpose — see ``test_github_blocker_returns_none_for_classic_brief``
    in ``tests/test_save_destination_config.py``, formerly
    ``test_github_blocker_always_returns_none`` before P6.9 repurposed
    this slot). Repurposed for the one brief-shaped GitHub
    readiness check that exists: :func:`github.health.probe_github_readiness`
    is brief-agnostic (auth/token only) and is called with zero arguments
    from ``cloris/api/_monolith.py:_readiness_blockers``'s Layer 1, which
    never resolves a brief path — there is no seam there to thread a brief
    through without changing that dispatch's call arity. Layer 2
    (``save_destination_blocker_fn``) already receives ``brief_path`` for
    every source, so the posture gate lives here instead, mirroring
    :func:`_linkedin_save_destination_blocker`'s raw-JSON read pattern.

    Delegates the actual posture logic to
    :func:`github.health.github_target_projects_blocker` and translates
    its :class:`github.health.ReadinessBlocker` into this registry's
    :class:`SaveDestinationBlocker` shape.
    """

    from github.health import github_target_projects_blocker
    from shared.storage import read_json

    try:
        raw = read_json(brief_path)
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None

    target_projects = raw.get("target_projects")
    if not isinstance(target_projects, list):
        target_projects = []
    maintainership_level = raw.get("maintainership_level")
    if not isinstance(maintainership_level, str) or not maintainership_level:
        maintainership_level = "contributor"

    blocker = github_target_projects_blocker(
        target_projects=target_projects,
        maintainership_level=maintainership_level,
    )
    if blocker is None:
        return None

    return SaveDestinationBlocker(
        kind=blocker.kind,
        message=blocker.message,
        remediation=blocker.remediation,
    )


def _github_state_dir(brief_path: str) -> Path:
    from shared.output_paths import resolve_github_state_dir

    return resolve_github_state_dir(brief_path=brief_path)


def _github_orchestrator_argv(
    brief_path: str,
    state_dir: str,
    *,
    resume: bool,
    fresh: bool = False,
    python_executable: str = sys.executable,
) -> list[str]:
    """GitHub execvp argv.

    The GitHub orchestrator's CLI accepts ``--brief``, ``--state-dir``,
    and an optional ``--resume`` flag (per
    ``github/session_orchestrator.py:main``). It does not accept
    ``--input-mode`` — Cloris v0 is concurrent-only and the GitHub
    orchestrator's session model already runs concurrently. The
    ``cloris.worker`` wrapper writes the sidecar with
    ``input_mode="concurrent"`` regardless of source so the on-wire
    contract stays uniform.
    """

    argv: list[str] = [
        python_executable,
        "-m",
        "github.session_orchestrator",
        "--brief",
        brief_path,
        "--state-dir",
        state_dir,
    ]
    if resume:
        argv.append("--resume")
    return argv


def _researcher_state_key(brief_path: str) -> str:
    from shared.output_paths import researcher_state_key

    return researcher_state_key(brief_path=brief_path)


def _researcher_state_dir(brief_path: str) -> Path:
    from shared.output_paths import resolve_researcher_state_dir

    return resolve_researcher_state_dir(brief_path=brief_path)


def _researcher_orchestrator_argv(
    brief_path: str,
    state_dir: str,
    *,
    resume: bool,
    fresh: bool = False,
    python_executable: str = sys.executable,
) -> list[str]:
    """Researcher execvp argv.

    Mirrors the GitHub argv shape; the researcher orchestrator's CLI
    accepts ``--brief`` and ``--state-dir``.

    Reopen P7.5(b): ``--resume`` is deliberately NEVER appended, even
    when ``resume=True`` (e.g. the Cloris worker passes
    ``resume=(mode == "resume")`` uniformly across every source —
    cloris/worker.py:453-457 — so this function cannot simply not be
    called with ``resume=True``). Researcher's CLI
    (``researcher.session_orchestrator.main``) now treats ``--resume``
    as a hard "not implemented" error rather than the previous silent
    re-run-from-scratch theater; the launcher must not hand it a flag
    its own CLI will refuse. ``resume`` stays a required kwarg so this
    function's signature matches every other source's
    ``orchestrator_argv_fn`` in the registry.
    """

    return [
        python_executable,
        "-m",
        "researcher.session_orchestrator",
        "--brief",
        brief_path,
        "--state-dir",
        state_dir,
    ]


def _researcher_save_destination_blocker(
    brief_path: str,
) -> SaveDestinationBlocker | None:
    """Researcher saves always land in the workspace (no per-brief destination).

    Per Researcher Module Spec Opinion 4: researchers without LinkedIn
    profiles can't be saved to LinkedIn Recruiter; every saved researcher
    is a `candidates` row with SAVE-class `terminal_decision`. Workspace
    is always available, so no readiness blocker fires.
    """

    return None


def _researcher_readiness_probe() -> "ReadinessReport":
    """Registry adapter — Phase 2.2.

    Wraps :func:`researcher.health.probe_researcher_readiness`. Inline
    import pattern matches :func:`_linkedin_readiness_probe`; see that
    docstring for the monkeypatch-compatibility rationale.
    """

    from researcher.health import probe_researcher_readiness

    return probe_researcher_readiness()


def _designer_state_key(brief_path: str) -> str:
    from shared.output_paths import designer_state_key

    return designer_state_key(brief_path=brief_path)


def _designer_state_dir(brief_path: str) -> Path:
    from shared.output_paths import resolve_designer_state_dir

    return resolve_designer_state_dir(brief_path=brief_path)


def _designer_orchestrator_argv(
    brief_path: str,
    state_dir: str,
    *,
    resume: bool,
    fresh: bool = False,
    python_executable: str = sys.executable,
) -> list[str]:
    """Designer execvp argv.

    Mirrors the GitHub argv shape; the designer orchestrator's CLI
    accepts ``--brief``, ``--state-dir``, and an optional ``--resume``
    flag (matched by `designer.session_orchestrator.main`). Slice 1
    ships the stub with placeholder evaluator; Slices 2-5 wire the
    real source adapters and vision pipeline.
    """

    argv: list[str] = [
        python_executable,
        "-m",
        "designer.session_orchestrator",
        "--brief",
        brief_path,
        "--state-dir",
        state_dir,
    ]
    if resume:
        argv.append("--resume")
    return argv


def _designer_save_destination_blocker(
    brief_path: str,
) -> SaveDestinationBlocker | None:
    """Designer saves always land in the workspace (no per-brief destination).

    Mirrors the Researcher posture: the workspace is the implicit save
    destination — Designer-evaluated candidates are `candidates` rows
    with SAVE-class `terminal_decision` and `surface_type:
    "hitl_visual_review"` in `terminal_payload_json`. No per-brief
    destination configuration to gate on.
    """

    return None


def _exec_search_save_destination_blocker(
    brief_path: str,
) -> SaveDestinationBlocker | None:
    """Executive Search saves land in the Cloris-native shortlist — Slice A.8.

    Mirrors the Researcher / Designer posture: the workspace
    (specifically the ``surface_type: "exec_search_dossier"`` workspace
    card per Phase D.4) is the implicit save destination, so no
    per-brief destination configuration to gate on. The LinkedIn
    full-eval branch the exec_search dossier pipeline extends does
    write to LinkedIn Recruiter — but only when ``brief.target_modules``
    includes ``"linkedin"`` and the recruiter has a configured project
    id; that gate is the LinkedIn ``save_destination_blocker_fn`` at
    the LinkedIn launcher entry, NOT this one.
    """

    return None


def _designer_readiness_probe() -> "ReadinessReport":
    """Registry adapter — Phase 2.2.

    Wraps :func:`designer.health.probe_designer_readiness`. Inline
    import pattern matches :func:`_linkedin_readiness_probe`; see that
    docstring for the monkeypatch-compatibility rationale.
    """

    from designer.health import probe_designer_readiness

    return probe_designer_readiness()


def _exec_search_state_key(brief_path: str) -> str:
    from shared.output_paths import exec_search_state_key

    return exec_search_state_key(brief_path=brief_path)


def _exec_search_state_dir(brief_path: str) -> Path:
    from shared.output_paths import resolve_exec_search_state_dir

    return resolve_exec_search_state_dir(brief_path=brief_path)


def _exec_search_readiness_probe() -> "ReadinessReport":
    """Registry adapter — Phase 2.2.

    Wraps :func:`exec_search.health.probe_exec_search_readiness`.
    Inline import pattern matches :func:`_linkedin_readiness_probe`;
    see that docstring for the monkeypatch-compatibility rationale.
    """

    from exec_search.health import probe_exec_search_readiness

    return probe_exec_search_readiness()


def _exec_search_orchestrator_argv(
    brief_path: str,
    state_dir: str,
    *,
    resume: bool,
    fresh: bool = False,
    python_executable: str = sys.executable,
) -> list[str]:
    """Executive Search execvp argv.

    Mirrors the GitHub argv shape; the exec_search orchestrator's CLI
    accepts ``--brief``, ``--state-dir``, and an optional ``--resume``
    flag (matched by `exec_search.session_orchestrator.main`). Slice 1
    ships the stub (`main()` exits 0); Slices 2-10 wire the real
    pipeline (LinkedIn evaluation pipeline extension + off-LinkedIn
    signals + Cloris-native shortlist destination).
    """

    argv: list[str] = [
        python_executable,
        "-m",
        "exec_search.session_orchestrator",
        "--brief",
        brief_path,
        "--state-dir",
        state_dir,
    ]
    if resume:
        argv.append("--resume")
    return argv


# ---------------------------------------------------------------------------
# Slice 1.6 (multi-agent-execution Phase 1): per-source ``form_strategy_fn``
# adapter shims. Each shim lazy-imports the module's
# ``form_strategy_for_registry`` to keep the launcher registry free of
# import-time coupling to LLM clients / kit extractors / vision pipelines.
# Mirrors the lazy-import pattern used by ``_derive_brief_id`` and the
# Slice 1.1 / 1.2 / 1.3 / 1.5 shims above. Per correction 3a in
# ``plans/multi-agent-execution-plan.md``, Designer's adapter is
# deterministic — the registry surface unifies the *signature*, not the
# decision to invoke Opus.
# ---------------------------------------------------------------------------


def _linkedin_form_strategy(
    brief: Any,
    prior_run_data: dict | None = None,
) -> "ExecutionPlan":
    from linkedin.strategy import form_strategy_for_registry

    return form_strategy_for_registry(brief, prior_run_data)


def _github_form_strategy(
    brief: Any,
    prior_run_data: dict | None = None,
) -> "ExecutionPlan":
    from github.strategy import form_strategy_for_registry

    return form_strategy_for_registry(brief, prior_run_data)


def _researcher_form_strategy(
    brief: Any,
    prior_run_data: dict | None = None,
) -> "ExecutionPlan":
    from researcher.strategy import form_strategy_for_registry

    return form_strategy_for_registry(brief, prior_run_data)


def _designer_form_strategy(
    brief: Any,
    prior_run_data: dict | None = None,
) -> "ExecutionPlan":
    from designer.strategy import form_strategy_for_registry

    return form_strategy_for_registry(brief, prior_run_data)


def _exec_search_form_strategy(
    brief: Any,
    prior_run_data: dict | None = None,
) -> "ExecutionPlan":
    from exec_search.strategy import form_strategy_for_registry

    return form_strategy_for_registry(brief, prior_run_data)


# ---------------------------------------------------------------------------
# Slice 1.4 (multi-agent-execution Phase 1): per-source snapshot helpers.
# Closes the five LinkedIn-special-case branches at
# ``market_intelligence/run_snapshots.py:287, 315, 409, 461, 535``. Each
# helper is a thin lazy-import adapter so ``cloris.launchers`` stays free
# of import-time coupling to ``shared.output_paths`` /
# ``shared.runtime_state.projections`` /
# ``market_intelligence.research_context``.
#
# Population pattern matches the partial-population precedent set by
# Slice 1.1 (``readiness_probe_fn``: linkedin/github only) and Slice 1.3
# (``research_packet_builder_fn``: linkedin/github only):
#
# - ``brief_id_for_snapshot_fn`` (linkedin only) — closes :287. The
#   non-LinkedIn else branch at :289 was a generic ``brief.id ||
#   role_title || stem`` fallback shared by every other source; the
#   dispatch site preserves it as the None-fallback.
# - ``progress_projection_fn`` (linkedin + github) — closes :315.
#   LinkedIn writes 3 projections (progress, stage, search-memory);
#   GitHub writes 2 (progress, stage). The pre-Slice-1.4 code's else-
#   branch wrote GitHub's two projections for ANY non-LinkedIn source —
#   a dormant bug for researcher / designer / exec_search that this
#   slice closes by registering only the two real sources and falling
#   through to a no-op for the others (no production caller hits the
#   non-{linkedin,github} path; see plan §"Out of scope: snapshot
#   semantics changes" — this is registry cleanup, not behavior change).
# - ``snapshot_research_packet_fn`` (linkedin only) — closes :409.
#   Parallel to Slice 1.3's ``research_packet_builder_fn`` but in the
#   snapshot-time code path. Distinct registry slot so the two seams
#   can evolve independently (different kwargs / different upstream
#   builders) without coupling.
# - ``snapshot_state_key_fn`` (linkedin + github) — closes :461.
#   LinkedIn wraps :func:`shared.output_paths.derive_brief_id`;
#   GitHub wraps :func:`shared.output_paths.github_state_key`. The
#   dispatch site preserves the legacy ``github_state_key`` else-
#   fallback for unrecognized sources (today they don't reach the
#   dispatch site, but the fallback honors the pre-Slice-1.4 contract).
# - ``reconstruct_report_analysis`` (bool flag, True on linkedin) —
#   closes :535's ``source == "linkedin"`` predicate. The runtime check
#   ``not (run_dir / "run-report.json").exists()`` stays in the snapshot
#   module since it's a snapshot-time disk check, not a per-source
#   contract.
# ---------------------------------------------------------------------------


def _linkedin_brief_id_for_snapshot(
    brief: Any,
    raw: dict,
    brief_path: Path,
) -> str:
    """Adapter for the LinkedIn-special ``project_id`` brief-id fallback.

    Slice 1.4 of multi-agent-execution. Replaces the legacy
    ``if source == "linkedin"`` branch at
    ``market_intelligence/run_snapshots.py:287`` (in
    ``_runtime_brief_id``). LinkedIn keys runtime-state rows by the
    ``linkedin_project_id`` so resume / projection rebuilds find the
    right run; the resolution order matches the pre-slice fallback
    chain verbatim:

        raw["linkedin_project_id"] || brief.linkedin_project_id
        || brief.id || brief_path.stem

    Other sources don't have this LinkedIn-specific fallback — they
    register ``None`` and the dispatch site falls through to the
    generic ``brief.id || role_title || stem`` shape that the legacy
    else-branch supplied.
    """

    return str(
        raw.get("linkedin_project_id")
        or getattr(brief, "linkedin_project_id", "")
        or getattr(brief, "id", "")
        or brief_path.stem
    )


def _linkedin_progress_projection(
    *,
    store: Any,
    brief_id: str,
    run_id: int,
    run_dir: Path,
) -> None:
    """Adapter for LinkedIn's snapshot-time projection rebuild.

    Slice 1.4 of multi-agent-execution. Replaces the legacy
    ``if source == "linkedin"`` branch at
    ``market_intelligence/run_snapshots.py:315``. LinkedIn writes three
    projection artifacts back into the snapshot's run-dir:
    ``progress.json`` (per-run progress projection), the per-stage
    snippets / facial / profile JSONLs (``write_linkedin_stage_projections``),
    and the brief-scoped ``search_memory-<brief_id>.json``. Each call
    is wrapped in the snapshot module's ``_try_projection`` guard
    (legacy-payload-tolerant); the adapter does the orchestration here
    so the snapshot module just dispatches and the per-source contract
    lives in the registry.
    """

    from shared.runtime_state.projections import (
        write_linkedin_progress_projection,
        write_linkedin_search_memory_projection,
        write_linkedin_stage_projections,
    )

    _try_snapshot_projection(
        write_linkedin_progress_projection,
        store,
        run_id,
        run_dir / "progress.json",
    )
    _try_snapshot_projection(
        write_linkedin_stage_projections,
        store,
        brief_id=brief_id,
        output_dir=run_dir,
        run_id=run_id,
    )
    _try_snapshot_projection(
        write_linkedin_search_memory_projection,
        store,
        brief_id=brief_id,
        path=run_dir / f"search_memory-{brief_id}.json",
        run_id=run_id,
    )


def _github_progress_projection(
    *,
    store: Any,
    brief_id: str,
    run_id: int,
    run_dir: Path,
) -> None:
    """Adapter for GitHub's snapshot-time projection rebuild.

    Slice 1.4 of multi-agent-execution. Replaces the legacy fall-
    through (the ``return``-less else branch) at
    ``market_intelligence/run_snapshots.py:332-339``. GitHub writes
    two projection artifacts: ``progress.json`` and the per-stage
    JSONLs (``write_github_stage_projections``). No
    ``search_memory-<brief_id>.json`` — GitHub's runtime state doesn't
    project a per-brief search memory the way LinkedIn does (the
    GitHub orchestrator persists query-level state inside its own
    ``session_*_strategy.jsonl`` artifacts which the snapshot copy
    pattern handles separately).
    """

    from shared.runtime_state.projections import (
        write_github_progress_projection,
        write_github_stage_projections,
    )

    _try_snapshot_projection(
        write_github_progress_projection,
        store,
        run_id,
        run_dir / "progress.json",
    )
    _try_snapshot_projection(
        write_github_stage_projections,
        store,
        brief_id=brief_id,
        output_dir=run_dir,
        run_id=run_id,
    )


def _try_snapshot_projection(builder: Any, /, *args: Any, **kwargs: Any) -> None:
    """Behavior-preserving copy of ``run_snapshots._try_projection``.

    Pre-Slice-1.4 the guard lived inline inside
    ``_rebuild_run_scoped_projections``; post-slice the guard moves
    into the registered adapters so each per-source projection
    rebuild is the single thing the dispatch site invokes. The
    legacy comment is preserved verbatim because the contract is the
    same: legacy runtime payloads aren't always shape-stable enough
    to rebuild every compatibility projection, and snapshot creation
    should preserve the run rather than failing the whole import.
    """

    try:
        builder(*args, **kwargs)
    except (TypeError, ValueError, KeyError):
        return


def _linkedin_snapshot_research_packet(
    batch: "MarketEvidenceBatch",
    *,
    reconstruct_report_analysis: bool,
) -> "MarketEvidenceBatch":
    """Adapter for LinkedIn's snapshot-time research-packet build.

    Slice 1.4 of multi-agent-execution. Replaces the legacy
    ``if source == "linkedin"`` branch at
    ``market_intelligence/run_snapshots.py:409``. Calls the same
    underlying :func:`maybe_build_and_persist_research_packet` that
    Slice 1.3's reflection-time adapter does, but registered on a
    distinct field so the two call sites can diverge if their kwarg
    shapes ever do (today they share the
    ``reconstruct_report_analysis`` kwarg; a future Researcher /
    Designer snapshot research-packet contract may not).
    """

    from market_intelligence.research_context import (
        maybe_build_and_persist_research_packet,
    )

    return maybe_build_and_persist_research_packet(
        batch,
        reconstruct_report_analysis=reconstruct_report_analysis,
    )


def _linkedin_snapshot_state_key(
    *,
    brief_path: str | Path,
    brief: Any,
    raw: dict,
) -> str:
    """Adapter for LinkedIn's snapshot-time state-key derivation.

    Slice 1.4 of multi-agent-execution. Replaces the legacy
    ``if source == "linkedin"`` branch at
    ``market_intelligence/run_snapshots.py:461``. Wraps
    :func:`shared.output_paths.derive_brief_id` (which already
    encodes the source_config / flat / brief / stem fallback chain
    that Slice F2 stabilized).
    """

    from shared.output_paths import derive_brief_id

    return derive_brief_id(brief_path=brief_path, brief=brief, raw=raw)


def _github_snapshot_state_key(
    *,
    brief_path: str | Path,
    brief: Any,
    raw: dict,
) -> str:
    """Adapter for GitHub's snapshot-time state-key derivation.

    Slice 1.4 of multi-agent-execution. Replaces the legacy fall-
    through at ``market_intelligence/run_snapshots.py:463``. Wraps
    :func:`shared.output_paths.github_state_key`. ``raw`` is accepted
    for signature uniformity across registered adapters and discarded
    — GitHub doesn't read the raw brief dict (no
    ``source_config.github`` lookup analogous to LinkedIn's
    ``source_config.linkedin.project_id``).
    """

    del raw
    from shared.output_paths import github_state_key

    return github_state_key(brief_path=brief_path, brief=brief)


def _researcher_snapshot_state_key(
    *,
    brief_path: str | Path,
    brief: Any,
    raw: dict,
) -> str:
    """Adapter for Researcher's snapshot-time state-key derivation — Slice A.7.

    Wraps :func:`shared.output_paths.researcher_state_key`. Closes the
    dormant ``finalize_run_snapshot`` non-{linkedin,github} fall-through
    at ``market_intelligence/run_snapshots.py:_brief_namespace_key``,
    which previously routed researcher snapshots through
    ``github_state_key`` — a wrong-source fallback that would corrupt
    the snapshot namespace if Researcher ever invoked
    ``finalize_run_snapshot``. Researcher's pipeline (Slice 6 shipped)
    reaches snapshot finalization on customer-launch (Phase B.5), so
    this adapter must be registered before the first researcher
    customer run.

    ``raw`` is accepted for signature uniformity and discarded —
    ``researcher_state_key`` derives the key from the brief dataclass
    + path (no ``source_config.researcher`` project-id analog yet).
    """

    del raw
    from shared.output_paths import researcher_state_key

    return researcher_state_key(brief_path=brief_path, brief=brief)


def _designer_snapshot_state_key(
    *,
    brief_path: str | Path,
    brief: Any,
    raw: dict,
) -> str:
    """Adapter for Designer's snapshot-time state-key derivation — Slice A.7.

    Wraps :func:`shared.output_paths.designer_state_key`. Closes the
    same dormant fall-through Researcher closes; Designer reaches
    ``finalize_run_snapshot`` once Phase C.1's orchestrator ships, so
    this adapter must be registered before C.10's customer launch.
    ``raw`` is accepted for signature uniformity and discarded.
    """

    del raw
    from shared.output_paths import designer_state_key

    return designer_state_key(brief_path=brief_path, brief=brief)


def _exec_search_snapshot_state_key(
    *,
    brief_path: str | Path,
    brief: Any,
    raw: dict,
) -> str:
    """Adapter for Executive Search's snapshot-time state-key derivation — Slice A.7.

    Wraps :func:`shared.output_paths.exec_search_state_key`. Closes
    the dormant fall-through; exec_search reaches snapshot
    finalization once Phase D.1's orchestrator ships. ``raw`` is
    accepted for signature uniformity and discarded.
    """

    del raw
    from shared.output_paths import exec_search_state_key

    return exec_search_state_key(brief_path=brief_path, brief=brief)


# Module-scope source registry. Adding a new source is a single-line
# append below. Keep keys lowercase to match the URL path convention.
#
# Phase F Slice F2: ``save_destination_blocker_fn`` is added per source.
# GitHub returns ``None`` (no per-brief destination concept today —
# saves are JSONL on disk in the run folder, addressable without
# recruiter input). LinkedIn requires a project_id; the blocker fires
# when the brief lacks ``source_config.linkedin.project_id`` (fallback
# to the flat ``linkedin_project_id`` is handled by the helper).
# Researcher returns ``None`` (workspace-only saves per Spec Opinion 4).
# Designer returns ``None`` (workspace-only saves; the visual judgment
# payload lands in `terminal_payload_json` for the HITL visual review
# surface).
# Executive Search (Slice 1): no blocker yet — the Cloris-native
# shortlist destination ships in Slice 7 (which depends on
# multi-module-foundation Slices 6-7). Until then, exec_search uses
# the default no-op blocker so launches don't fail readiness.
#
# Slice 1.5 (multi-agent-execution Phase 1): ``progress_kind`` is
# populated per source from the canonical work-unit-kind constants
# at ``shared/runtime_state/store.py:37-41``. The control_plane
# status aggregator and run-report builder read
# ``LAUNCHERS[source].progress_kind`` directly; the legacy
# ``_progress_kind_for_source`` ladder was removed in this slice.
# ``exec_search`` keeps the default empty string — it has no
# work-unit-aggregation channel today, so the ``if progress_kind:``
# guards in ``cloris/control_plane.py`` skip the work-unit read for
# its state dirs (preserving today's behavior).
#
# Slice 1.7 (multi-agent-execution Phase 1): ``summarize_run_fn`` is
# populated per source by the read-only helpers at
# ``shared/runtime_state/read_models.py:391-447``. Every source has
# canonical runtime state in its per-state-dir
# ``runtime_state.sqlite3``, so all five register a non-None helper —
# unlike ``readiness_probe_fn`` (1.1) or ``form_strategy_fn`` (1.6)
# this is fully populated from day one. Each helper opens SQLite via
# ``mode=ro`` URI through ``read_models._open_readonly``, so the
# registry-side dispatch the chief-of-staff agent uses (Phase 2.4
# synthesis extensions, Phase 2.5 dispatch heuristic) cannot trigger
# DDL or INSERT side effects on the per-source state — the same
# invariant the read-models layering test
# (``tests/test_read_models.py:45``) pins for the broader read path.
# Top-level import (rather than per-shim lazy import) is intentional:
# ``read_models`` is hermetic stdlib-only, so there's no transitive
# coupling to source modules that would re-introduce the
# ``cloris.launchers``-imports-``linkedin/`` cycle the lazy-import
# pattern exists to prevent.
from shared.runtime_state.read_models import (
    summarize_designer_run,
    summarize_exec_search_run,
    summarize_github_run,
    summarize_linkedin_run,
    summarize_researcher_run,
)
from shared.runtime_state.store import (
    DESIGNER_BEHANCE_QUERY_KIND,
    DESIGNER_CSE_QUERY_KIND,
    EXEC_SEARCH_QUERY_KIND,
    GITHUB_QUERY_KIND,
    LINKEDIN_STRING_KIND,
    RESEARCHER_AUTHOR_QUERY_KIND,
)

LAUNCHERS: dict[str, LauncherEntry] = {
    "linkedin": LauncherEntry(
        state_key_fn=_derive_brief_id,
        state_dir_fn=_linkedin_state_dir,
        orchestrator_argv_fn=_linkedin_orchestrator_argv,
        save_destination_blocker_fn=_linkedin_save_destination_blocker,
        readiness_probe_fn=_linkedin_readiness_probe,
        in_process_dispatch_fn=_linkedin_in_process_dispatch,
        research_packet_builder_fn=_linkedin_research_packet_builder,
        progress_kind=LINKEDIN_STRING_KIND,
        form_strategy_fn=_linkedin_form_strategy,
        summarize_run_fn=summarize_linkedin_run,
        brief_id_for_snapshot_fn=_linkedin_brief_id_for_snapshot,
        progress_projection_fn=_linkedin_progress_projection,
        snapshot_research_packet_fn=_linkedin_snapshot_research_packet,
        snapshot_state_key_fn=_linkedin_snapshot_state_key,
        reconstruct_report_analysis=True,
        pipeline_state="production",
    ),
    "github": LauncherEntry(
        state_key_fn=_github_state_key,
        state_dir_fn=_github_state_dir,
        orchestrator_argv_fn=_github_orchestrator_argv,
        save_destination_blocker_fn=_github_save_destination_blocker,
        readiness_probe_fn=_github_readiness_probe,
        in_process_dispatch_fn=_github_in_process_dispatch,
        research_packet_builder_fn=_github_research_packet_builder,
        progress_kind=GITHUB_QUERY_KIND,
        form_strategy_fn=_github_form_strategy,
        summarize_run_fn=summarize_github_run,
        progress_projection_fn=_github_progress_projection,
        snapshot_state_key_fn=_github_snapshot_state_key,
        pipeline_state="partial",
    ),
    "researcher": LauncherEntry(
        state_key_fn=_researcher_state_key,
        state_dir_fn=_researcher_state_dir,
        orchestrator_argv_fn=_researcher_orchestrator_argv,
        save_destination_blocker_fn=_researcher_save_destination_blocker,
        readiness_probe_fn=_researcher_readiness_probe,
        in_process_dispatch_fn=_researcher_in_process_dispatch,
        # A.6 — pass-through shim until F.2b ships per-module packet content.
        research_packet_builder_fn=_researcher_research_packet_builder,
        # A.7 — close the dormant fall-through at run_snapshots.py.
        snapshot_state_key_fn=_researcher_snapshot_state_key,
        progress_kind=RESEARCHER_AUTHOR_QUERY_KIND,
        form_strategy_fn=_researcher_form_strategy,
        summarize_run_fn=summarize_researcher_run,
        pipeline_state="partial",
    ),
    "designer": LauncherEntry(
        state_key_fn=_designer_state_key,
        state_dir_fn=_designer_state_dir,
        orchestrator_argv_fn=_designer_orchestrator_argv,
        save_destination_blocker_fn=_designer_save_destination_blocker,
        readiness_probe_fn=_designer_readiness_probe,
        in_process_dispatch_fn=_designer_in_process_dispatch,
        # A.6 — pass-through shim until F.2b ships per-module packet content.
        research_packet_builder_fn=_designer_research_packet_builder,
        # A.7 — close the dormant fall-through at run_snapshots.py.
        snapshot_state_key_fn=_designer_snapshot_state_key,
        progress_kind=DESIGNER_BEHANCE_QUERY_KIND,
        progress_kinds=(DESIGNER_BEHANCE_QUERY_KIND, DESIGNER_CSE_QUERY_KIND),
        form_strategy_fn=_designer_form_strategy,
        summarize_run_fn=summarize_designer_run,
        pipeline_state="partial",
        # Reopen P7.1 — Designer is sunset (consolidation decision of
        # record, 2026-07-02): gate off launches, do not fix internals.
        launchable=False,
        sunset=True,
    ),
    "exec_search": LauncherEntry(
        state_key_fn=_exec_search_state_key,
        state_dir_fn=_exec_search_state_dir,
        orchestrator_argv_fn=_exec_search_orchestrator_argv,
        # A.8 — Cloris-native shortlist always available; no per-brief
        # destination configuration to gate on.
        save_destination_blocker_fn=_exec_search_save_destination_blocker,
        readiness_probe_fn=_exec_search_readiness_probe,
        in_process_dispatch_fn=_exec_search_in_process_dispatch,
        # A.6 — pass-through shim until F.2b ships per-module packet content.
        research_packet_builder_fn=_exec_search_research_packet_builder,
        # A.7 — close the dormant fall-through at run_snapshots.py.
        snapshot_state_key_fn=_exec_search_snapshot_state_key,
        # A.8 — populate the work-unit kind so the control plane's
        # progress aggregator can distinguish exec_search dossier work
        # from LinkedIn's plain-eval work, even though both run inside
        # the LinkedIn orchestrator process.
        progress_kind=EXEC_SEARCH_QUERY_KIND,
        form_strategy_fn=_exec_search_form_strategy,
        summarize_run_fn=summarize_exec_search_run,
        pipeline_state="partial",
        # Reopen P7.1 — Executive search is sunset (consolidation decision
        # of record, 2026-07-02): gate off launches, do not fix internals.
        launchable=False,
        sunset=True,
    ),
}


def known_sources() -> tuple[str, ...]:
    """Return the registered source names, in stable ascending order.

    Used by the API layer to compose a 422 ``allowed`` list when an
    unknown source is requested, and by F5's module picker / F7's
    home aggregator to enumerate sources without re-declaring the set.
    """

    return tuple(sorted(LAUNCHERS.keys()))


def pipeline_state_for_source(source: str) -> PipelineState:
    """Return the product maturity state for a registered source."""

    return LAUNCHERS[source].pipeline_state


def get_launcher(source: str) -> LauncherEntry:
    """Return the registered :class:`LauncherEntry` for ``source``.

    Raises :class:`KeyError` for unknown sources; callers that need to
    surface a structured 422 should check ``source in LAUNCHERS`` first
    and use :func:`known_sources` for the allow-list payload.
    """

    return LAUNCHERS[source]
