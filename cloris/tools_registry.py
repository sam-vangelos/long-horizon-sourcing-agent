"""Phase G Slice G4: Tools index registry.

Source of truth for the catalog at ``GET /api/tools``. Each entry
carries its tier (A/B/C), editorial-voice description, args schema,
and execution model. The C tier is documentation-only — Cloris shows
the CLI command + one-line description without offering a UI run
button. A and B tier entries get UI execution via the runtime in
``cloris.tools_runtime``.

Why a registry instead of dynamic discovery:
- Hard arg validation per tool. Each tool's args_schema is an explicit
  Pydantic model; subprocess argv is constructed from validated fields.
  We never pass user-supplied flags through to the shell.
- Editorial framing per tool, written by hand. Auto-generated from
  argparse helps would read like a man page; recruiters need
  Cloris-voice prose.
- A registry-driven catalog means adding a new tool is a single file
  edit; no plumbing changes per tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict


Tier = Literal["A", "B", "C"]
ExecutionModel = Literal["sync", "async", "cli_only"]


# ---------------------------------------------------------------------------
# Per-tool argument schemas. Pydantic validates JSON bodies; the executor
# reads validated fields and constructs subprocess argv from them.
# ---------------------------------------------------------------------------


class IterateBriefArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brief_path: str
    report_path: str
    search_memory_path: str | None = None


class UpdateMarketIntelArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brief_path: str
    run_id: int
    output_root: str | None = None


class RunRecruiterIdentityResolverArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow_mode: Literal["dry_run", "apply"] = "dry_run"


class BackfillCleanLinkedinUrlsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state_root: str | None = None
    dry_run: bool = True


class BackfillBriefSnapshotArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["linkedin", "github", "designer"] | None = None
    state_key: str | None = None
    state_root: str | None = None
    apply: bool = False  # default dry-run


class GeminiConsultArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slug: str
    consult_type: Literal["review", "roadmap-fit"] = "review"
    ensemble: bool = False


# ---------------------------------------------------------------------------
# Tool registry shape.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolDefinition:
    tool_id: str
    tier: Tier
    label: str  # short title, e.g. "Iterate brief"
    pitch: str  # editorial Cloris-voice one-liner
    cli_command: str  # canonical CLI invocation (always shown)
    execution_model: ExecutionModel
    args_schema: type[BaseModel] | None = None
    argv_builder: Callable[[BaseModel], list[str]] | None = None
    runner: Callable[[BaseModel], tuple[int, str, str]] | None = None


# ---------------------------------------------------------------------------
# Argv builders. Each takes the validated args model and returns the full
# subprocess argv. NEVER reads user-supplied strings into a shell command.
# ---------------------------------------------------------------------------


def _argv_iterate_brief(args: IterateBriefArgs) -> list[str]:
    argv = [
        "tools/iterate_brief.py",
        "--brief", args.brief_path,
        "--report", args.report_path,
    ]
    if args.search_memory_path is not None:
        argv += ["--search-memory", args.search_memory_path]
    return argv


def _argv_update_market_intel(args: UpdateMarketIntelArgs) -> list[str]:
    argv = [
        "tools/update_market_intel.py",
        "--brief", args.brief_path,
        "--run-id", str(args.run_id),
    ]
    if args.output_root is not None:
        argv += ["--output-root", args.output_root]
    return argv


def _argv_run_recruiter_identity_resolver(
    args: RunRecruiterIdentityResolverArgs,
) -> list[str]:
    return [
        "tools/run_recruiter_identity_resolver.py",
        "--workflow-mode", args.workflow_mode,
    ]


def _argv_backfill_clean_linkedin_urls(
    args: BackfillCleanLinkedinUrlsArgs,
) -> list[str]:
    argv = ["tools/backfill_clean_linkedin_urls.py"]
    if args.state_root is not None:
        argv += ["--state-root", args.state_root]
    if args.dry_run:
        argv += ["--dry-run"]
    return argv


def _run_backfill_clean_linkedin_urls(
    args: BackfillCleanLinkedinUrlsArgs,
) -> tuple[int, str, str]:
    from tools.backfill_clean_linkedin_urls import run_backfill_clean_linkedin_urls

    return run_backfill_clean_linkedin_urls(
        state_root=Path(args.state_root) if args.state_root is not None else None,
        dry_run=args.dry_run,
    )


def _argv_backfill_brief_snapshot(
    args: BackfillBriefSnapshotArgs,
) -> list[str]:
    argv = ["tools/backfill_brief_snapshot.py"]
    if args.source is not None:
        argv += ["--source", args.source]
    if args.state_key is not None:
        argv += ["--state-key", args.state_key]
    if args.state_root is not None:
        argv += ["--state-root", args.state_root]
    if args.apply:
        argv += ["--apply"]
    return argv


def _argv_gemini_consult(args: GeminiConsultArgs) -> list[str]:
    argv = [
        "tools/gemini_consult.py",
        "--slug", args.slug,
        "--type", args.consult_type,
    ]
    if args.ensemble:
        argv += ["--ensemble"]
    return argv


# ---------------------------------------------------------------------------
# The registry. Order in the list = order in the UI catalog (top to bottom).
# Tier A first, then B, then C — recruiter affinity sort.
# ---------------------------------------------------------------------------


_REGISTRY: list[ToolDefinition] = [
    # --- Tier A — recruiter-facing, full UI ---
    ToolDefinition(
        tool_id="iterate_brief",
        tier="A",
        label="Iterate brief",
        pitch="Cloris drafts the next version of this brief based on what worked and what didn't on the last run. Review the draft before saving.",
        cli_command="tools/iterate_brief.py --brief <brief_path> --report <run_report_dir>",
        execution_model="async",
        args_schema=IterateBriefArgs,
        argv_builder=_argv_iterate_brief,
    ),
    ToolDefinition(
        tool_id="update_market_intel",
        tier="A",
        label="Update market intelligence",
        pitch="Refresh the market intelligence artifact for a brief's role + geography. Use after a new run lands so the next brief picks up the latest signal.",
        cli_command="tools/update_market_intel.py --brief <brief_path> --run-id <id>",
        execution_model="async",
        args_schema=UpdateMarketIntelArgs,
        argv_builder=_argv_update_market_intel,
    ),
    ToolDefinition(
        tool_id="run_recruiter_identity_resolver",
        tier="A",
        label="Reconcile GitHub leads against LinkedIn",
        pitch="Walks GitHub-sourced leads and tries to match them to LinkedIn profiles. Conservative by default; pass apply=true to write the results.",
        cli_command="tools/run_recruiter_identity_resolver.py --workflow-mode dry_run",
        execution_model="async",
        args_schema=RunRecruiterIdentityResolverArgs,
        argv_builder=_argv_run_recruiter_identity_resolver,
    ),
    ToolDefinition(
        tool_id="backfill_clean_linkedin_urls",
        tier="A",
        label="Re-normalize LinkedIn URLs",
        pitch="Strips tracking parameters off saved LinkedIn URLs across every state DB. Idempotent; defaults to dry-run so you can preview the cleanup.",
        cli_command="tools/backfill_clean_linkedin_urls.py --dry-run",
        execution_model="async",
        args_schema=BackfillCleanLinkedinUrlsArgs,
        argv_builder=_argv_backfill_clean_linkedin_urls,
        runner=_run_backfill_clean_linkedin_urls,
    ),
    # --- Tier B — operator-diagnostic, full UI ---
    ToolDefinition(
        tool_id="backfill_brief_snapshot",
        tier="B",
        label="Backfill brief snapshots",
        pitch="Walk every state dir and pin each run's brief snapshot. Heals legacy runs that pre-date Phase 3's brief identity pinning.",
        cli_command="tools/backfill_brief_snapshot.py --apply",
        execution_model="async",
        args_schema=BackfillBriefSnapshotArgs,
        argv_builder=_argv_backfill_brief_snapshot,
    ),
    ToolDefinition(
        tool_id="gemini_consult",
        tier="B",
        label="Consult Gemini on a surface",
        pitch="Reads captured surface artifacts and asks Gemini for an editorial critique. Use ensemble for the consensus view.",
        cli_command="tools/gemini_consult.py --slug <surface> --type review",
        execution_model="async",
        args_schema=GeminiConsultArgs,
        argv_builder=_argv_gemini_consult,
    ),
    # --- Tier B/C — documentation-only entries (cli_only) ---
    # These tools are real but expensive / interactive / deck-production
    # workflows that don't benefit from a UI wrapper today. The catalog
    # surfaces them as CLI documentation so recruiters / operators see
    # the full tool inventory in one place without lying about what's
    # wrapped.
    ToolDefinition(
        tool_id="audit_surfaces",
        tier="B",
        label="Capture UI surfaces (Playwright)",
        pitch="Walks every Cloris surface deterministically at 1024 / 1280 / 1440px, dumps screenshots + DOM facts. Long-running; CLI only for now.",
        cli_command="make audit-ui",
        execution_model="cli_only",
    ),
    ToolDefinition(
        tool_id="audit_rules",
        tier="B",
        label="Score captured surfaces against design doctrine",
        pitch="Reads an audit dir + emits rule-results.json. Pairs with audit-surfaces.",
        cli_command="tools/audit_rules.py --audit-dir <dir>",
        execution_model="cli_only",
    ),
    ToolDefinition(
        tool_id="audit_report",
        tier="B",
        label="Render audit violations report",
        pitch="Reads rule-results.json + writes the markdown violations report.",
        cli_command="tools/audit_report.py --audit-dir <dir>",
        execution_model="cli_only",
    ),
    ToolDefinition(
        tool_id="runtime_state_admin",
        tier="B",
        label="Runtime state forensics",
        pitch="Repair, replay, requeue. Touches the runtime SQLite — use with intent. CLI only.",
        cli_command="tools/runtime_state_admin.py --command <name>",
        execution_model="cli_only",
    ),
    ToolDefinition(
        tool_id="aggregate_shadow_judgments",
        tier="C",
        label="Aggregate shadow judgments",
        pitch="Offline aggregator for shadow_final_judgments evidence bakeoff.",
        cli_command="tools/aggregate_shadow_judgments.py --discover --json-out <path>",
        execution_model="cli_only",
    ),
    ToolDefinition(
        tool_id="evaluate_shadow_thresholds",
        tier="C",
        label="Evaluate shadow thresholds",
        pitch="Threshold pass/fail verdict for a bakeoff summary.",
        cli_command="tools/evaluate_shadow_thresholds.py --summary <path>",
        execution_model="cli_only",
    ),
]


def list_tools() -> list[ToolDefinition]:
    """Return the registry in catalog order."""

    return list(_REGISTRY)


def find_tool(tool_id: str) -> ToolDefinition | None:
    for tool in _REGISTRY:
        if tool.tool_id == tool_id:
            return tool
    return None
