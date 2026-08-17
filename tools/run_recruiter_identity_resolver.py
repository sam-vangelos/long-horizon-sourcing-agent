"""Run the Recruiter-first reconciliation tool for saved GitHub leads."""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
from pathlib import Path

from github.reconciliation_input import load_saved_github_reconciliation_batch_with_fallback
from github.recruiter_identity_report import (
    build_recruiter_identity_row,
    write_recruiter_identity_csv,
    write_recruiter_identity_summary,
    write_recruiter_reconciliation_saved_csv,
    write_recruiter_reconciliation_saved_jsonl,
)
from linkedin.browser import LinkedInBrowser
from linkedin.recruiter_identity_resolver import (
    RecruiterIdentityResolver,
    RecruiterResolverConfig,
)
from shared.brief_loader import load_brief
from shared.governor import SessionGovernor
from shared.judger import init_judger
from shared.recruiter_brief_resolution import resolve_linkedin_brief_path_for_github_run
from shared.storage import append_jsonl, read_json, read_jsonl, write_json

WORKFLOW_MODE_IDENTITY_COLLECT = "identity_collect"
WORKFLOW_MODE_FIT_GATED_SAVE = "fit_gated_save"
ALLOWED_WORKFLOW_MODES = (WORKFLOW_MODE_IDENTITY_COLLECT, WORKFLOW_MODE_FIT_GATED_SAVE)

ALLOWED_QUERY_EXPANSION_POLICIES = ("auto", "name_first", "enriched")
DEFAULT_QUERY_EXPANSION_POLICY = "auto"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconcile GitHub-sourced leads against LinkedIn Recruiter (identity + fit + engagement)."
    )
    parser.add_argument(
        "--github-output-dir",
        required=True,
        help="GitHub run output directory containing saved leads.",
    )
    parser.add_argument(
        "--project-url",
        help="LinkedIn Recruiter project/search URL to use when the agent should navigate itself.",
    )
    parser.add_argument(
        "--output-dir",
        help="Where to write resolver artifacts; defaults to the GitHub output directory.",
    )
    parser.add_argument(
        "--max-leads",
        type=int,
        default=25,
        help="Maximum number of saved GitHub leads to process.",
    )
    parser.add_argument(
        "--lead-offset",
        type=int,
        default=0,
        help="Skip this many saved GitHub leads before processing. Useful for staged cohorts.",
    )
    parser.add_argument(
        "--location-filter",
        default="",
        help="Fixed Recruiter location filter for the run. In --use-current-search mode this is recorded as metadata only.",
    )
    parser.add_argument(
        "--use-current-search",
        action="store_true",
        help="Assume you have already opened the Recruiter project and set the location filter manually.",
    )
    parser.add_argument(
        "--max-cards",
        type=int,
        default=5,
        help="How many Recruiter result cards to inspect per lead.",
    )
    parser.add_argument(
        "--skip-profile-open",
        action="store_true",
        help="Stop before opening the matched profile (disables holistic fit and auto-save).",
    )
    parser.add_argument(
        "--linkedin-brief",
        help="Explicit path to the canonical LinkedIn brief JSON (fit authority). "
        "If omitted, a sibling brief is resolved from the GitHub run manifest.",
    )
    parser.add_argument(
        "--github-brief",
        help="Override path to the GitHub brief JSON when run-manifest.json is missing or wrong.",
    )
    parser.add_argument(
        "--dry-run-save",
        action="store_true",
        help="If reconciliation reaches SAVE, skip the Recruiter save click (for testing).",
    )
    parser.add_argument(
        "--workflow-mode",
        choices=list(ALLOWED_WORKFLOW_MODES),
        default=WORKFLOW_MODE_IDENTITY_COLLECT,
        help=(
            "Operator workflow mode. "
            "'identity_collect' (default) collects identity-confirmed Recruiter profiles "
            "into the current project regardless of brief fit; "
            "'fit_gated_save' preserves the legacy fit+engagement gated save behavior."
        ),
    )
    parser.add_argument(
        "--per-lead-timeout-seconds",
        type=float,
        default=240.0,
        help=(
            "Maximum wall-clock seconds to spend resolving one lead before the runner "
            "calls browser.check_and_recover(), retries once, and then records a "
            "tool_failure failure row to continue the cohort."
        ),
    )
    parser.add_argument(
        "--query-expansion-policy",
        choices=list(ALLOWED_QUERY_EXPANSION_POLICIES),
        default=DEFAULT_QUERY_EXPANSION_POLICY,
        help=(
            "Recruiter keyword query expansion policy. 'auto' (default) derives "
            "from --workflow-mode and --use-current-search: identity_collect with "
            "use-current-search or a fixed --location-filter runs name-first only "
            "(no enriched fallback); everything else runs the bounded enriched plan. "
            "'name_first' forces name-only. 'enriched' forces the legacy bounded "
            "plan (company/location/title fallbacks)."
        ),
    )
    return parser


def _select_leads_window(leads: list, *, lead_offset: int, max_leads: int) -> list:
    start = max(int(lead_offset or 0), 0)
    size = max(int(max_leads or 0), 0)
    if size == 0:
        return []
    return leads[start : start + size]


def _build_input_stats(
    *,
    batch_stats: dict,
    lead_offset: int,
    max_leads: int,
    target_lead_count: int,
    location_filter: str,
    use_current_search: bool,
    linkedin_brief_path: Path,
) -> dict:
    stats = dict(batch_stats)
    stats["lead_offset"] = max(int(lead_offset or 0), 0)
    stats["requested_max_leads"] = max(int(max_leads or 0), 0)
    stats["target_lead_count"] = max(int(target_lead_count or 0), 0)
    stats["location_filter"] = location_filter
    stats["use_current_search"] = bool(use_current_search)
    stats["linkedin_brief_path"] = str(linkedin_brief_path)
    return stats


def _progress_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "jsonl": output_dir / "recruiter_identity_resolutions.jsonl",
        "csv": output_dir / "recruiter_identity_resolutions.csv",
        "saved_jsonl": output_dir / "recruiter_reconciliation_saved.jsonl",
        "saved_csv": output_dir / "recruiter_reconciliation_saved.csv",
        "summary": output_dir / "recruiter_identity_resolutions_summary.json",
        "metadata": output_dir / "recruiter_identity_run_metadata.json",
    }


def _diff_metadata_keys(existing: dict, incoming: dict) -> list[str]:
    """Return human-readable list of keys whose values differ between two metadata dicts."""
    diffs: list[str] = []
    keys = sorted(set(existing.keys()) | set(incoming.keys()))
    for key in keys:
        prior = existing.get(key, "<missing>")
        current = incoming.get(key, "<missing>")
        if prior != current:
            diffs.append(f"  - {key}: prior={prior!r} new={current!r}")
    return diffs


def _seed_or_validate_run_metadata(path: Path, metadata: dict) -> None:
    if path.exists():
        existing = read_json(path)
        if existing != metadata:
            diffs = _diff_metadata_keys(existing, metadata)
            detail = "\n".join(diffs) if diffs else "(no per-key diff available)"
            raise SystemExit(
                "Output directory already contains reconciliation progress for a different cohort/window.\n"
                "Use a fresh --output-dir or rerun with matching parameters.\n"
                f"Mismatched fields:\n{detail}"
            )
        return
    write_json(path, metadata)


def _detect_code_version_marker() -> str:
    """Best-effort short git revision so resumes notice silent code drift.

    Returns an empty string when git is unavailable or the cwd is not a repo;
    the marker is recorded in run metadata only when truthy.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _load_existing_progress_rows(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    if not rows:
        return []
    seen: set[str] = set()
    deduped: list[dict] = []
    for row in rows:
        username = str(row.get("github_username", "") or "").strip()
        if not username or username in seen:
            continue
        seen.add(username)
        deduped.append(row)
    return deduped


def _rewrite_progress_artifacts(output_dir: Path, rows: list[dict], *, input_stats: dict) -> None:
    paths = _progress_paths(output_dir)
    write_recruiter_identity_csv(paths["csv"], rows)
    write_recruiter_reconciliation_saved_jsonl(paths["saved_jsonl"], rows)
    write_recruiter_reconciliation_saved_csv(paths["saved_csv"], rows)
    summary_input_stats = dict(input_stats)
    summary_input_stats["processed_leads"] = len(rows)
    write_recruiter_identity_summary(
        paths["summary"],
        rows,
        input_stats=summary_input_stats,
    )


def _append_progress_row(output_dir: Path, rows: list[dict], row: dict, *, input_stats: dict) -> None:
    paths = _progress_paths(output_dir)
    append_jsonl(paths["jsonl"], row)
    rows.append(row)
    _rewrite_progress_artifacts(output_dir, rows, input_stats=input_stats)


async def _run(args: argparse.Namespace) -> None:
    if not args.use_current_search and not args.project_url:
        raise SystemExit("--project-url is required unless --use-current-search is set")

    workflow_mode = str(getattr(args, "workflow_mode", WORKFLOW_MODE_IDENTITY_COLLECT) or WORKFLOW_MODE_IDENTITY_COLLECT)
    if workflow_mode not in ALLOWED_WORKFLOW_MODES:
        raise SystemExit(f"Unknown --workflow-mode: {workflow_mode!r}")

    query_expansion_policy = str(
        getattr(args, "query_expansion_policy", DEFAULT_QUERY_EXPANSION_POLICY)
        or DEFAULT_QUERY_EXPANSION_POLICY
    )
    if query_expansion_policy not in ALLOWED_QUERY_EXPANSION_POLICIES:
        raise SystemExit(f"Unknown --query-expansion-policy: {query_expansion_policy!r}")

    github_output_dir = Path(args.github_output_dir)
    output_dir = Path(args.output_dir) if args.output_dir else github_output_dir
    batch = load_saved_github_reconciliation_batch_with_fallback(github_output_dir)
    leads = _select_leads_window(
        batch.leads,
        lead_offset=args.lead_offset,
        max_leads=args.max_leads,
    )

    # Identity-collect mode does not require a LinkedIn brief or a judger;
    # confirmed identities are collected without holistic-fit evaluation.
    linkedin_brief = None
    linkedin_brief_path: Path | None = None
    if workflow_mode == WORKFLOW_MODE_FIT_GATED_SAVE:
        linkedin_brief_path = resolve_linkedin_brief_path_for_github_run(
            github_output_dir,
            explicit_linkedin_brief=args.linkedin_brief,
            github_brief_path=args.github_brief,
        )
        linkedin_brief = load_brief(str(linkedin_brief_path))
        init_judger(linkedin_brief)
    else:
        if args.linkedin_brief:
            # Operator pinned a brief explicitly; honor the path for provenance even
            # though identity_collect must not call the judger.
            explicit_path = Path(args.linkedin_brief).expanduser().resolve()
            if not explicit_path.is_file():
                raise FileNotFoundError(f"LinkedIn brief not found: {explicit_path}")
            linkedin_brief_path = explicit_path

    input_stats = _build_input_stats(
        batch_stats=batch.stats.to_dict(),
        lead_offset=args.lead_offset,
        max_leads=args.max_leads,
        target_lead_count=len(leads),
        location_filter=args.location_filter,
        use_current_search=args.use_current_search,
        linkedin_brief_path=linkedin_brief_path or Path(""),
    )
    input_stats["workflow_mode"] = workflow_mode

    # Build the static portion of the run metadata up front. recruiter_url_after_attach
    # is added below once the browser has attached to the Recruiter project, so the
    # ENTIRE metadata dict (including the URL) is written exactly once and validated
    # against the same shape on resume. Do not mutate this dict after validation.
    run_metadata: dict = {
        "workflow_mode": workflow_mode,
        "github_output_dir": str(github_output_dir.resolve()),
        "project_url": str(args.project_url or ""),
        "lead_offset": max(int(args.lead_offset or 0), 0),
        "requested_max_leads": max(int(args.max_leads or 0), 0),
        "target_lead_count": len(leads),
        "location_filter": args.location_filter,
        "use_current_search": bool(args.use_current_search),
        "max_cards": max(int(args.max_cards or 0), 1),
        "skip_profile_open": bool(args.skip_profile_open),
        "dry_run_save": bool(args.dry_run_save),
        "query_expansion_policy": query_expansion_policy,
    }
    if linkedin_brief_path is not None:
        run_metadata["linkedin_brief_path"] = str(linkedin_brief_path)
    code_version = _detect_code_version_marker()
    if code_version:
        run_metadata["code_version"] = code_version
    paths = _progress_paths(output_dir)
    rows = _load_existing_progress_rows(paths["jsonl"])
    processed_usernames = {
        str(row.get("github_username", "") or "").strip()
        for row in rows
        if str(row.get("github_username", "") or "").strip()
    }
    if rows:
        _rewrite_progress_artifacts(output_dir, rows, input_stats=input_stats)
        print(f"Resuming existing reconciliation output: {len(rows)} lead(s) already recorded.")

    # P8.1: this is the canonical reconciliation entry point (see
    # linkedin/reconciliation.py's deprecation note) and it opens Recruiter
    # profiles for every lead — it must be governed like the sourcing
    # pipeline so its opens count against the shared 24h profile-open budget
    # in daily_stats.json, not just the sourcing session's own counter.
    browser = LinkedInBrowser(governor=SessionGovernor())
    await browser.connect()
    try:
        resolver = RecruiterIdentityResolver(
            browser=browser,
            project_url=str(args.project_url or ""),
            config=RecruiterResolverConfig(
                max_cards=max(args.max_cards, 1),
                open_profile_on_likely_match=not bool(args.skip_profile_open),
                dry_run_save=bool(args.dry_run_save),
                workflow_mode=workflow_mode,
                query_expansion_policy=query_expansion_policy,
            ),
            linkedin_brief=linkedin_brief,
            linkedin_brief_path=str(linkedin_brief_path) if linkedin_brief_path is not None else "",
        )
        if args.use_current_search:
            await resolver.use_existing_search(args.location_filter)
        else:
            await resolver.prepare_search(args.location_filter)

        # Capture the actual Recruiter URL after attach. This becomes part of the
        # FIRST metadata write so resume sees the exact same shape and the
        # validation diff stays meaningful (followups plan §1).
        run_metadata["recruiter_url_after_attach"] = _read_recruiter_url(browser)
        _seed_or_validate_run_metadata(paths["metadata"], run_metadata)

        timeout_seconds = max(float(getattr(args, "per_lead_timeout_seconds", 240.0) or 240.0), 1.0)
        for index, lead in enumerate(leads, start=1):
            if lead.username in processed_usernames:
                print(f"[{index}/{len(leads)}] Skipping {lead.candidate_name} ({lead.username}) — already recorded")
                continue
            print(f"[{index}/{len(leads)}] Resolving {lead.candidate_name} ({lead.username})")
            try:
                result = await _resolve_lead_with_timeout_and_recovery(
                    resolver=resolver,
                    browser=browser,
                    lead=lead,
                    timeout_seconds=timeout_seconds,
                )
                row = build_recruiter_identity_row(result)
            except ResolverLeadFailure as exc:
                print(f"  [failure-row] {lead.username}: {exc}")
                row = _build_failure_row(
                    lead=lead,
                    workflow_mode=workflow_mode,
                    linkedin_brief_path=linkedin_brief_path,
                    note=str(exc),
                )
            _append_progress_row(output_dir, rows, row, input_stats=input_stats)
            processed_usernames.add(lead.username)

        print(f"Wrote {paths['jsonl']}")
        print(f"Wrote {paths['csv']}")
        print(f"Wrote {paths['saved_jsonl']}")
        print(f"Wrote {paths['saved_csv']}")
        print(f"Wrote {paths['summary']}")
    finally:
        await browser.disconnect()


def _read_recruiter_url(browser: LinkedInBrowser) -> str:
    """Best-effort current page URL from the browser. Returns empty string on any failure."""
    try:
        page = getattr(browser, "page", None)
        if page is None:
            return ""
        url = getattr(page, "url", "")
        return str(url or "").strip()
    except Exception:
        return ""


def _build_failure_row(
    *,
    lead,
    workflow_mode: str,
    linkedin_brief_path: Path | None,
    note: str,
) -> dict:
    """Synthesize a tool_failure failure row for a lead the resolver could not finish.

    Plan §7: when a lead times out (or both attempts raise unrecoverable browser
    errors), the runner records a tool_failure row and continues the cohort.
    The row carries enough provenance that resume + downstream summary counts
    can recognize it (identity_status / collection_action / project_save_state),
    plus a one-line note explaining the failure.
    """
    from github.recruiter_identity_report import build_recruiter_identity_row
    from shared.recruiter_identity_schemas import RecruiterIdentityResolution

    resolution = RecruiterIdentityResolution(
        github_username=getattr(lead, "username", "") or "",
        candidate_name=getattr(lead, "candidate_name", "") or "",
        lookup_name=getattr(lead, "candidate_name", "") or "",
        github_url=getattr(lead, "github_url", "") or "",
        github_company=getattr(lead, "company", "") or "",
        github_location=getattr(lead, "location", "") or "",
        github_title=getattr(lead, "title", "") or "",
        workflow_mode=workflow_mode,
        identity_status="tool_failure",
        identity_subreason="resolver_failure",
        collection_action="MANUAL_REVIEW",
        collection_subreason="tool_failure",
        project_save_state="not_attempted",
        identity_classification="resolver_failure",
        final_action="MANUAL_REVIEW",
        final_subreason="tool_failure",
        notes=[note],
        linkedin_brief_path=str(linkedin_brief_path) if linkedin_brief_path else "",
    )
    return build_recruiter_identity_row(resolution)


async def _resolve_lead_with_timeout_and_recovery(
    *,
    resolver: RecruiterIdentityResolver,
    browser: LinkedInBrowser,
    lead,
    timeout_seconds: float,
):
    """Run ``resolver.resolve_lead(lead)`` with one timeout-and-recover retry.

    Behavior (plan §7):
      - First attempt: ``asyncio.wait_for(resolver.resolve_lead(lead), timeout)``.
      - On timeout or browser exception, call ``browser.check_and_recover()``
        once and retry once with the same timeout.
      - On second failure, raise ``ResolverLeadFailure`` so the caller can record
        a tool_failure row and continue the cohort.
    """
    try:
        return await asyncio.wait_for(resolver.resolve_lead(lead), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        first_failure = (
            f"first attempt timed out after {timeout_seconds:.0f}s; calling "
            "browser.check_and_recover() and retrying once"
        )
        print(f"  [recovery] {first_failure}")
    except Exception as exc:
        first_failure = f"first attempt raised {type(exc).__name__}: {exc!r}; retrying once"
        print(f"  [recovery] {first_failure}")

    try:
        await browser.check_and_recover()
    except Exception as exc:
        raise ResolverLeadFailure(
            f"check_and_recover raised {type(exc).__name__}: {exc!r} after first-attempt failure"
        ) from exc

    try:
        return await asyncio.wait_for(resolver.resolve_lead(lead), timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise ResolverLeadFailure(
            f"retry also timed out after {timeout_seconds:.0f}s"
        ) from exc
    except Exception as exc:
        raise ResolverLeadFailure(
            f"retry raised {type(exc).__name__}: {exc!r}"
        ) from exc


class ResolverLeadFailure(RuntimeError):
    """Raised when a lead cannot be resolved even after one recovery+retry attempt.

    The runner catches this, records a synthesized tool_failure row, and moves
    on to the next lead so a hung browser does not abort the whole cohort.
    """


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
