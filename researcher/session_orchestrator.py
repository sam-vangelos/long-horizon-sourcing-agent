"""CLI ownership for bounded, resumable frontier researcher sessions.

The state-directory lock covers session creation, execution and publication.
Offline transports exercise this same entry point in acceptance tests."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

from researcher.orchestrator import ResearcherPipeline, build_pipeline


logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="researcher.session_orchestrator",
        description="Researcher session orchestrator (Slice 6).",
    )
    parser.add_argument(
        "--brief",
        required=True,
        help="Path to the brief JSON.",
    )
    parser.add_argument(
        "--state-dir",
        required=True,
        help="Per-source state directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run.",
    )
    parser.add_argument("--check", action="store_true", help="Validate offline without state writes or provider requests.")
    parser.add_argument("--replan", action="store_true", help="Explicitly resume under a revised brief.")
    parser.add_argument("--revisit", metavar="IDENTITY_KEY", help="Explicitly reopen one non-save review on resume.")
    parser.add_argument("--max-queries", type=int, help="Campaign logical query cap (default 12).")
    parser.add_argument("--max-candidates", type=int, help="Campaign evaluation cap (default 50).")
    parser.add_argument("--max-cost-usd", type=float, help="Required campaign dollar cap; inherited on resume.")
    parser.add_argument("--max-minutes", type=float, help="Session time cap (default 30 minutes).")
    return parser


def main(argv: Sequence[str] | None = None, *, transport_overrides: dict | None = None) -> int:
    """Own one bounded session. Test transports use the same CLI and production loop."""
    import signal
    import sqlite3
    from researcher.health import frontier_preflight
    from shared.runtime_state.lock import RuntimeStateLock

    args = _build_arg_parser().parse_args(argv)
    overrides = dict(transport_overrides or {})
    identity_db_path = overrides.pop("identity_db_path", None)
    limits = {key: getattr(args, key) for key in ("max_queries", "max_candidates", "max_cost_usd", "max_minutes")}
    def preflight():
        checked = frontier_preflight(
            brief_path=args.brief, state_dir=args.state_dir, limits=limits,
            resume=args.resume, replan=args.replan, revisit=args.revisit,
            check_providers=transport_overrides is None,
        )
        if transport_overrides is not None:
            required = {"facial_llm_caller", "full_llm_caller"}
            if not overrides.get("page_fetcher"):
                required.add("openalex_client")
                if any(q["provider"] == "exa" for q in checked["plan"].generated_strings):
                    required.update({"exa_client_factory", "public_extract_caller"})
            if any(overrides.get(key) is None for key in required):
                raise ValueError("Offline CLI requires all selected provider and judgment transports")
        return checked
    try:
        checked = preflight()
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f"researcher: preflight refused: {exc}", file=sys.stderr)
        return 2
    if args.check:
        print("researcher: configuration valid; live connectivity and named-venue resolution unchecked.")
        return 0

    lock = RuntimeStateLock(args.state_dir, resource_name="researcher state")
    try:
        lock.acquire()
    except (RuntimeError, OSError) as exc:
        print(f"researcher: state ownership refused: {exc}", file=sys.stderr)
        return 2
    stopped = False
    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True
    original_handlers = {}
    pipeline = None
    run_id = None
    try:
        checked = preflight()  # Recheck ownership/hash after taking the lock.
        for sig in (signal.SIGINT, signal.SIGTERM):
            original_handlers[sig] = signal.signal(sig, request_stop)
        test_stop = overrides.pop("stop_requested", lambda: False)
        print(f"researcher.session_orchestrator: starting brief={args.brief} state_dir={args.state_dir} resume={args.resume}")
        pipeline, run_id = build_pipeline(
            brief_path=Path(args.brief), state_dir=Path(args.state_dir),
            openalex_polite_pool_email=os.environ.get("OPENALEX_POLITE_POOL_EMAIL", ""),
            resume=args.resume, replan=args.replan, revisit=args.revisit,
            limits=checked["limits"], campaign_brief_id=checked["brief"].id, stop_requested=lambda: stopped or test_stop(), **overrides,
        )
        stats = pipeline.run(run_id=run_id)
        try:
            from shared.identity_resolution_service import resolve_persons_for_brief
            resolve_persons_for_brief(pipeline.bridge.brief_id, identity_db_path=Path(identity_db_path) if identity_db_path else None,
                                       state_dirs=[("researcher", Path(args.state_dir))])
        except Exception as exc:
            pipeline.bridge.store.record_event(run_id=run_id, event_type="researcher_identity_degradation", payload={"error": type(exc).__name__})
        print(f"researcher.session_orchestrator: done. queries={stats.queries_completed}/{stats.queries_total} discovered={stats.candidates_discovered} saves={stats.saves} review={stats.reviews} rejects={stats.rejects} failures={stats.failures} stop={stats.stop_reason}")
        run = pipeline.bridge.store.get_run(run_id)
        return 1 if run["status"] == "error" or stats.artifact_error else 0
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f"researcher: {'runtime failure' if run_id else 'setup refusal'} ({type(exc).__name__})", file=sys.stderr)
        return 1 if run_id else 2
    except KeyboardInterrupt:
        if pipeline and run_id:
            pipeline.bridge.store.finish_run(run_id, "interrupted", stop_reason="operator_stop")
        return 0
    except Exception as exc:
        print(f"researcher: runtime failure ({type(exc).__name__})", file=sys.stderr)
        return 1
    finally:
        for sig, handler in original_handlers.items():
            signal.signal(sig, handler)
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
