"""Pluggable ``recruiter_id`` resolver — the auth seam for reopen Stage 2.

The reopen decision (LOCKED, Sam 2026-06-01): the recruiter, not the
brief, is Cloris's durable entity. Every double-write site that records a
recruiter-scoped signal (taste signals, brief membership) needs to know
*which* recruiter is acting. In Stage 1 there is exactly one implicit
recruiter — Sam — so the default resolver returns recruiter id ``1``.

Why a resolver seam instead of a hardcoded ``1`` everywhere
(adversarial-ledger flaw "backfill-corruption", major):
hardcoding ``recruiter_id=1`` at every write site would strand all of
Stage 1's accreted data the moment Phase 2 wires real auth and the
authenticated principal resolves to a *different* id. The seam means
Phase 2 swaps one function (:func:`set_recruiter_id_resolver`) and every
write site follows; the init guard in
``recruiter_store.RecruiterStore.initialize`` refuses to open a store
whose existing rows were written under a different id than the resolver
now returns, pointing the operator at the Phase-2 migration tool rather
than silently mixing two recruiters' data.

Stage-2 callers MUST call :func:`get_current_recruiter_id` — never a
literal ``1`` — so the swap is total.
"""

from __future__ import annotations

from typing import Callable

# The canonical Stage-1 handle. The default resolver get-or-creates this
# recruiter (idempotently) so the recruiter store is never queried for an
# id that doesn't exist yet. Phase 2 auth replaces the whole resolver, at
# which point this constant is only the Stage-1 seed.
DEFAULT_RECRUITER_HANDLE = "operator@example.com"
DEFAULT_RECRUITER_DISPLAY_NAME = "Sam"

# Stage-1 implicit-recruiter id. ``upsert_recruiter`` on a fresh store
# returns ``1`` (AUTOINCREMENT from empty), and the default resolver
# bootstraps that row, so this is the id every Stage-1 write lands under.
STAGE1_RECRUITER_ID = 1


def _default_recruiter_id_resolver() -> int:
    """Stage-1 default: get-or-create Sam, return his id (``1``).

    Self-bootstrapping rather than relying on a separate control-plane
    init step: ``cloris.control_plane`` is a deliberately read-only
    aggregator that refuses to import a DDL-running store class (see its
    module docstring), so the recruiter bootstrap can't live there
    without breaking that invariant. Doing the idempotent upsert here
    keeps the seam self-contained — the first call to
    :func:`get_current_recruiter_id` materializes the recruiter row, and
    every later call is a cheap get.

    Fail-soft: if the store can't be opened (e.g. a read-only filesystem
    in a frozen bundle before user-data is writable), fall back to the
    Stage-1 id constant rather than raising into a write path that has
    already done its primary work.
    """

    try:
        from shared.output_paths import resolve_recruiter_db_path
        from shared.runtime_state.recruiter_store import RecruiterStore

        store = RecruiterStore(resolve_recruiter_db_path())
        return store.upsert_recruiter(
            DEFAULT_RECRUITER_HANDLE,
            display_name=DEFAULT_RECRUITER_DISPLAY_NAME,
        )
    except Exception:  # noqa: BLE001 — resolver must never break a write path
        return STAGE1_RECRUITER_ID


# Module-level resolver slot. Swapped wholesale by Phase 2 auth via
# :func:`set_recruiter_id_resolver`; defaults to the Stage-1 bootstrapper.
_resolver: Callable[[], int] = _default_recruiter_id_resolver


def get_current_recruiter_id() -> int:
    """Return the id of the recruiter currently acting.

    Stage 1: the implicit recruiter (Sam) → ``1``. Phase 2: the
    authenticated principal, once :func:`set_recruiter_id_resolver` has
    installed an auth-backed resolver. Every recruiter-scoped write site
    routes through here so the Phase-2 swap is a single point of change.
    """

    return int(_resolver())


def set_recruiter_id_resolver(fn: Callable[[], int]) -> None:
    """Install the resolver Phase-2 auth (or a test) uses.

    ``fn`` takes no arguments and returns the acting recruiter's id. The
    swap is process-wide and total — every subsequent
    :func:`get_current_recruiter_id` uses ``fn``.
    """

    global _resolver
    _resolver = fn


def reset_recruiter_id_resolver() -> None:
    """Restore the Stage-1 default resolver. Primarily for test teardown."""

    global _resolver
    _resolver = _default_recruiter_id_resolver
