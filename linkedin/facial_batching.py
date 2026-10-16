"""Pure bounded-concurrency coordination for LinkedIn facial judgments.

This module deliberately knows nothing about browsers, runtime state, logging,
or model providers.  The orchestrator owns those side effects; this helper only
partitions an already-extracted page, runs the supplied blocking judge in
worker threads, validates attribution, and restores input order.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast


InputT = TypeVar("InputT")
ResultT = TypeVar("ResultT")


class FacialBatchContractError(RuntimeError):
    """A worker returned a result shape that cannot be attributed safely."""


@dataclass(frozen=True, slots=True)
class FacialBatchFailureOutcome(Generic[InputT]):
    """One failed result slot, retaining its original candidate identity."""

    candidate: InputT
    candidate_identity: object | None
    error: BaseException
    batch_index: int


@dataclass(frozen=True, slots=True)
class FacialBatchSlice(Generic[InputT]):
    """One contiguous, stable slot in a page-level facial dispatch."""

    index: int
    start: int
    stop: int
    snippets: tuple[InputT, ...]

    @property
    def size(self) -> int:
        return self.stop - self.start


def partition_facial_batches(
    snippets: Sequence[InputT],
    *,
    max_concurrency: int,
    target_batch_size: int,
) -> tuple[FacialBatchSlice[InputT], ...]:
    """Return balanced contiguous slices, placing remainder items last.

    ``25`` inputs with concurrency ``3`` and target size ``8`` become
    ``8/8/9``.  Fewer than one target batch stay in one call.  The target is a
    sizing hint rather than a hard ceiling because the concurrency cap wins.
    """

    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    if target_batch_size <= 0:
        raise ValueError("target_batch_size must be positive")

    total = len(snippets)
    if total == 0:
        return ()

    batch_count = min(max_concurrency, math.ceil(total / target_batch_size))
    base_size, remainder = divmod(total, batch_count)
    sizes = [base_size] * (batch_count - remainder) + [base_size + 1] * remainder

    batches: list[FacialBatchSlice[InputT]] = []
    start = 0
    for index, size in enumerate(sizes):
        stop = start + size
        batches.append(
            FacialBatchSlice(
                index=index,
                start=start,
                stop=stop,
                snippets=tuple(snippets[start:stop]),
            )
        )
        start = stop
    return tuple(batches)


def _profile_identity(value: object) -> object | None:
    return getattr(value, "profile_url", None)


def _validate_attribution(
    batch: FacialBatchSlice[InputT],
    results: Sequence[ResultT],
    *,
    input_identity: Callable[[InputT], object | None],
    result_identity: Callable[[ResultT], object | None],
) -> None:
    if len(results) != batch.size:
        raise FacialBatchContractError(
            f"facial batch {batch.index} returned {len(results)} result(s) "
            f"for {batch.size} input(s)"
        )

    expected = [input_identity(item) for item in batch.snippets]
    actual = [result_identity(item) for item in results]
    # Generic tests/callers without identity-bearing objects still get exact
    # cardinality validation.  Production CandidateSnippet/OpusDecision pairs
    # both carry profile_url and therefore always take the stronger branch.
    if all(value is None for value in expected + actual):
        return
    if any(value is None for value in expected + actual):
        raise FacialBatchContractError(
            f"facial batch {batch.index} returned incomplete identity metadata"
        )
    if actual != expected:
        raise FacialBatchContractError(
            f"facial batch {batch.index} result identities do not match input order"
        )


async def run_facial_batches(
    snippets: Sequence[InputT],
    judge_batch: Callable[[list[InputT], dict[str, Any]], Sequence[ResultT]],
    *,
    max_concurrency: int,
    target_batch_size: int,
    base_context: dict[str, Any] | None = None,
    input_identity: Callable[[InputT], object | None] = _profile_identity,
    result_identity: Callable[[ResultT], object | None] = _profile_identity,
) -> list[ResultT | FacialBatchFailureOutcome[InputT]]:
    """Run blocking batch judgments concurrently and merge deterministically.

    ``asyncio.to_thread`` propagates the caller's ContextVars into each worker.
    The gather is shielded so cancellation of the coordinator does not pretend
    to cancel provider work: every started worker is settled before the
    cancellation is re-raised to the orchestrator.
    """

    batches = partition_facial_batches(
        snippets,
        max_concurrency=max_concurrency,
        target_batch_size=target_batch_size,
    )
    if not batches:
        return []

    common = dict(base_context or {})
    batch_count = len(batches)

    async def invoke(batch: FacialBatchSlice[InputT]) -> Sequence[ResultT]:
        context = dict(common)
        context.update(
            {
                "batch_index": batch.index,
                "batch_number": batch.index + 1,
                "batch_count": batch_count,
                "batch_size": batch.size,
                "batch_start": batch.start,
                "batch_stop": batch.stop,
                "batch_slot": batch.index,
            }
        )
        return await asyncio.to_thread(judge_batch, list(batch.snippets), context)

    tasks = [asyncio.create_task(invoke(batch)) for batch in batches]
    settled = asyncio.gather(*tasks, return_exceptions=True)
    try:
        outcomes = await asyncio.shield(settled)
    except asyncio.CancelledError:
        # A thread launched by to_thread keeps running after its asyncio task is
        # cancelled.  Shielding the gather, then awaiting it after delivery of
        # the cancellation, makes the drain contract explicit and testable.
        while not settled.done():
            try:
                await asyncio.shield(settled)
            except asyncio.CancelledError:
                continue
        raise

    unfilled = object()
    merged: list[object] = [unfilled] * len(snippets)
    for batch, outcome in zip(batches, outcomes):
        if isinstance(outcome, BaseException):
            merged[batch.start : batch.stop] = [
                FacialBatchFailureOutcome(
                    candidate=candidate,
                    candidate_identity=input_identity(candidate),
                    error=outcome,
                    batch_index=batch.index,
                )
                for candidate in batch.snippets
            ]
            continue
        try:
            results = list(outcome)
            _validate_attribution(
                batch,
                results,
                input_identity=input_identity,
                result_identity=result_identity,
            )
        except BaseException as exc:
            merged[batch.start : batch.stop] = [
                FacialBatchFailureOutcome(
                    candidate=candidate,
                    candidate_identity=input_identity(candidate),
                    error=exc,
                    batch_index=batch.index,
                )
                for candidate in batch.snippets
            ]
        else:
            merged[batch.start : batch.stop] = results

    if any(result is unfilled for result in merged):
        raise FacialBatchContractError("facial batch merge left an unfilled result slot")
    return cast(
        list[ResultT | FacialBatchFailureOutcome[InputT]],
        merged,
    )
