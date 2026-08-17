"""Phase G Slice G4: Tools execution runtime.

In-process async job queue. Each tool execution gets a job_id; sync tools
block in the request handler and return JSON; async tools spawn a detached
asyncio subprocess and return ``{job_id}``. The frontend polls
``GET /api/tools/jobs/<job_id>`` for status + truncated stdout/stderr.

We deliberately don't pull in Celery / RQ / a real queue:

- Cloris is single-process FastAPI by design. A real queue means a
  second daemon, a redis broker, deployment complexity. None of that
  earns its keep at one-recruiter-tool scale.
- The job state is in-memory — a process restart loses jobs in flight.
  That's the right tradeoff. If we needed durable jobs we'd pick a real
  queue; we don't.

Subprocess invocation hardening:
- ``asyncio.create_subprocess_exec`` (NOT ``shell=True``).
- argv is built from validated Pydantic fields by the registry's
  ``argv_builder``; user-supplied strings never reach the shell.
- stdout/stderr are captured; first 16 KB visible via the status route
  so the UI can show progress without leaking gigabytes.
- Jobs auto-purge after 1h to bound memory.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from cloris.tools_registry import ToolDefinition


JobStatus = Literal["queued", "running", "succeeded", "failed", "purged"]


@dataclass
class _Job:
    job_id: str
    tool_id: str
    status: JobStatus
    started_at: float
    finished_at: float | None = None
    exit_code: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    error_message: str | None = None
    task: asyncio.Task | None = field(default=None, repr=False)


class _JobRegistry:
    """In-memory job table, async-safe.

    Single shared instance per process; created lazily on first use to
    avoid pulling an event loop into module import.
    """

    PURGE_AGE_SECONDS = 60 * 60  # 1 hour
    OUTPUT_TAIL_BYTES = 16 * 1024  # 16 KB per stream

    def __init__(self) -> None:
        self._jobs: dict[str, _Job] = {}
        self._lock = asyncio.Lock()

    async def create(self, tool_id: str) -> _Job:
        job_id = uuid.uuid4().hex
        async with self._lock:
            job = _Job(
                job_id=job_id,
                tool_id=tool_id,
                status="queued",
                started_at=time.time(),
            )
            self._jobs[job_id] = job
            self._purge_expired_locked()
        return job

    async def get(self, job_id: str) -> _Job | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def update(self, job_id: str, **fields) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for k, v in fields.items():
                setattr(job, k, v)

    def _purge_expired_locked(self) -> None:
        cutoff = time.time() - self.PURGE_AGE_SECONDS
        for job_id, job in list(self._jobs.items()):
            if job.finished_at is not None and job.finished_at < cutoff:
                del self._jobs[job_id]


_registry: _JobRegistry | None = None


def _get_registry() -> _JobRegistry:
    global _registry
    if _registry is None:
        _registry = _JobRegistry()
    return _registry


def _truncate_tail(buf: bytes, *, limit: int) -> str:
    """Decode + tail-truncate a captured stream for the status payload."""

    text = buf.decode("utf-8", errors="replace")
    return _truncate_text_tail(text, limit=limit)


def _truncate_text_tail(text: str, *, limit: int) -> str:
    """Tail-truncate an already decoded stream for the status payload."""

    if len(text) > limit:
        return "…" + text[-limit:]
    return text


def _tools_repo_root() -> Path:
    """Project root — used as cwd for subprocess invocations.

    Tools assume cwd is the repo root (config/, output/ are relative).
    """

    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Public API consumed by cloris/api.py.
# ---------------------------------------------------------------------------


class ToolRunSyncResult(BaseModel):
    """Wire shape for a sync tool execution."""

    tool_id: str
    exit_code: int
    stdout_tail: str
    stderr_tail: str


class ToolRunAsyncResult(BaseModel):
    """Wire shape returned when a tool is queued asynchronously."""

    tool_id: str
    job_id: str


class ToolJobStatus(BaseModel):
    """Wire shape for ``GET /api/tools/jobs/{job_id}``."""

    job_id: str
    tool_id: str
    status: JobStatus
    started_at: float
    finished_at: float | None
    exit_code: int | None
    stdout_tail: str
    stderr_tail: str
    error_message: str | None


async def execute_sync(tool: ToolDefinition, args_model: BaseModel) -> ToolRunSyncResult:
    """Run a sync tool to completion in the request handler.

    Use only for tools whose worst case is < 10s. Anything longer should
    be ``execution_model="async"`` so the request doesn't time out.
    """

    if tool.runner is not None:
        exit_code, stdout, stderr = await asyncio.to_thread(tool.runner, args_model)
        return ToolRunSyncResult(
            tool_id=tool.tool_id,
            exit_code=exit_code,
            stdout_tail=_truncate_text_tail(
                stdout, limit=_JobRegistry.OUTPUT_TAIL_BYTES
            ),
            stderr_tail=_truncate_text_tail(
                stderr, limit=_JobRegistry.OUTPUT_TAIL_BYTES
            ),
        )

    if tool.argv_builder is None:
        raise ValueError(f"tool {tool.tool_id!r} has no argv_builder")

    argv = tool.argv_builder(args_model)
    full_argv = [sys.executable, *argv]
    proc = await asyncio.create_subprocess_exec(
        *full_argv,
        cwd=str(_tools_repo_root()),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return ToolRunSyncResult(
        tool_id=tool.tool_id,
        exit_code=int(proc.returncode or 0),
        stdout_tail=_truncate_tail(stdout, limit=_JobRegistry.OUTPUT_TAIL_BYTES),
        stderr_tail=_truncate_tail(stderr, limit=_JobRegistry.OUTPUT_TAIL_BYTES),
    )


async def execute_async(tool: ToolDefinition, args_model: BaseModel) -> ToolRunAsyncResult:
    """Queue an async tool and return immediately with a job_id."""

    if tool.runner is None and tool.argv_builder is None:
        raise ValueError(f"tool {tool.tool_id!r} has no argv_builder")

    registry = _get_registry()
    job = await registry.create(tool.tool_id)

    async def _runner() -> None:
        await registry.update(job.job_id, status="running")
        try:
            if tool.runner is not None:
                returncode, stdout_text, stderr_text = await asyncio.to_thread(
                    tool.runner,
                    args_model,
                )
                stdout_tail = _truncate_text_tail(
                    stdout_text, limit=_JobRegistry.OUTPUT_TAIL_BYTES
                )
                stderr_tail = _truncate_text_tail(
                    stderr_text, limit=_JobRegistry.OUTPUT_TAIL_BYTES
                )
            else:
                argv_builder = tool.argv_builder
                if argv_builder is None:
                    raise ValueError(f"tool {tool.tool_id!r} has no argv_builder")
                argv = argv_builder(args_model)
                full_argv = [sys.executable, *argv]
                proc = await asyncio.create_subprocess_exec(
                    *full_argv,
                    cwd=str(_tools_repo_root()),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()
                returncode = int(proc.returncode or 0)
                stdout_tail = _truncate_tail(
                    stdout, limit=_JobRegistry.OUTPUT_TAIL_BYTES
                )
                stderr_tail = _truncate_tail(
                    stderr, limit=_JobRegistry.OUTPUT_TAIL_BYTES
                )
            await registry.update(
                job.job_id,
                status="succeeded" if returncode == 0 else "failed",
                exit_code=returncode,
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
                finished_at=time.time(),
            )
        except Exception as exc:
            await registry.update(
                job.job_id,
                status="failed",
                exit_code=-1,
                error_message=str(exc),
                finished_at=time.time(),
            )

    task = asyncio.create_task(_runner())
    await registry.update(job.job_id, task=task)
    return ToolRunAsyncResult(tool_id=tool.tool_id, job_id=job.job_id)


async def get_job_status(job_id: str) -> ToolJobStatus | None:
    registry = _get_registry()
    job = await registry.get(job_id)
    if job is None:
        return None
    return ToolJobStatus(
        job_id=job.job_id,
        tool_id=job.tool_id,
        status=job.status,
        started_at=job.started_at,
        finished_at=job.finished_at,
        exit_code=job.exit_code,
        stdout_tail=job.stdout_tail,
        stderr_tail=job.stderr_tail,
        error_message=job.error_message,
    )
