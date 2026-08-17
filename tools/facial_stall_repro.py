"""Discriminating repro for the Fireworks facial-timeout stall (2026-08-03).

Builds a byte-faithful facial batch request from the app's own code
(system prompt + tool schema verified against production sha256 hashes in
llm-attempts.jsonl), then fires it two ways per round:

  B) RAW  — fresh httpx client per call, no app transport code at all
  A) APP  — the production path: facial_llm -> _fireworks_primary_chat

3 concurrent calls per transport per round, matching production's 3-way
batch fanout. A call is a STALL if it exceeds STALL_AFTER seconds.

Interpretation:
  RAW stalls too            -> provider-side for this request shape
  only APP stalls           -> our client path
  neither stalls repeatedly -> intermittent provider shedding; rerun later

Attempt timeout is capped at 120s (production runs 300s) purely to keep
the loop tight; the production stall class produces null tokens for the
full window, so a 120s cap loses no signal.
"""

import concurrent.futures as cf
import dataclasses
import hashlib
import json
import os
import sys
import time

REPO = "/Users/operator/Personal/cloris"
STATE = REPO + "/output/state/linkedin/3000000003-aiel-20260729"
SCRATCH = os.path.dirname(os.path.abspath(__file__))

# Production reference hashes from llm-attempts.jsonl (2026-08-04T02:15Z entry)
PROD_SYSTEM_SHA = "9297aa4fb3337925c047f9ce4c3f1747a279227d7d5ef19f5501c21f187ae1c2"
PROD_TOOL_SHA = "9e7382876d92dcf02fe9db8e93f6d2f981920c7b088e180fee19b69ee287fcf9"

ATTEMPT_TIMEOUT = 120.0
STALL_AFTER = 60.0
ROUNDS = int(os.environ.get("REPRO_ROUNDS", "3"))
CONCURRENCY = 3

os.chdir(REPO)
sys.path.insert(0, REPO)

from shared import config  # noqa: E402
from shared.brief_loader import load_brief  # noqa: E402
from shared.judgment.templates import (  # noqa: E402
    _facial_ternary_selected,
    assemble_facial_tool_system,
)
from shared.judgment.tool_contracts import (  # noqa: E402
    facial_tool_contract,
    generate_opaque_candidate_ids,
    render_facial_tool_user_message,
)
import shared.judger as judger  # noqa: E402
from shared.llm_clients import facial_llm  # noqa: E402
from shared.llm_usage import llm_usage_session  # noqa: E402

import httpx  # noqa: E402


def build_request():
    brief = load_brief(STATE + "/preflight_v2_brief.json")
    assert brief.has_v2_schema, "brief lost v2 schema"
    system = assemble_facial_tool_system(brief._new_brief, batch=True)
    sys_sha = hashlib.sha256(system.encode("utf-8")).hexdigest()
    contract = facial_tool_contract(
        allow_borderline=_facial_ternary_selected(brief._new_brief)
    )
    tool_sha = hashlib.sha256(
        json.dumps(contract.tool_spec(), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    print(f"system sha  {sys_sha}  match={sys_sha == PROD_SYSTEM_SHA}")
    print(f"tool   sha  {tool_sha}  match={tool_sha == PROD_TOOL_SHA}")

    # Real snippet content from the campaign, rendered to text.
    rows = []
    with open(STATE + "/snippets.jsonl") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    picked = rows[-7:]
    texts = []
    for r in picked:
        parts = [
            f"{r.get('name', '')} — {r.get('current_title', '')}",
            f"Company: {r.get('current_company', '')}",
            f"Headline: {r.get('headline', '')}",
            f"Location: {r.get('location', '')}",
            "Experience: " + "; ".join(r.get("experience_entries") or []),
            f"Education: {r.get('education_snippet', '')}",
        ]
        texts.append("\n".join(p for p in parts if p.strip()))
    ids = generate_opaque_candidate_ids(len(texts))
    user_msg = render_facial_tool_user_message(texts, ids, prompt_prefix="")
    print(f"user_msg chars={len(user_msg)} system chars={len(system)}")
    return system, user_msg, contract


def raw_call(system, user_msg, contract, tag):
    body = {
        "model": config.FACIAL_MODEL_NAME,
        "max_tokens": max(16384, config.FIREWORKS_PRIMARY_MIN_MAX_TOKENS),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 1.0,
        "reasoning_effort": "high",
        "tools": [contract.tool_spec()],
        "tool_choice": contract.forced_choice(),
        "parallel_tool_calls": False,
    }
    t0 = time.monotonic()
    try:
        with httpx.Client(
            timeout=httpx.Timeout(ATTEMPT_TIMEOUT, connect=15.0)
        ) as client:
            resp = client.post(
                "https://api.fireworks.ai/inference/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.FIREWORKS_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
        dt = time.monotonic() - t0
        h = resp.headers
        if resp.status_code != 200:
            return (tag, "HTTP_%d" % resp.status_code, dt, resp.text[:200])
        d = resp.json()
        usage = d.get("usage", {})
        info = (
            f"srv={h.get('fireworks-server-processing-time')}s "
            f"ttft={h.get('fireworks-server-time-to-first-token')}s "
            f"in={usage.get('prompt_tokens')} out={usage.get('completion_tokens')} "
            f"finish={d['choices'][0].get('finish_reason')}"
        )
        verdict = "STALL" if dt > STALL_AFTER else "ok"
        return (tag, verdict, dt, info)
    except Exception as exc:
        dt = time.monotonic() - t0
        verdict = "STALL" if dt > STALL_AFTER else "ERR"
        return (tag, f"{verdict}:{type(exc).__name__}", dt, str(exc)[:200])


def app_call(system, user_msg, contract, tag):
    policy = judger._fireworks_judgment_policy(
        stage="facial",
        system_prompt=system,
        contract_version="linkedin_facial_tool_v1",
        usage_context={"lane_id": "repro", "batch_slot": 0},
    )
    if policy is None:
        return (tag, "NO_POLICY (FIREWORKS_JUDGMENT_POLICY_ENABLED off)", 0.0, "")
    policy = dataclasses.replace(
        policy,
        attempt_timeout_seconds=ATTEMPT_TIMEOUT,
        total_deadline_seconds=ATTEMPT_TIMEOUT + 10.0,
        max_attempts=1,
    )
    t0 = time.monotonic()
    try:
        facial_llm(
            system,
            user_msg,
            expect_json=False,
            max_tokens=16384,
            usage_context={"stage": "facial", "repro": tag},
            policy=policy,
            tool_contract=contract,
        )
        dt = time.monotonic() - t0
        verdict = "STALL" if dt > STALL_AFTER else "ok"
        return (tag, verdict, dt, "completed via app path")
    except Exception as exc:
        dt = time.monotonic() - t0
        verdict = "STALL" if dt > STALL_AFTER else "ERR"
        return (tag, f"{verdict}:{type(exc).__name__}", dt, str(exc)[:200])


def run_wave(fn, system, user_msg, contract, label, rnd):
    with cf.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futs = [
            pool.submit(fn, system, user_msg, contract, f"{label}{rnd}.{i}")
            for i in range(CONCURRENCY)
        ]
        return [f.result() for f in futs]


def main():
    system, user_msg, contract = build_request()
    results = []
    with llm_usage_session(SCRATCH + "/repro-usage.jsonl", harness="facial_repro"):
        for rnd in range(1, ROUNDS + 1):
            for label, fn in (("RAW-", raw_call), ("APP-", app_call)):
                wave = run_wave(fn, system, user_msg, contract, label, rnd)
                for tag, verdict, dt, info in wave:
                    print(f"[{tag}] {verdict:24s} {dt:7.1f}s  {info}", flush=True)
                results.extend(wave)
    stalls = [r for r in results if "STALL" in r[1]]
    print(f"\nTOTAL {len(results)} calls, {len(stalls)} stalls")
    for r in stalls:
        print("  stall:", r[0], r[1], f"{r[2]:.1f}s")


if __name__ == "__main__":
    main()
