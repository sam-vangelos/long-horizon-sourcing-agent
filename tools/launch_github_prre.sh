#!/usr/bin/env bash
# GitHub PRRE: route Anthropic tiers through claude-cli seat OAuth; facial judgments remain metered Fireworks.
# Seat exhaustion raises ApiBudgetExhaustedError and stops the day cycle with resumable state.
#
# Usage:
#   tools/launch_github_prre.sh         # preflight only (default)
#   tools/launch_github_prre.sh run     # preflight, then detached fresh run
#   tools/launch_github_prre.sh resume  # preflight, then detached resume
#
# Use this launcher to pin the CLI transport; bare defaults select metered API paths.
set -euo pipefail

MODE="check"
for tok in "${@:-}"; do
  case "$tok" in
    ""|--check|check) ;;
    run)    MODE="run" ;;
    resume) MODE="resume" ;;
    *) echo "usage: $0 [run|resume|--check]  (unknown: $tok)" >&2; exit 2 ;;
  esac
done

# The campaign's state directory is durable across sessions so a resume keeps
# its history; override STATE_DIR to point at an existing campaign.
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$REPO/.venv/bin/python}"
BRIEF="$REPO/config/prre-code/brief-prre-code-github-v2.json"
STATE_DIR="${STATE_DIR:-$REPO/output/state/github/prre-code-cli}"
LOG_DIR="$STATE_DIR/launch-logs"

cd "$REPO"

# Validate resolved interpreter settings because inherited environment variables outrank dotenv.
export STRATEGY_MODEL_NAME="claude-cli:claude-fable-5-1"
export OPUS_MODEL_NAME="claude-cli:claude-opus-4-8"
export FULL_EVAL_MODEL_NAME="claude-cli:claude-opus-4-8"
export CHEAP_MODEL_PROVIDER="claude_cli"
export CHEAP_MODEL_NAME="claude-haiku-4-5-20251001"
# Keep facial evaluation on the batched Fireworks model.
export FACIAL_MODEL_NAME=claude-opus-5

echo "=== preflight: routing resolves as the run will see it ==="
"$PY" - <<'PYEOF'
import sys
sys.path.insert(0, ".")
from shared import config

want = {
    "STRATEGY_MODEL_NAME": "claude-cli:claude-fable-5-1",
    "OPUS_MODEL_NAME": "claude-cli:claude-opus-4-8",
    "FULL_EVAL_MODEL_NAME": "claude-cli:claude-opus-4-8",
    "CHEAP_MODEL_PROVIDER": "claude_cli",
    "CHEAP_MODEL_NAME": "claude-haiku-4-5-20251001",
    "FACIAL_MODEL_NAME": "claude-opus-5",
}
bad = []
for key, expected in want.items():
    got = getattr(config, key, None)
    mark = "ok " if got == expected else "BAD"
    print(f"  {mark} {key:<24} {got}")
    if got != expected:
        bad.append((key, got, expected))
if bad:
    print("\nNO-GO — routing drift between this launcher and what config resolved:")
    for key, got, expected in bad:
        print(f"  {key}: got {got!r}, want {expected!r}")
    raise SystemExit(1)

# Startup requires ANTHROPIC_API_KEY in the parent; the CLI transport must scrub it from every child.
for key in ("ANTHROPIC_API_KEY",):
    val = getattr(config, key, None)
    print(f"  {'ok ' if val else 'BAD'} {key:<24} {'present' if val else 'MISSING'}")
    if not val:
        raise SystemExit(1)
PYEOF

echo
echo "=== preflight: GitHub token live ==="
"$PY" - <<'PYEOF'
import json, sys, urllib.request
sys.path.insert(0, ".")
from github import config as gh_config

tok = gh_config.GITHUB_TOKEN
if not tok:
    print("  BAD GITHUB_TOKEN missing")
    raise SystemExit(1)
req = urllib.request.Request(
    "https://api.github.com/rate_limit",
    headers={
        "Authorization": f"Bearer {tok}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "cloris-preflight",
    },
)
try:
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.load(resp)
except Exception as exc:
    print(f"  BAD GITHUB_TOKEN rejected: {type(exc).__name__}: {exc}")
    raise SystemExit(1)
for name in ("core", "search", "code_search"):
    bucket = data["resources"].get(name)
    if bucket:
        print(f"  ok  {name:<12} {bucket['remaining']}/{bucket['limit']}")
PYEOF

echo
echo "=== preflight: claude CLI seat (one tiny live call) ==="
# Probe the real CLI transport before launch to verify the binary, OAuth login, environment scrub and envelope.
# The probe consumes seat usage and must refuse a drained or logged-out seat.
"$PY" - <<'PYEOF'
import sys, time
sys.path.insert(0, ".")
from shared.claude_cli_transport import claude_cli_chat

t0 = time.time()
ctx: dict = {}
result = claude_cli_chat(
    model="claude-opus-4-8",
    system_prompt="You are a preflight probe.",
    user_prompt="Reply with exactly: OK",
    expect_json=False,
    usage_context=ctx,
)
wall = time.time() - t0
ok = result.strip() == "OK"
print(f"  {'ok ' if ok else 'BAD'} seat round-trip {wall:.1f}s  result={result.strip()!r}"
      f"  cost-equiv=${ctx.get('cli_reported_cost_usd', 0):.4f}")
if not ok:
    raise SystemExit(1)
PYEOF

if [[ "$MODE" == "check" ]]; then
  echo
  echo "Preflight clean. Launch with: $0 run"
  exit 0
fi

RESUME_FLAG=""
if [[ "$MODE" == "resume" ]]; then
  RESUME_FLAG="--resume"
  if [[ ! -f "$STATE_DIR/runtime_state.sqlite3" ]]; then
    echo "NO-GO — resume requested but $STATE_DIR has no runtime_state.sqlite3" >&2
    exit 1
  fi
else
  if [[ -f "$STATE_DIR/runtime_state.sqlite3" ]]; then
    echo "NO-GO — $STATE_DIR already holds a run. Use: $0 resume" >&2
    exit 1
  fi
fi

mkdir -p "$STATE_DIR" "$LOG_DIR"
RUN_LOG="$LOG_DIR/run.log"

# Keep the machine awake and detach with os.setsid so harness exit cannot reap the campaign.
LAUNCH_PID=$("$PY" - "$RUN_LOG" "$REPO" "$PY" "$BRIEF" "$STATE_DIR" "$RESUME_FLAG" <<'PYEOF'
import os, subprocess, sys

log_path, cwd, py, brief, state_dir, resume_flag = sys.argv[1:7]
argv = ["caffeinate", "-dimsu", py, "-u", "-m", "github.session_orchestrator",
        "--brief", brief, "--state-dir", state_dir]
if resume_flag:
    argv.append(resume_flag)
log = open(log_path, "ab", buffering=0)
proc = subprocess.Popen(
    argv, cwd=cwd, stdout=log, stderr=log,
    stdin=subprocess.DEVNULL, preexec_fn=os.setsid,
)
print(proc.pid)
PYEOF
)

echo
echo "Launched detached. PID $LAUNCH_PID"
echo
echo "  watch the run:   tail -f $RUN_LOG"
echo "  watch the burn:  watch -n 60 \"$PY -c \\\"
import json
tot = {}
for line in open('$STATE_DIR/token-cost-log.jsonl'):
    r = json.loads(line)
    p = r.get('provider')
    tot[p] = tot.get(p, 0.0) + float(r.get('estimated_cost_usd') or 0)
for p, c in sorted(tot.items()):
    print(f'{p:<12} \\\\\${c:.4f}')\\\"\""
echo "  stop cleanly:    kill $LAUNCH_PID   (state resumes with: $0 resume)"
