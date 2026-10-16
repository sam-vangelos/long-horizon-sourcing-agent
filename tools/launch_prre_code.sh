#!/usr/bin/env bash
# Principal Research Engineer - Code: Recruiter project 9000000005.
# Use the seed for fresh generation; resume requires existing canonical progress.
#
# Usage:
#   tools/launch_prre_code.sh          # fresh, one 4-5h session
#   tools/launch_prre_code.sh multi    # fresh, then keep cycling
#   tools/launch_prre_code.sh resume   # resume, then keep cycling
#   tools/launch_prre_code.sh --check  # preflight, including browser checks
#
# Use multi for unattended cycling; market reporting waits for complete strategy coverage.
# Model routing and inherited settings are pinned by this launcher.
set -euo pipefail

MODE="single"
CHECK_ONLY="no"
# Restart deletes the named strings' candidate attempts so they are re-judged.
# Check candidate counts before choosing IDs; use the launch argument rather than editing projections.
RESTART_IDS=""
for tok in "${@:-}"; do
  case "$tok" in
    ""|single) ;;
    multi)   MODE="multi" ;;
    resume)  MODE="resume" ;;
    --check) CHECK_ONLY="yes" ;;
    restart:*)
      RESTART_IDS="${tok#restart:}"
      if ! printf '%s' "$RESTART_IDS" | grep -Eq '^[0-9]+(,[0-9]+)*$'; then
        echo "usage: restart:<id>[,<id>...] with integer ids (got: $RESTART_IDS)" >&2
        exit 2
      fi
      ;;
    *) echo "usage: $0 [single|multi|resume] [restart:<ids>] [--check]  (unknown: $tok)" >&2; exit 2 ;;
  esac
done

# Execute the checkout containing this script; REPO resolves from its own location.
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$REPO/.venv/bin/python}"
# Inline the JD in the seed because relative jd_path values resolve against CWD.
# Fresh launch must require preflight so evaluation criteria receive the JD.
BRIEF="$REPO/config/prre-code/brief-prre-code-v2.json"
# Keep the campaign state directory for cross-session dedupe.
# Fresh generation works in that directory; a new directory loses candidate history.
STATE_DIR="${STATE_DIR:-$REPO/output/state/linkedin/9000000005-prre-code}"
PROJECT_ID="9000000005"
LOG_DIR="${STATE_DIR}/launch-logs"

# Model routing.
export STRATEGY_MODEL_NAME=claude-opus-5
export OPUS_MODEL_NAME=claude-opus-5
export CHEAP_MODEL_PROVIDER=anthropic
export CHEAP_MODEL_NAME=claude-haiku-4-5-20251001
# Pin the facial model explicitly; an inherited model setting must not choose the judge.
export FACIAL_MODEL_NAME=claude-opus-5

# Keep the serial full-evaluation path: save and refusal behavior were verified in this mode.
export FULL_EVAL_PIPELINE_ENABLED=false
export LINKEDIN_PAGE_ALLOCATOR_MODE=off
export SHADOW_STRATEGY_ENABLED=false
export SHADOW_FACIAL_MODEL_ENABLED=false

# On 2026-09-20 (plans/feature-flag-audit.md): off, 64% of the "look human"
# pauses land on exactly 24.0 or 42.0 minutes.
export LINKEDIN_CADENCE_READ_FIX_ENABLED=true
# Off, keystrokes carry zero dwell and never overlap — a stronger tell than any
# distribution mismatch.
export LINKEDIN_TYPING_DWELL_ENABLED=true
# Capped retry when a full verdict echoes the wrong candidate id; off, those
# ~0.9% of verdicts are lost outright.
export LINKEDIN_FULL_ID_MISMATCH_RETRY_ENABLED=true
# Deterministic hyphen↔space twins in the Boolean compiler; this changes the
# typed Boolean, so watch result counts on the first session.
export LINKEDIN_SURFACE_VARIANT_EXPANSION_ENABLED=true

# Clear companies/titles at each string start, unexported here before today and
# so left at the config default. On 2026-09-20 over the reconcile fix (CLO-240).
export LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED=true

# Anthropic judgments use the legacy contract with Fireworks judgment policy disabled.
# Changing provider requires revalidating the contract and its pinned settings together.
export LINKEDIN_V2_FACIAL_CONTRACT=legacy
export LINKEDIN_V2_FULL_CONTRACT=legacy
export FIREWORKS_JUDGMENT_POLICY_ENABLED=false
# The Fireworks tool contract requires explicit reasoning effort.
# Its token ceiling, attempt timeout, and total deadline must accommodate the same generation.
export FIREWORKS_FACIAL_REASONING_EFFORT=high
export FIREWORKS_FULL_REASONING_EFFORT=high

# Facial concurrency above one requires the Fireworks tool contract and judgment policy.
# Keep facial judgments serial with this Anthropic configuration.
export LINKEDIN_FACIAL_CONCURRENCY_ENABLED=false
export LINKEDIN_FACIAL_MAX_CONCURRENCY=1

# Publish market intelligence only after complete strategy coverage; a clean session is insufficient.
export MARKET_INTEL_REQUIRE_STRATEGY_COMPLETE=true

# Sleep through closed volume windows without changing caps; detection backoffs still stop the campaign.
# Absorb at most five consecutive errors per fault family, resetting on activity; wait out network loss.
export LINKEDIN_CAMPAIGN_PERSIST=true
export LINKEDIN_SESSION_ERROR_RETRIES=5
# Settle unmatched pending-full candidates with a receipt and continue recovery.
# JUDGMENT_FAILURE remains eligible for deliberate re-judgment.
export LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED=true
# Contain a provider fault by abandoning the facial page with receipts and continuing the session.
# Non-streaming still pays the full wall-clock timeout for that fault.
export LINKEDIN_FACIAL_PAGE_CONTAINMENT_ENABLED=true
# Dormant gaps shape the activity schedule; the 24h open cap still bounds daily volume.
export LINKEDIN_DORMANT_MEDIAN_MINUTES=30
export LINKEDIN_DORMANT_MIN_MINUTES=20
export LINKEDIN_DORMANT_MAX_MINUTES=45

# Enumerate technical artifacts before string formation using one strategy-tier call from the session budget.
# On failure, formation continues without the vocabulary kit.
export LINKEDIN_VOCABULARY_ENUMERATION_ENABLED=true
export LINKEDIN_VOCABULARY_ENUMERATION_RESEARCH_ENABLED=true

# The Fireworks completion budget includes reasoning; size timeouts for the same token ceiling.
export FIREWORKS_PRIMARY_MIN_MAX_TOKENS=32768
export FIREWORKS_FULL_ATTEMPT_TIMEOUT_SECONDS=600
export FIREWORKS_FULL_TOTAL_DEADLINE_SECONDS=1260
export FIREWORKS_FACIAL_ATTEMPT_TIMEOUT_SECONDS=300
export FIREWORKS_FACIAL_TOTAL_DEADLINE_SECONDS=660

# Validate resolved interpreter values: inherited environment settings outrank dotenv.

cd "$REPO"

echo "=== preflight: resolved config, read from inside the interpreter ==="
"$PY" - <<'PYEOF'
import sys
sys.path.insert(0, ".")
from shared import config

EXPECT = {
    "FULL_EVAL_PIPELINE_ENABLED": False,
    "LINKEDIN_PAGE_ALLOCATOR_MODE": "off",
    "SHADOW_STRATEGY_ENABLED": False,
    "SHADOW_FACIAL_MODEL_ENABLED": False,
    "STRATEGY_MODEL_NAME": "claude-opus-5",
    "OPUS_MODEL_NAME": "claude-opus-5",
    "CHEAP_MODEL_PROVIDER": "anthropic",
    "CHEAP_MODEL_NAME": "claude-haiku-4-5-20251001",
    "LINKEDIN_V2_FACIAL_CONTRACT": "legacy",
    "LINKEDIN_V2_FULL_CONTRACT": "legacy",
    "FIREWORKS_JUDGMENT_POLICY_ENABLED": False,
    "FACIAL_MODEL_NAME": "claude-opus-5",
    "LINKEDIN_FACIAL_CONCURRENCY_ENABLED": False,
    "LINKEDIN_FACIAL_MAX_CONCURRENCY": 1,
    "MARKET_INTEL_REQUIRE_STRATEGY_COMPLETE": True,
    "LINKEDIN_VOCABULARY_ENUMERATION_ENABLED": True,
    "LINKEDIN_VOCABULARY_ENUMERATION_RESEARCH_ENABLED": True,
    "LINKEDIN_CAMPAIGN_PERSIST": True,
    "LINKEDIN_SESSION_ERROR_RETRIES": 5,
    # Assert containment explicitly because its config default is disabled.
    "LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED": True,
    "LINKEDIN_FACIAL_PAGE_CONTAINMENT_ENABLED": True,
    "LINKEDIN_DORMANT_MEDIAN_MINUTES": 30.0,
    "LINKEDIN_DORMANT_MIN_MINUTES": 20.0,
    "LINKEDIN_DORMANT_MAX_MINUTES": 45.0,
    # On 2026-09-20; all five default false in shared/config.py, so
    # leaving one out of this dict would be silent.
    "LINKEDIN_CADENCE_READ_FIX_ENABLED": True,
    "LINKEDIN_TYPING_DWELL_ENABLED": True,
    "LINKEDIN_FULL_ID_MISMATCH_RETRY_ENABLED": True,
    "LINKEDIN_SURFACE_VARIANT_EXPANSION_ENABLED": True,
    "LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED": True,
}
bad = []
for key, want in EXPECT.items():
    got = getattr(config, key, "(absent)")
    ok = got == want
    if not ok:
        bad.append((key, got, want))
    print(f"  {'ok ' if ok else 'BAD'} {key:<30} = {got!r}")

# 0 means uncapped, which is intended here; a NEGATIVE value is rejected upstream.
print(f"      LINKEDIN_TOTAL_PAGE_CAP        = {config.LINKEDIN_TOTAL_PAGE_CAP!r} (0 = uncapped)")
print(f"      MAX_PAGES_PER_STRING           = {config.MAX_PAGES_PER_STRING!r}")

if bad:
    print("\nNO-GO — resolved config does not match the run contract:")
    for key, got, want in bad:
        print(f"  {key}: got {got!r}, want {want!r}")
    raise SystemExit(1)
PYEOF

echo
echo "=== preflight: judgment runtime + API keys ==="
# Use the same judgment-runtime validator as startup, including its required API keys.
"$PY" - <<'PYEOF'
import sys
sys.path.insert(0, ".")
from shared import config
from linkedin.orchestrator import Pipeline

for key in ("ANTHROPIC_API_KEY",):
    val = getattr(config, key, None)
    print(f"  {'ok ' if val else 'BAD'} {key:<22} {'present' if val else 'MISSING'}")

# If research is enabled, require its key here; runtime research failures otherwise degrade silently.
if config.LINKEDIN_VOCABULARY_ENUMERATION_RESEARCH_ENABLED:
    if config.PERPLEXITY_API_KEY:
        print(f"  ok  {'PERPLEXITY_API_KEY':<22} present (vocabulary research union)")
    else:
        print(f"  BAD {'PERPLEXITY_API_KEY':<22} MISSING")
        print("\nNO-GO — the run asks for the external-research vocabulary union but")
        print("carries no Perplexity key. Unset")
        print("LINKEDIN_VOCABULARY_ENUMERATION_RESEARCH_ENABLED to run on the")
        print("model's parametric enumeration alone (292 artifacts, 6 of 10 known")
        print("misses), or add the key.")
        raise SystemExit(1)

try:
    Pipeline._validate_judgment_runtime_configuration()
    print("  ok  _validate_judgment_runtime_configuration passed")
except Exception as exc:
    print(f"\nNO-GO — {type(exc).__name__}: {exc}")
    raise SystemExit(1)
PYEOF

echo
echo "=== preflight: brief ==="
"$PY" - <<PYEOF
import sys
sys.path.insert(0, ".")
from shared.brief_loader import load_brief
b = load_brief("$BRIEF")
print(f"  role_title      : {getattr(b, 'role_title', None)!r}")
print(f"  project id      : {b.linkedin_project_id!r}")
jd_len = len(getattr(b, "jd_text", "") or "")
print(f"  jd_text length  : {jd_len}")
print(f"  needs_preflight : {b.needs_preflight()}  (True is REQUIRED here — see below)")
assert str(b.linkedin_project_id) == "$PROJECT_ID", "brief does not pin the authorized project"

# The seed must contain the JD and require preflight generation.
assert jd_len > 0, (
    "brief carries an EMPTY jd_text — this is the exact bug being fixed. "
    "Most likely a relative jd_path resolving against CWD ($REPO) instead of "
    "against the brief file. Inline the JD under the 'jd' key instead."
)
assert b.needs_preflight(), (
    "brief will NOT regenerate — preflight only fires on the fresh-run branch "
    "with needs_preflight() True, so the JD would never reach the eval criteria."
)
PYEOF

echo
echo "=== preflight: browser ==="
if ! curl -s --max-time 3 http://127.0.0.1:9222/json/version >/dev/null; then
  echo "  NO-GO: CDP is down. Start Chrome with --remote-debugging-port=9222 and log into Recruiter."
  exit 1
fi
# Refuse a launch when installed Chrome differs from the running CDP browser; restart Chrome to match.
"$PY" - <<'VCHECK'
import json, plistlib, sys, urllib.request

try:
    version = json.load(
        urllib.request.urlopen("http://127.0.0.1:9222/json/version", timeout=3)
    )
except Exception as exc:
    print(f"  WARNING: /json/version unreadable ({type(exc).__name__}); "
          "skipping the version-match check.")
    raise SystemExit(0)
running = str(version.get("Browser", "")).split("/")[-1].strip()
plist_path = "/Applications/Google Chrome.app/Contents/Info.plist"
try:
    with open(plist_path, "rb") as fh:
        installed = str(
            plistlib.load(fh).get("CFBundleShortVersionString", "")
        ).strip()
except OSError:
    print(f"  WARNING: cannot read {plist_path}; skipping the version-match check.")
    raise SystemExit(0)
if not running or not installed:
    print("  WARNING: version fields incomplete "
          f"(running={running or 'unknown'}, installed={installed or 'unknown'}); "
          "cannot verify the match.")
    raise SystemExit(0)
if running != installed:
    print(f"  NO-GO: the running CDP browser is Chrome {running} but the app on")
    print(f"  disk is {installed}. A Chrome update replaced the app under the")
    print("  running instance — the stale-instance state behind the 'Browser")
    print("  context management is not supported' reconnect deaths (CLO-150).")
    print("  Quit the CDP Chrome, relaunch it with --remote-debugging-port=9222")
    print("  --user-data-dir=$HOME/.chrome-cdp, log into Recruiter, re-run this.")
    sys.exit(1)
print(f"  ok  CDP Chrome {running} matches the installed app")
VCHECK
"$PY" - <<PYEOF
import json, sys, urllib.request
targets = json.load(urllib.request.urlopen("http://127.0.0.1:9222/json", timeout=3))
rec = [t for t in targets if "linkedin.com/talent" in t.get("url", "")]
print(f"  {len(targets)} targets, {len(rec)} Recruiter")
for t in rec:
    print(f"    {t.get('url','')[:100]}")
mine = [t for t in rec if "/talent/hire/$PROJECT_ID/" in t.get("url", "")]
if not mine:
    print("  NO-GO: no Recruiter tab on project $PROJECT_ID.")
    sys.exit(1)
foreign = [t for t in rec if t not in mine]
if foreign:
    print(f"  WARNING: {len(foreign)} Recruiter tab(s) on OTHER projects are open.")
    print("  The run pins the brief's project and refuses a foreign-project save,")
    print("  but close them to avoid burning the run on corrective navigation.")
PYEOF

echo
echo "=== preflight: governor headroom ==="
"$PY" - <<'PYEOF'
import sys
sys.path.insert(0, ".")
from shared.governor import SessionGovernor, MAX_PROFILE_OPENS_PER_24H
from shared import cooldown

g = SessionGovernor()
ok, reason = g.can_start_session(session_type="linkedin_sourcing")
used = cooldown.get_sessions_recorded_today(session_type="linkedin_sourcing")
opens = cooldown.get_profile_opens_24h()
backoff = cooldown.get_active_backoff()

print(f"  can_start_session : {ok} ({reason})")
print(f"  sessions recorded today: {used} (no count gate — CLO-153)")
print(f"  profile opens 24h : {opens}/{MAX_PROFILE_OPENS_PER_24H}")
print(f"  forced backoff    : {backoff or 'none'}")
if not ok:
    sys.exit(1)
PYEOF

echo
echo "=== preflight: no run already live ==="
# Refuse a second sourcing process before it competes for the browser and runtime locks.
# Match the literal module invocation and verify python/caffeinate so editor and review text cannot match.
RUNNING_PID=""
for _pid in $(pgrep -f -- '-m linkedin\.session_orchestrator' 2>/dev/null || true); do
  _comm="$(ps -o comm= -p "$_pid" 2>/dev/null || true)"
  case "${_comm##*/}" in
    python*|caffeinate) RUNNING_PID="$_pid"; break ;;
  esac
done
if [ -n "$RUNNING_PID" ]; then
  echo "  NO-GO: a run is already live (pid $RUNNING_PID)." >&2
  ps -o pid,command= -p "$RUNNING_PID" 2>/dev/null | tail -1 | cut -c1-140 >&2
  echo "  Stop it, or wait for it to finish, before launching another." >&2
  exit 1
fi
echo "  clear"

echo
echo "=== preflight: state dir ==="
mkdir -p "$STATE_DIR" "$LOG_DIR"
echo "  $STATE_DIR"
echo "  durable (not /private/tmp, not inside a git worktree)"
# Reject candidates released by a changed brief hash if their lifecycle cannot return to snippet extraction.
"$PY" - <<PYEOF
import sqlite3, sys
from pathlib import Path
sys.path.insert(0, ".")
from shared.runtime_state.store import (
    ALLOWED_LIFECYCLE_TRANSITIONS,
    DEDUP_BLOCKING_DECISIONS,
    SAVE_FAMILY_DECISIONS,
)

from shared.brief_identity import compute_brief_identity

db = Path("$STATE_DIR") / "runtime_state.sqlite3"
if not db.exists():
    # An absent store is valid for a new campaign, but provides no cross-session dedupe.
    print("  WARNING: no runtime_state.sqlite3 here.")
    print("  Nothing can collide, but nothing dedupes either — every candidate")
    print("  is re-openable against the 400/24h cap.")
    raise SystemExit(0)

# Compute the next run's brief hash from the seed; a previous run hash misses newly released candidates.
_identity = compute_brief_identity("$BRIEF")
brief_hash = (_identity or {}).get("brief_content_hash")

# Fail closed when the canonical store is unreadable. The read-only URI prevents preflight writes.
try:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    prior = conn.execute(
        "SELECT brief_content_hash FROM runs WHERE source='linkedin' AND brief_id=? "
        "ORDER BY id DESC LIMIT 1",
        ("$PROJECT_ID",),
    ).fetchone()
    prior_hash = prior["brief_content_hash"] if prior else None
    current_hash = brief_hash

    rows = conn.execute(
        "SELECT identity_key, display_name, current_lifecycle_state s, terminal_decision d, "
        "brief_content_hash h, user_status u FROM candidates "
        "WHERE source='linkedin' AND brief_id=?",
        ("$PROJECT_ID",),
    ).fetchall()
    # *_started rows with an open attempt reconcile to failed_retryable at run
    # start (RuntimeStateStore.reconcile_open_attempts): count, do not refuse.
    interrupted = {
        r["identity_key"]
        for r in conn.execute(
            "SELECT c.identity_key FROM candidate_attempts ca "
            "JOIN candidates c ON c.id = ca.candidate_id "
            "WHERE ca.status = 'started' AND c.source = 'linkedin' AND c.brief_id = ?",
            ("$PROJECT_ID",),
        ).fetchall()
    }
except sqlite3.Error as exc:
    print(f"\nNO-GO — canonical store present but unreadable: {exc}")
    print("Cannot tell whether this dir holds candidates that would kill the run.")
    print("If the db is mid-write, wait and re-run --check.")
    raise SystemExit(1)

def blocked(r):
    if not (r["s"] == "failed_terminal" or r["d"] in DEDUP_BLOCKING_DECISIONS):
        return False
    if not current_hash:
        return True
    return (
        r["d"] in SAVE_FAMILY_DECISIONS
        or r["u"] is not None
        or r["h"] is None
        or r["h"] == current_hash
    )

detonators = [
    r for r in rows
    if not blocked(r)
    and r["s"] != "snippet_extracted"
    and "snippet_extracted" not in ALLOWED_LIFECYCLE_TRANSITIONS.get(r["s"], set())
    and not (r["s"].endswith("_started") and r["identity_key"] in interrupted)
]
print(f"  candidates {len(rows)} | dedup-blocked {sum(1 for r in rows if blocked(r))}")
if interrupted:
    print(f"  {len(interrupted)} interrupted attempt(s) reconcile to failed_retryable at run start")
print(f"  next run pins brief hash {(current_hash or 'NONE')[:20]}  (from $BRIEF)")
if prior_hash and current_hash and prior_hash != current_hash:
    print(f"  NOTE: last run pinned {prior_hash[:20]} — the brief CHANGED, so")
    print("  prior non-SAVE verdicts are released for re-judging.")
if detonators:
    print(f"\nNO-GO — {len(detonators)} candidate(s) are un-suppressed but sit in a")
    print("lifecycle state with no path back to snippet_extracted. Re-encountering")
    print("any one of them raises and kills the session. Examples:")
    for r in detonators[:5]:
        print(f"    {r['s']:16} {str(r['d']):14} {r['display_name']}")
    raise SystemExit(1)
print("  ok  0 lifecycle detonators under the pinned brief hash")
PYEOF

# Fresh modes discard resumable progress and regenerate in the existing state directory.
# Resume keeps the current run; retaining the directory preserves cross-session dedupe.
case "$MODE" in
  single) MODE_FLAGS=(--single-session --fresh) ;;
  multi)  MODE_FLAGS=(--multi-session --fresh) ;;
  resume) MODE_FLAGS=(--multi-session --resume) ;;
esac
if [ -n "$RESTART_IDS" ]; then
  MODE_FLAGS+=(--restart-strings "$RESTART_IDS")
fi

# Resolve and display launch arguments before --check exits.
echo
echo "=== preflight: launch argv ==="
echo "  mode:  $MODE"
echo "  flags: ${MODE_FLAGS[*]}"
if [ -n "$RESTART_IDS" ]; then
  echo "  RESTART: string(s) ${RESTART_IDS} return to the queue at launch."
  echo "  Any candidate attempts recorded under them are deleted and re-judged."
fi
if [ "$CHECK_ONLY" = "yes" ]; then
  echo
  echo "preflight only — nothing launched."
  echo "  NOTE: the orchestrator's own argparse gate runs at launch and is NOT"
  echo "  exercised here. The flags above are what it will receive."
  exit 0
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="${LOG_DIR}/run-${STAMP}.log"

echo
echo "=== LAUNCH (mode: $MODE) ==="
echo "  flags: ${MODE_FLAGS[*]}"
echo "  log:   $LOG"
echo "  Ctrl-C once = graceful shutdown (finishes current activity)."
if [ "$MODE" = "single" ]; then
  echo
  echo "  SINGLE session: one 4-5h block, then stop. Nothing here pauses at the"
  echo "  first save — the header's stage-inspection check was retired once the"
  echo "  2026-07-27 session landed 24 live saves. If you want to inspect one"
  echo "  anyway, that is a manual Ctrl-C, then:"
  echo "      tools/launch_prre_code.sh resume"
else
  echo
  echo "  MULTI-SESSION: 4-5h of work, then a 75-180 min dormant gap, repeating"
  echo "  until the 24h profile-open cap or exhausted"
  echo "  work stops it. That can span most of a day — keep this terminal open."
  echo "  It will NOT pause for the first-save stage check."
fi
echo

# Keep the terminal attached and the machine awake; return the Python exit status through tee.
caffeinate -dimsu "$PY" -u -m linkedin.session_orchestrator \
  --brief "$BRIEF" \
  --state-dir "$STATE_DIR" \
  "${MODE_FLAGS[@]}" \
  --input-mode concurrent \
  2>&1 | tee "$LOG"
RUN_EXIT="${PIPESTATUS[0]}"

echo
echo "=== run exited: ${RUN_EXIT} ==="
echo "  log: $LOG"
echo "  A ZERO exit is NOT proof the run sourced anything — it also returns 0"
echo "  when there is no pending work and when the governor refuses a session."
echo "  Success = a NEW canonical run id in ${STATE_DIR}/runtime_state.sqlite3."
exit "$RUN_EXIT"
