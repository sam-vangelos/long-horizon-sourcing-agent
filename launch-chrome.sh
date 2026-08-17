#!/bin/bash
# Launch Chrome with CDP debugging enabled — dev compatibility shim.
#
# As of Phase 0 ``chrome-launcher`` slice, the source-of-truth logic
# lives in cloris/chrome_launcher.py so the trial-day .app can spawn
# Chrome itself (the recipient cannot reasonably open a Terminal).
#
# This shell script remains for dev muscle memory. It delegates to the
# Python module, which:
#
#   - Uses the Cloris-namespaced profile under
#     ~/Library/Application Support/Cloris/chrome-profile (frozen .app)
#     OR the historical ~/.chrome-cdp (dev) so existing dev sessions
#     keep working without re-login.
#   - NEVER pkill -9's "Google Chrome" globally — it only terminates
#     processes whose --user-data-dir matches the dedicated Cloris
#     profile. The recipient's personal Chrome is untouched.
#
# Flags:
#   --force    Recycle Chrome even if CDP is already healthy.
#   --status   Emit the current ChromeStatus as JSON and exit.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON:-python3}"
if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
fi

exec "$PYTHON_BIN" -m cloris.chrome_launcher "$@"
