#!/bin/bash
# Reset GitHub sourcing agent for a fresh run.
# Usage: ./reset-github.sh

echo "Resetting GitHub sourcing agent..."

if [ -f output/github/runtime_state.sqlite3 ]; then
  echo "runtime_state.sqlite3 is present; use tools/runtime_state_admin.py instead of deleting JSON artifacts directly"
  exit 1
fi

# Reset governor daily stats
echo '{"api_calls": [], "enrichments": [], "sessions_today": []}' > ~/.sourcing-governor/github/daily_stats.json
echo "  Governor stats cleared"

# Clear output files
rm -f output/github/*.jsonl output/github/progress.json output/github/saved_candidates.csv
echo "  Output files cleared"

echo "Done. Ready for a fresh run."
