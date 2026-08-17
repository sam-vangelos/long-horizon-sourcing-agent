# moved to shared/judgment/tool_contracts.py per spec §A2 move 1. This shim is LOAD-BEARING:
# linkedin/orchestrator.py's facial-triage borderline resolver imports through it on the
# live run path, alongside tools/ and ~15 test modules. Do not delete until every
# importer of linkedin.judgment_tool_contracts is re-pointed at shared.judgment.tool_contracts.
import sys

import shared.judgment.tool_contracts as _tool_contracts

sys.modules[__name__] = _tool_contracts
