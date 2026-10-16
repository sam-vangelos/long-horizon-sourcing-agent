# The implementation lives in shared/judgment/templates.py. This shim is LOAD-BEARING:
# linkedin/orchestrator.py's facial-triage borderline resolver imports through it on the
# live run path, alongside tools/ and ~15 test modules. Do not delete until every
# importer of linkedin.judgment_templates is re-pointed at shared.judgment.templates.
import sys

import shared.judgment.templates as _templates

sys.modules[__name__] = _templates
