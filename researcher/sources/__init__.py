"""Researcher source clients — Slice 2.

Three thin REST wrappers over academic publication sources:

- :mod:`researcher.sources.openalex` — author + works discovery (the spine)
- :mod:`researcher.sources.semantic_scholar` — paper similarity + h-index
  cross-validation
- :mod:`researcher.sources.arxiv` — preprint feed (Atom XML)

Each client:

- Honors its source-specific rate limit (OpenAlex polite pool 10 req/s,
  Semantic Scholar 1 req/s, arXiv 1 req / 3 sec).
- Returns parsed dicts (no orchestration); the acquisition layer in
  Slice 4 composes them.
- Is synchronous; the orchestrator in Slice 6 drives them sequentially.

Per Researcher Module Spec Slice 2: rate-limiting honored at the source
boundary; no async dependency. The existing
:mod:`shared.rate_limiter` is GitHub-specific (auth headers, retry-after
semantics) and would have required generalization that's out of scope
for Slice 2; per-source spacing lives inline as a small spec amendment.
"""

from researcher.sources.arxiv import ArxivClient
from researcher.sources.openalex import OpenAlexClient
from researcher.sources.semantic_scholar import SemanticScholarClient

__all__ = [
    "ArxivClient",
    "OpenAlexClient",
    "SemanticScholarClient",
]
