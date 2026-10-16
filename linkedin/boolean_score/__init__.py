"""Boolean quality scoring: a composite in [0, 1] for one LinkedIn Recruiter
Boolean string, read against a role lexicon supplied as JSON data.

Standard library only, and no imports from anywhere else in this repo, so the
directory can be vendored by ``cp -r`` into another project. See
plans/boolean-scorer.md for the spec.
"""

from __future__ import annotations

from .lexicon import (
    LEXICON_DIR,
    Concept,
    Form,
    Lexicon,
    LexiconError,
    build_lexicon,
    load_lexicon,
)
from .parser import And, Finding, Not, Or, ParseError, ParseResult, Term, parse

__all__ = [
    "And",
    "Concept",
    "Finding",
    "Form",
    "LEXICON_DIR",
    "Lexicon",
    "LexiconError",
    "Not",
    "Or",
    "ParseError",
    "ParseResult",
    "Term",
    "build_lexicon",
    "load_lexicon",
    "parse",
]
