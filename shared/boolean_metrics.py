"""Single owner of boolean structural analysis (skeletonization)."""

import re

_QUOTED = re.compile(r'"([^"]*)"')
_TOKEN = re.compile(r"\(|\)|[^\s()]+")
_OPERATORS = {"AND", "OR", "NOT"}
# A parenthesized OR-alternation of >=2 already-collapsed items. A lone
# "(T)" deliberately does NOT collapse: "(term) AND (group)" and
# "(group) AND (group)" are different shapes, and the skeleton exists to
# make shape visible.
_OR_GROUP = re.compile(r"\((?:T|\(G\))(?: OR (?:T|\(G\)))+\)")
_T_RUN = re.compile(r"\bT(?: T)+\b")


def _skeletonize(boolean: str) -> str:
    """Normalize one boolean to its structural skeleton.

    Quoted phrases and bare terms both become ``T`` (a run of adjacent bare
    tokens collapses to one ``T`` — an unquoted multi-word term is still one
    term), operators uppercase, then ``(T OR T ...)`` groups collapse to
    ``(G)`` iteratively so nested OR-trees of any depth read as one group:
    ``("a" OR b) AND (c OR "d e") NOT f`` -> ``(G) AND (G) NOT T``.
    """
    tokens = _TOKEN.findall(_QUOTED.sub("T", boolean))
    parts: list[str] = []
    for tok in tokens:
        if tok in ("(", ")"):
            parts.append(tok)
        elif tok.upper() in _OPERATORS:
            parts.append(tok.upper())
        else:
            parts.append("T")
    skeleton = " ".join(parts).replace("( ", "(").replace(" )", ")")
    skeleton = _T_RUN.sub("T", skeleton)
    prev = None
    while prev != skeleton:
        prev = skeleton
        skeleton = _OR_GROUP.sub("(G)", skeleton)
    return skeleton


skeletonize = _skeletonize
