"""Tokenizer and recursive-descent parser for the LinkedIn Recruiter Boolean
dialect (plans/boolean-scorer.md §4).

Dialect: double-quoted phrases, bare tokens, uppercase ``AND`` / ``OR`` /
``NOT``, parentheses. Precedence is NOT > AND > OR; parentheses override.
Adjacent operands are joined by an implicit AND.

The module reimplements, on a real parse tree, the legality semantics the
repo's existing linter checks with regexes over flat groups. It shares no code
with that linter and imports nothing outside the standard library, so the
package can be vendored by copy.

Legality errors (any one makes the parse illegal, which zeroes the composite):
``empty_input``, ``unbalanced_parenthesis``, ``unclosed_quote``,
``malformed_operator`` (``&&``, ``||``, a bare ``&``), ``wildcard_operator``
(``*`` outside quotes), ``empty_group``, ``dangling_operator``,
``not_group_contains_and``.

Warnings, which never make a parse illegal: ``lowercase_operator`` and
``unquoted_multiword_run`` (one per maximal run of adjacent bare tokens).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Union

# Wire-format token kinds, not vocabulary.
_TERM_KIND = "term"
_OPERATOR_KIND = "operator"
_OPEN_KIND = "open"
_CLOSE_KIND = "close"

_AND_KEY = "AND"
_OR_KEY = "OR"
_NOT_KEY = "NOT"
_OPERATOR_KEYS = frozenset({_AND_KEY, _OR_KEY, _NOT_KEY})
_LOWERCASE_OPERATOR_KEYS = frozenset(key.lower() for key in _OPERATOR_KEYS)

_DOUBLE_OPERATOR_RE = re.compile(r"&&|\|\|")
_BARE_AMPERSAND_RE = re.compile(r"(?<![A-Za-z_])&(?![A-Za-z_])")
_WILDCARD_RE = re.compile(r"\*")

_WARNING = "warning"
_ERROR = "error"


# ---------------------------------------------------------------------------
# Tree
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Term:
    """One search term. ``text`` is as authored, whitespace collapsed."""

    text: str
    quoted: bool
    span: tuple[int, int] = field(default=(-1, -1), compare=False)

    @property
    def normalized(self) -> str:
        return " ".join(self.text.lower().split())


@dataclass(frozen=True)
class And:
    children: tuple["Node", ...]


@dataclass(frozen=True)
class Or:
    children: tuple["Node", ...]


@dataclass(frozen=True)
class Not:
    child: "Node"


Node = Union[Term, And, Or, Not]


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParseError:
    code: str
    message: str
    span: tuple[int, int] | None = None

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "span": list(self.span) if self.span else None}


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str
    message: str
    span: tuple[int, int] | None = None

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "span": list(self.span) if self.span else None,
        }


@dataclass(frozen=True)
class ParseResult:
    source: str
    tree: Node | None
    errors: tuple[ParseError, ...]
    findings: tuple[Finding, ...]
    max_depth: int = 0

    @property
    def legal(self) -> bool:
        return not self.errors


# ---------------------------------------------------------------------------
# Lexing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str
    span: tuple[int, int]
    quoted: bool = False


def mask_quoted_spans(boolean: str) -> str:
    """``boolean`` with quoted interiors blanked, offsets and length preserved.

    A mirror of the existing linter's masking helper: regexes run against the
    masked string map back to the same offsets in the original, and quoted
    content can never match an operator pattern.
    """
    chars = list(boolean)
    inside = False
    for index, char in enumerate(boolean):
        if char == '"' and (index == 0 or boolean[index - 1] != "\\"):
            inside = not inside
            continue
        if inside:
            chars[index] = " "
    return "".join(chars)


def _collapse(value: str) -> str:
    return " ".join(value.split())


def _tokenize(boolean: str) -> tuple[list[_Token], list[Finding]]:
    tokens: list[_Token] = []
    findings: list[Finding] = []
    index = 0
    length = len(boolean)
    while index < length:
        char = boolean[index]
        if char.isspace():
            index += 1
            continue
        if char == "(":
            tokens.append(_Token(_OPEN_KIND, char, (index, index + 1)))
            index += 1
            continue
        if char == ")":
            tokens.append(_Token(_CLOSE_KIND, char, (index, index + 1)))
            index += 1
            continue
        if char == '"':
            end = index + 1
            while end < length:
                if boolean[end] == '"' and boolean[end - 1] != "\\":
                    break
                end += 1
            inner = _collapse(boolean[index + 1 : end])
            if inner:
                tokens.append(_Token(_TERM_KIND, inner, (index, min(end + 1, length)), True))
            index = end + 1
            continue
        end = index
        while end < length and not boolean[end].isspace() and boolean[end] not in '()"':
            end += 1
        word = boolean[index:end]
        span = (index, end)
        if word in _OPERATOR_KEYS:
            tokens.append(_Token(_OPERATOR_KIND, word, span))
        else:
            if word.lower() in _LOWERCASE_OPERATOR_KEYS:
                findings.append(
                    Finding(
                        "lowercase_operator",
                        _WARNING,
                        f"{word!r} is not an operator here; only uppercase operators bind.",
                        span,
                    )
                )
            tokens.append(_Token(_TERM_KIND, word, span))
        index = end
    return tokens, findings


def _bare_run_findings(tokens: list[_Token]) -> list[Finding]:
    findings: list[Finding] = []
    run: list[_Token] = []

    def flush() -> None:
        if len(run) >= 2:
            findings.append(
                Finding(
                    "unquoted_multiword_run",
                    _WARNING,
                    "Adjacent bare tokens search as separate terms, not as a phrase.",
                    (run[0].span[0], run[-1].span[1]),
                )
            )
        run.clear()

    for token in tokens:
        if token.kind == _TERM_KIND and not token.quoted:
            run.append(token)
        else:
            flush()
    flush()
    return findings


# ---------------------------------------------------------------------------
# Scan-phase legality (offsets are reliable before any tree exists)
# ---------------------------------------------------------------------------


def _scan_errors(boolean: str) -> tuple[list[ParseError], str]:
    if not boolean.strip():
        return [ParseError("empty_input", "Empty Boolean string.", (0, len(boolean)))], ""

    inside = False
    for index, char in enumerate(boolean):
        if char == '"' and (index == 0 or boolean[index - 1] != "\\"):
            inside = not inside
    if inside:
        return [ParseError("unclosed_quote", "Unclosed double quote.", None)], ""

    masked = mask_quoted_spans(boolean)
    errors: list[ParseError] = []

    depth = 0
    for index, char in enumerate(masked):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                errors.append(
                    ParseError(
                        "unbalanced_parenthesis",
                        "Unmatched closing parenthesis.",
                        (index, index + 1),
                    )
                )
                depth = 0
    if depth > 0:
        errors.append(
            ParseError("unbalanced_parenthesis", "Unclosed parenthesis.", None)
        )

    for match in _DOUBLE_OPERATOR_RE.finditer(masked):
        errors.append(
            ParseError(
                "malformed_operator",
                f"{match.group(0)!r} is not valid syntax here.",
                (match.start(), match.end()),
            )
        )
    for match in _BARE_AMPERSAND_RE.finditer(masked):
        errors.append(
            ParseError(
                "malformed_operator",
                "A bare '&' is not valid syntax here.",
                (match.start(), match.end()),
            )
        )
    for match in _WILDCARD_RE.finditer(masked):
        errors.append(
            ParseError(
                "wildcard_operator",
                "'*' outside a quoted phrase is not supported.",
                (match.start(), match.end()),
            )
        )
    return errors, masked


def _max_depth(masked: str) -> int:
    depth = 0
    deepest = 0
    for char in masked:
        if char == "(":
            depth += 1
            deepest = max(deepest, depth)
        elif char == ")":
            depth = max(0, depth - 1)
    return deepest


# ---------------------------------------------------------------------------
# Recursive descent
# ---------------------------------------------------------------------------


class _Abort(Exception):
    def __init__(self, error: ParseError) -> None:
        super().__init__(error.message)
        self.error = error


def contains_and(node: Node | None) -> bool:
    """True when an ``And`` sits anywhere in this subtree."""
    if isinstance(node, And):
        return True
    if isinstance(node, Or):
        return any(contains_and(child) for child in node.children)
    if isinstance(node, Not):
        return contains_and(node.child)
    return False


class _Parser:
    def __init__(self, tokens: list[_Token], source_length: int) -> None:
        self._tokens = tokens
        self._pos = 0
        self._end = source_length

    # -- cursor
    def _peek(self) -> _Token | None:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _advance(self) -> _Token:
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    def _tail_span(self) -> tuple[int, int]:
        if self._pos > 0:
            previous = self._tokens[self._pos - 1]
            return previous.span
        return (0, self._end)

    # -- grammar
    def parse(self) -> Node:
        node = self._parse_or()
        leftover = self._peek()
        if leftover is not None:
            raise _Abort(
                ParseError(
                    "dangling_operator",
                    "Trailing input after the expression.",
                    leftover.span,
                )
            )
        return node

    def _parse_or(self) -> Node:
        children = [self._parse_and()]
        while True:
            token = self._peek()
            if token is not None and token.kind == _OPERATOR_KIND and token.value == _OR_KEY:
                self._advance()
                children.append(self._parse_and())
                continue
            break
        return children[0] if len(children) == 1 else Or(tuple(children))

    def _parse_and(self) -> Node:
        children = [self._parse_not()]
        while True:
            token = self._peek()
            if token is None:
                break
            if token.kind == _OPERATOR_KIND and token.value == _AND_KEY:
                self._advance()
                children.append(self._parse_not())
                continue
            if token.kind in (_TERM_KIND, _OPEN_KIND) or (
                token.kind == _OPERATOR_KIND and token.value == _NOT_KEY
            ):
                children.append(self._parse_not())
                continue
            break
        return children[0] if len(children) == 1 else And(tuple(children))

    def _parse_not(self) -> Node:
        token = self._peek()
        if token is not None and token.kind == _OPERATOR_KIND and token.value == _NOT_KEY:
            self._advance()
            child = self._parse_not()
            if contains_and(child):
                raise _Abort(
                    ParseError(
                        "not_group_contains_and",
                        "NOT applies to a group holding an AND; it must take a "
                        "single term or a pure OR group.",
                        (token.span[0], self._tail_span()[1]),
                    )
                )
            return Not(child)
        return self._parse_primary()

    def _parse_primary(self) -> Node:
        token = self._peek()
        if token is None:
            raise _Abort(
                ParseError("dangling_operator", "Operator with no operand.", self._tail_span())
            )
        if token.kind == _TERM_KIND:
            self._advance()
            return Term(token.value, token.quoted, token.span)
        if token.kind == _OPEN_KIND:
            if not self._group_holds_a_term():
                raise _Abort(
                    ParseError(
                        "empty_group",
                        "Parenthetical group holds no search term.",
                        (token.span[0], self._group_close_offset()),
                    )
                )
            self._advance()
            node = self._parse_or()
            closing = self._peek()
            if closing is None or closing.kind != _CLOSE_KIND:
                raise _Abort(
                    ParseError(
                        "dangling_operator",
                        "Group ends without an operand.",
                        self._tail_span(),
                    )
                )
            self._advance()
            return node
        raise _Abort(
            ParseError("dangling_operator", "Operand expected here.", token.span)
        )

    # -- lookahead helpers
    def _group_holds_a_term(self) -> bool:
        depth = 0
        for token in self._tokens[self._pos :]:
            if token.kind == _OPEN_KIND:
                depth += 1
            elif token.kind == _CLOSE_KIND:
                depth -= 1
                if depth == 0:
                    return False
            elif token.kind == _TERM_KIND:
                return True
        return False

    def _group_close_offset(self) -> int:
        depth = 0
        for token in self._tokens[self._pos :]:
            if token.kind == _OPEN_KIND:
                depth += 1
            elif token.kind == _CLOSE_KIND:
                depth -= 1
                if depth == 0:
                    return token.span[1]
        return self._end


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse(boolean: str) -> ParseResult:
    """Parse one Boolean string into a tree plus its legality findings."""
    source = boolean if isinstance(boolean, str) else ""
    errors, masked = _scan_errors(source)
    if errors:
        return ParseResult(source, None, tuple(errors), (), _max_depth(masked))

    tokens, findings = _tokenize(source)
    findings.extend(_bare_run_findings(tokens))
    if not tokens:
        return ParseResult(
            source,
            None,
            (ParseError("empty_input", "No search terms.", (0, len(source))),),
            (),
            0,
        )
    try:
        tree = _Parser(tokens, len(source)).parse()
    except _Abort as abort:
        return ParseResult(source, None, (abort.error,), tuple(findings), _max_depth(masked))
    return ParseResult(source, tree, (), tuple(findings), _max_depth(masked))
