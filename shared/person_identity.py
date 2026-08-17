"""Evidence-ranked person-key model for multi-hub OSS identity.

Pure data structures and merge logic — no I/O, no runtime-state wiring.
Merge rules are conservative: registry owner records naming a GitHub login are
authoritative; same hub + same handle (case-insensitive) is authoritative;
shared verified commit emails are strong; matching display names alone never
merge.

Known limitation (accepted for W1): a person's key changes when GitHub evidence
arrives for a previously registry-only person (e.g. ``npm:zeta`` becomes
``gh:zed`` once a corroborating ``github``-hub record joins the group).
Cross-wave re-keying policy is deferred to W2.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IdentityEvidence:
    """One hub's view of a person."""

    hub: str
    handle: str
    display_name: str = ""
    commit_emails: frozenset[str] = frozenset()
    declared_github_login: str = ""
    linkedin_url: str = ""


@dataclass(frozen=True, slots=True)
class PersonKey:
    """Stable internal identifier joining evidence across hubs."""

    key: str
    github_login: str
    handles: dict[str, str]
    commit_emails: frozenset[str]
    display_name: str
    linkedin_url: str


def merge_candidates(records: Iterable[IdentityEvidence]) -> list[PersonKey]:
    """Group evidence into person keys via ranked merge rules.

    Merge evidence (transitive closure via union-find):

    * **Authoritative:** same ``hub`` + same ``handle`` (case-insensitive), or
      ``declared_github_login`` on one record matches the ``handle`` of a
      ``github``-hub record (case-insensitive), or two records declare the same
      non-empty GitHub login.
    * **Strong:** a shared commit email that is non-empty and not a noreply
      address (``users.noreply.github.com`` or any local-part/domain containing
      ``noreply``, case-insensitive).
    * Matching ``display_name`` alone never merges.

    Default: do not merge.

    When unioning would assign two different handles to the same hub, the merge
    is rejected — a handle conflict is evidence *against* merging, and the
    conservative default keeps those people separate. Merged ``handles`` unions
    must not lose entries.

    Input records are sorted canonically before grouping so output depends only
    on the evidence multiset, not arrival order. A ``gh:`` key is minted only
    when the group contains a ``github``-hub record; uncorroborated
    ``declared_github_login`` alone keys off the lexicographically-first
    ``<hub>:<handle>`` pair.

    Output is sorted deterministically by ``key``.
    """
    indexed = sorted(records, key=_canonical_sort_key)
    if not indexed:
        return []

    uf = _UnionFind(len(indexed), indexed)

    for left in range(len(indexed)):
        for right in range(left + 1, len(indexed)):
            if _merge_evidence(indexed[left], indexed[right]):
                uf.try_union(left, right)

    groups: dict[int, list[IdentityEvidence]] = {}
    for idx, record in enumerate(indexed):
        root = uf.find(idx)
        groups.setdefault(root, []).append(record)

    keys = [_person_key_from_group(group) for group in groups.values()]
    key_strings = [person.key for person in keys]
    if len(key_strings) != len(set(key_strings)):
        duplicates = sorted(
            key for key in set(key_strings) if key_strings.count(key) > 1
        )
        raise AssertionError(f"duplicate person keys in merge output: {duplicates}")
    keys.sort(key=lambda person: person.key)
    return keys


def _person_key_from_group(records: list[IdentityEvidence]) -> PersonKey:
    handles: dict[str, str] = {}
    emails: set[str] = set()
    display_name = ""
    linkedin_url = ""
    github_login = ""

    for record in sorted(records, key=lambda item: (item.hub, item.handle)):
        handles[record.hub] = record.handle
        emails.update(record.commit_emails)
        if not display_name and record.display_name:
            display_name = record.display_name
        if not linkedin_url and record.linkedin_url:
            linkedin_url = record.linkedin_url

    if "github" in handles:
        github_login = handles["github"]
    else:
        for record in sorted(records, key=lambda item: (item.hub, item.handle)):
            if record.declared_github_login:
                github_login = record.declared_github_login
                break

    key = _derive_key(handles)
    return PersonKey(
        key=key,
        github_login=github_login,
        handles=handles,
        commit_emails=frozenset(emails),
        display_name=display_name,
        linkedin_url=linkedin_url,
    )


def _derive_key(handles: dict[str, str]) -> str:
    if "github" in handles:
        return f"gh:{handles['github'].lower()}"
    if handles:
        first_hub = min(handles, key=str.lower)
        return f"{first_hub.lower()}:{handles[first_hub]}"
    return "unknown:"


def _canonical_sort_key(record: IdentityEvidence) -> tuple[str, str, str, tuple[str, ...]]:
    return (
        record.hub.lower(),
        record.handle.lower(),
        record.declared_github_login.lower(),
        tuple(sorted(record.commit_emails)),
    )


def _merge_evidence(left: IdentityEvidence, right: IdentityEvidence) -> bool:
    if _authoritative_merge(left, right):
        return True
    return _shared_verified_email(left, right)


def _authoritative_merge(left: IdentityEvidence, right: IdentityEvidence) -> bool:
    if _same_hub_same_handle(left, right):
        return True
    if _declared_login_matches_github_handle(left, right):
        return True
    if _declared_login_matches_github_handle(right, left):
        return True
    left_login = left.declared_github_login.strip()
    right_login = right.declared_github_login.strip()
    if left_login and right_login and left_login.lower() == right_login.lower():
        return True
    return False


def _same_hub_same_handle(left: IdentityEvidence, right: IdentityEvidence) -> bool:
    return (
        left.hub.lower() == right.hub.lower()
        and left.handle.lower() == right.handle.lower()
    )


def _declared_login_matches_github_handle(
    source: IdentityEvidence, github_record: IdentityEvidence
) -> bool:
    declared = source.declared_github_login.strip()
    if not declared:
        return False
    if github_record.hub.lower() != "github":
        return False
    return declared.lower() == github_record.handle.lower()


def _shared_verified_email(left: IdentityEvidence, right: IdentityEvidence) -> bool:
    left_emails = {_normalize_email(email) for email in left.commit_emails if _is_verified_email(email)}
    right_emails = {
        _normalize_email(email) for email in right.commit_emails if _is_verified_email(email)
    }
    return bool(left_emails & right_emails)


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _is_verified_email(email: str) -> bool:
    normalized = _normalize_email(email)
    if not normalized or "@" not in normalized:
        return False
    local_part, domain = normalized.split("@", 1)
    if "noreply" in local_part or "noreply" in domain:
        return False
    if domain == "users.noreply.github.com":
        return False
    return True


class _UnionFind:
    def __init__(self, size: int, records: list[IdentityEvidence]) -> None:
        self._parent = list(range(size))
        self._handles = [{record.hub: record.handle} for record in records]

    def find(self, node: int) -> int:
        if self._parent[node] != node:
            self._parent[node] = self.find(self._parent[node])
        return self._parent[node]

    def try_union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if not self._compatible(left_root, right_root):
            return
        self._parent[right_root] = left_root
        self._handles[left_root].update(self._handles[right_root])

    def _compatible(self, left_root: int, right_root: int) -> bool:
        for hub, handle in self._handles[right_root].items():
            existing = self._handles[left_root].get(hub)
            if existing is not None and existing.lower() != handle.lower():
                return False
        return True
