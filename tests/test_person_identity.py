"""Tests for shared/person_identity.py — evidence-ranked person-key merging."""

import itertools

from shared.person_identity import IdentityEvidence, merge_candidates


def test_registry_owner_github_login_merges_authoritatively() -> None:
    records = [
        IdentityEvidence(
            hub="crates",
            handle="rustacean",
            declared_github_login="octocat",
        ),
        IdentityEvidence(hub="github", handle="octocat"),
    ]

    result = merge_candidates(records)

    assert len(result) == 1
    person = result[0]
    assert person.key == "gh:octocat"
    assert person.github_login == "octocat"
    assert person.handles == {"crates": "rustacean", "github": "octocat"}


def test_shared_commit_email_merges() -> None:
    shared_email = "alice@work.com"
    records = [
        IdentityEvidence(
            hub="npm",
            handle="alice-npm",
            commit_emails=frozenset({shared_email}),
        ),
        IdentityEvidence(
            hub="pypi",
            handle="alice-pypi",
            commit_emails=frozenset({shared_email}),
        ),
    ]

    result = merge_candidates(records)

    assert len(result) == 1
    person = result[0]
    assert person.key == "npm:alice-npm"
    assert person.handles == {"npm": "alice-npm", "pypi": "alice-pypi"}
    assert shared_email in person.commit_emails


def test_display_name_alone_never_merges() -> None:
    records = [
        IdentityEvidence(hub="npm", handle="user-one", display_name="Jane Doe"),
        IdentityEvidence(hub="pypi", handle="user-two", display_name="Jane Doe"),
    ]

    result = merge_candidates(records)

    assert len(result) == 2
    keys = {person.key for person in result}
    assert keys == {"npm:user-one", "pypi:user-two"}


def test_merge_is_transitive_and_stable() -> None:
    shared_email = "alice@example.com"
    records = [
        IdentityEvidence(
            hub="npm",
            handle="pkg-author",
            commit_emails=frozenset({shared_email}),
        ),
        IdentityEvidence(
            hub="github",
            handle="alice-dev",
            commit_emails=frozenset({shared_email}),
        ),
        IdentityEvidence(
            hub="crates",
            handle="crates-user",
            declared_github_login="alice-dev",
        ),
    ]

    first = merge_candidates(records)
    second = merge_candidates(list(reversed(records)))

    assert len(first) == 1
    assert len(second) == 1
    assert first[0].key == "gh:alice-dev"
    assert second[0].key == first[0].key
    assert first[0].handles == {
        "crates": "crates-user",
        "github": "alice-dev",
        "npm": "pkg-author",
    }


def test_noreply_email_never_merges() -> None:
    noreply_email = "12345+alice@users.noreply.github.com"
    records = [
        IdentityEvidence(
            hub="github",
            handle="gh-user",
            commit_emails=frozenset({noreply_email}),
        ),
        IdentityEvidence(
            hub="npm",
            handle="npm-user",
            commit_emails=frozenset({noreply_email}),
        ),
    ]

    result = merge_candidates(records)

    assert len(result) == 2
    keys = {person.key for person in result}
    assert keys == {"gh:gh-user", "npm:npm-user"}


def test_conflicting_declared_logins_get_distinct_keys() -> None:
    records = [
        IdentityEvidence(
            hub="npm",
            handle="npm-user-a",
            declared_github_login="gh1",
        ),
        IdentityEvidence(
            hub="npm",
            handle="npm-user-b",
            declared_github_login="gh1",
        ),
    ]

    result = merge_candidates(records)

    assert len(result) == 2
    keys = {person.key for person in result}
    assert keys == {"npm:npm-user-a", "npm:npm-user-b"}
    assert "gh:gh1" not in keys


def test_same_hub_same_handle_merges() -> None:
    records = [
        IdentityEvidence(hub="github", handle="octocat"),
        IdentityEvidence(hub="github", handle="Octocat"),
    ]

    result = merge_candidates(records)

    assert len(result) == 1
    person = result[0]
    assert person.key == "gh:octocat"
    assert person.handles == {"github": "octocat"}


def _normalized_grouping(result: list) -> list[tuple[str, dict[str, str]]]:
    return sorted((person.key, dict(person.handles)) for person in result)


def test_grouping_is_input_order_independent() -> None:
    shared_email = "shared@example.com"
    records = [
        IdentityEvidence(hub="npm", handle="alpha", declared_github_login="dev1"),
        IdentityEvidence(hub="npm", handle="beta", declared_github_login="dev1"),
        IdentityEvidence(
            hub="github",
            handle="dev1",
            commit_emails=frozenset({shared_email}),
        ),
        IdentityEvidence(
            hub="pypi",
            handle="gamma",
            commit_emails=frozenset({shared_email}),
        ),
    ]

    baseline = _normalized_grouping(merge_candidates(records))

    for permutation in itertools.permutations(records):
        assert _normalized_grouping(merge_candidates(permutation)) == baseline


def test_output_keys_are_unique() -> None:
    scenarios = [
        [
            IdentityEvidence(
                hub="npm",
                handle="npm-user-a",
                declared_github_login="gh1",
            ),
            IdentityEvidence(
                hub="npm",
                handle="npm-user-b",
                declared_github_login="gh1",
            ),
        ],
        [
            IdentityEvidence(hub="github", handle="octocat"),
            IdentityEvidence(hub="github", handle="Octocat"),
        ],
        [
            IdentityEvidence(hub="npm", handle="alpha", declared_github_login="dev1"),
            IdentityEvidence(hub="npm", handle="beta", declared_github_login="dev1"),
            IdentityEvidence(hub="github", handle="dev1"),
            IdentityEvidence(
                hub="pypi",
                handle="gamma",
                commit_emails=frozenset({"shared@example.com"}),
            ),
        ],
        [
            IdentityEvidence(hub="npm", handle="user-one", display_name="Jane Doe"),
            IdentityEvidence(hub="pypi", handle="user-two", display_name="Jane Doe"),
        ],
    ]

    for records in scenarios:
        result = merge_candidates(records)
        keys = [person.key for person in result]
        assert len(keys) == len(set(keys)), f"duplicate keys in output: {keys}"
