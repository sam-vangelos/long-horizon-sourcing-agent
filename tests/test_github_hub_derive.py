"""Tests for :mod:`github.hubs.derive`."""

from __future__ import annotations

from dataclasses import dataclass, field

from github.hubs.derive import RegistryTarget, derive_registry_targets


@dataclass
class _CapabilityArea:
    name: str
    github_code_signals: list[str] = field(default_factory=list)


@dataclass
class _NewBrief:
    capability_areas: list[_CapabilityArea] = field(default_factory=list)


@dataclass
class _Brief:
    target_stacks: list[str] = field(default_factory=list)
    capability_areas: list[_CapabilityArea] = field(default_factory=list)
    _new_brief: _NewBrief | None = None


def test_derives_npm_from_target_stacks_and_code_signals() -> None:
    brief = _Brief(
        target_stacks=["npm", "go"],
        _new_brief=_NewBrief(
            capability_areas=[
                _CapabilityArea(
                    name="Tooling",
                    github_code_signals=[
                        "require('lodash')",
                        "import express from 'express'",
                    ],
                )
            ]
        ),
    )

    targets = derive_registry_targets(brief)

    assert targets == [
        RegistryTarget("npmjs.org", ["lodash", "express"]),
    ]


def test_derives_crates_from_target_stacks_and_code_signals() -> None:
    brief = _Brief(
        target_stacks=["rust"],
        _new_brief=_NewBrief(
            capability_areas=[
                _CapabilityArea(
                    name="Serialization",
                    github_code_signals=[
                        "use serde::Deserialize",
                        "cargo add tokio",
                    ],
                )
            ]
        ),
    )

    targets = derive_registry_targets(brief)

    assert targets == [
        RegistryTarget("crates.io", ["serde", "tokio"]),
    ]


def test_derives_both_ecosystems_when_brief_spans_npm_and_crates() -> None:
    brief = _Brief(
        target_stacks=["npm", "cargo"],
        _new_brief=_NewBrief(
            capability_areas=[
                _CapabilityArea(
                    name="Mixed",
                    github_code_signals=[
                        "'react' in package.json dependencies",
                        "use axum::Router",
                    ],
                )
            ]
        ),
    )

    targets = derive_registry_targets(brief)

    assert targets == [
        RegistryTarget("npmjs.org", ["react"]),
        RegistryTarget("crates.io", ["axum"]),
    ]


def test_code_signals_without_stacks_still_derive_ecosystem() -> None:
    brief = _Brief(
        target_stacks=[],
        _new_brief=_NewBrief(
            capability_areas=[
                _CapabilityArea(
                    name="Node",
                    github_code_signals=["require('zod')"],
                )
            ]
        ),
    )

    targets = derive_registry_targets(brief)

    assert targets == [RegistryTarget("npmjs.org", ["zod"])]


def test_target_stacks_without_code_signals_yield_empty_seed_lists() -> None:
    brief = _Brief(target_stacks=["rust"], _new_brief=_NewBrief())

    targets = derive_registry_targets(brief)

    assert targets == [RegistryTarget("crates.io", [])]


def test_pypi_stack_yields_nothing_for_w2_scope() -> None:
    brief = _Brief(target_stacks=["python", "pypi"], _new_brief=_NewBrief())

    targets = derive_registry_targets(brief)

    assert targets == []


def test_no_targets_for_unrelated_brief() -> None:
    brief = _Brief(
        target_stacks=["go", "kubernetes"],
        _new_brief=_NewBrief(
            capability_areas=[
                _CapabilityArea(
                    name="Infra",
                    github_code_signals=["verl", "trl", "OpenRLHF"],
                )
            ]
        ),
    )

    assert derive_registry_targets(brief) == []


def test_reads_capability_areas_from_dict_brief() -> None:
    brief = {
        "target_stacks": ["npm"],
        "capability_areas": [
            {
                "name": "Node",
                "github_code_signals": ['require("axios")'],
            }
        ],
    }

    targets = derive_registry_targets(brief)

    assert targets == [RegistryTarget("npmjs.org", ["axios"])]
