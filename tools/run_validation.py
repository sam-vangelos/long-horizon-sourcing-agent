#!/usr/bin/env python3
"""Run named validation profiles for the repo."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys


@dataclass(frozen=True)
class ValidationProfile:
    name: str
    description: str
    pytest_args: tuple[str, ...]
    known_excluded: tuple[str, ...] = ()
    known_non_blocking: tuple[str, ...] = ()


PROJECT_ROOT = Path(__file__).resolve().parent.parent

VALIDATION_PROFILES: dict[str, ValidationProfile] = {
    "default": ValidationProfile(
        name="default",
        description="Green default suite for day-to-day engineering work.",
        pytest_args=("-q", "--ignore=tests/test_fde_iteration_dataset.py"),
        known_excluded=("tests/test_fde_iteration_dataset.py",),
    ),
    "full": ValidationProfile(
        name="full",
        description="Full pytest suite, including the dataset replay band.",
        pytest_args=("-q",),
    ),
}


def build_pytest_command(profile_name: str, *, project_root: Path = PROJECT_ROOT) -> list[str]:
    """Build the pytest command for a named validation profile."""
    profile = VALIDATION_PROFILES[profile_name]
    return [sys.executable, "-m", "pytest", *profile.pytest_args]


def run_profile(profile_name: str, *, project_root: Path = PROJECT_ROOT) -> int:
    """Execute a validation profile."""
    profile = VALIDATION_PROFILES[profile_name]
    cmd = build_pytest_command(profile_name, project_root=project_root)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)

    print(f"[validation] profile: {profile.name}")
    print(f"[validation] purpose: {profile.description}")
    if profile.known_excluded:
        print("[validation] known excluded tests:")
        for item in profile.known_excluded:
            print(f"  - {item}")
    if profile.known_non_blocking:
        print("[validation] known non-blocking tests:")
        for item in profile.known_non_blocking:
            print(f"  - {item}")
    else:
        print("[validation] known non-blocking tests: none")
    print(f"[validation] command: {' '.join(cmd)}")

    completed = subprocess.run(cmd, cwd=project_root, env=env)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run named validation profiles")
    parser.add_argument(
        "profile",
        nargs="?",
        choices=sorted(VALIDATION_PROFILES),
        default="default",
        help="Validation profile to run",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available validation profiles and exit",
    )
    args = parser.parse_args()

    if args.list:
        for profile in VALIDATION_PROFILES.values():
            print(f"{profile.name}: {profile.description}")
        return 0

    return run_profile(args.profile)


if __name__ == "__main__":
    raise SystemExit(main())
