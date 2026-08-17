# Validation Standard

This repo has one default validation profile for day-to-day work and one full profile for exhaustive replay coverage.

## Default green suite

Use this for normal engineering work:

```bash
python3 tools/run_validation.py default
```

Equivalent Make target:

```bash
make validate
```

This runs hygiene plus this profile — see Hygiene below.

The default profile currently means:

- run `pytest -q`
- exclude `tests/test_fde_iteration_dataset.py`

That dataset replay band is intentionally excluded from the default profile because it is heavier and not required for most feature work.

Note: `tests/test_fde_iteration_dataset.py` is not present in this checkout, so the `default`/`full` split is currently inert — both profiles collect the same tests.

## Full suite

Use this when you explicitly want the full pytest surface, including dataset replay coverage:

```bash
python3 tools/run_validation.py full
```

Equivalent Make target:

```bash
make test-full
```

## Non-blocking drift tests

There are currently **no** named non-blocking calibration-drift tests in the default profile.

If a future test is intentionally allowed to drift without blocking normal engineering work, it should be documented here before it is excluded from the default profile.

## Hygiene

For a quick repo hygiene pass:

```bash
python3 tools/check_repo_hygiene.py
```

That check is intentionally lightweight. It currently verifies:

- tracked incidental junk such as `.DS_Store`
- tracked `__pycache__` artifacts
- tracked timestamped backup JSON files
- prints a non-blocking brief-lifecycle inventory summary from `config/` (informational only)
- fails any tracked doc missing a provenance stamp (era-alignment B4, 2026-08-02 — see docs/INDEX.md)

## Rule of thumb

- Structural regressions should block the default suite.
- Heavy replay bands can be excluded from the default suite if they remain available in the full suite.
- Calibration drift should only become non-blocking when it is called out explicitly and intentionally.
