# Config Lifecycle

The `config/` tree is the source of truth for briefs, JDs, and supporting search assets, but not every brief in this tree should be treated as equally runnable.

## Brief lifecycle

- `active`
  - the current runnable briefs
  - filename pattern: `brief-*.json`
  - should not contain `-draft` or `.bak-`
- `draft`
  - active thinking, scratch iterations, or intermediate rewrites
  - use `-draft` in the filename
  - timestamped backups such as `*.bak-YYYYMMDD-HHMMSS.json` also count as draft
- `archived`
  - historical artifacts that should not appear in normal operator surfaces
  - place these under an `archive/`, `archives/`, or `archived/` directory (or mark the filename stem itself with `archived`/`superseded` — either the parent directory name or the filename is checked)
- `superseded`
  - intentionally retired briefs that remain useful for comparison or provenance
  - place these under a `superseded/` directory

## Current launcher behavior

There is no interactive brief picker in the terminal workflow — LinkedIn/GitHub runs launch via an explicit `--brief <path>` or a fixed per-role script under `tools/launch_*.sh`. The desktop-app catalog endpoint (`cloris/api/briefs.py:_scan_authored_briefs`, feeding the parked shell's `/api/briefs`) walks the whole config tree recursively and surfaces every `brief-*.json`/`brief.json`, excluding only `*-draft.json`, `*.bak-*`, and `*-fixture/` directories. Nested role folders are organization, not visibility control.

## Naming rules

- Use `brief-...json` for real brief files.
- Use `-draft` for active scratch versions.
- Use `.bak-...json` for timestamped backups if you absolutely need them.
- Do not rely on a draft or backup file being discoverable from the normal launcher surface.
