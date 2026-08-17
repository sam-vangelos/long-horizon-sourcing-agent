"""Module entrypoint so ``python -m cloris`` delegates to the CLI."""

from __future__ import annotations

from cloris.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
