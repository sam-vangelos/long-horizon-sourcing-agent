#!/usr/bin/env python3
"""Regenerate non-PDF intake upload fixtures for browser certification."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "tests" / "fixtures" / "intake"

SYNTHETIC_TEXT = (
    "Certification Intake Role\n\n"
    "Head of Applied AI Lab for banking and financial services. The person owns "
    "production agent systems, model evaluation infrastructure, governance, "
    "executive translation, and a small applied engineering team. Must have "
    "shipped durable systems, not demos, in regulated enterprise settings."
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "cert_intake_jd.txt").write_text(SYNTHETIC_TEXT, encoding="utf-8")
    (OUT_DIR / "cert_intake_jd.md").write_text(
        "# Certification Intake Role\n\n"
        "Head of Applied AI Lab for banking and financial services. The person owns "
        "production agent systems, model evaluation infrastructure, governance, "
        "executive translation, and a small applied engineering team.\n",
        encoding="utf-8",
    )
    from docx import Document

    doc = Document()
    for block in SYNTHETIC_TEXT.split("\n\n"):
        for line in block.split("\n"):
            doc.add_paragraph(line)
    doc.save(OUT_DIR / "cert_intake_jd.docx")
    print(f"wrote fixtures under {OUT_DIR}")


if __name__ == "__main__":
    main()
