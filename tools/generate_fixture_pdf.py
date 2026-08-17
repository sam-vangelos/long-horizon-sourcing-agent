#!/usr/bin/env python3
"""Regenerate the sanitized synthetic JD PDF fixture.

Run once when the fixture needs to be refreshed (or after a reportlab
upgrade); the resulting binary at
``tests/fixtures/intake/realistic_jd.pdf`` is committed alongside the
plan so test/cert runs don't need ``reportlab`` at runtime.

The contents are deliberately synthetic. Do **not** swap in real
employer JDs (Acme or otherwise); the fixture's job is to exercise
the upload + synthesis path with a realistic size and shape, not to
reflect any real role.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (
    ROOT / "tests" / "fixtures" / "intake" / "realistic_jd.pdf"
)


SYNTHETIC_JD_TITLE = "Head of Applied AI Lab — Financial Services"


SYNTHETIC_JD_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "About the role",
        (
            "We are hiring a Head of Applied AI Lab to own applied AI strategy",
            "and lab buildout across our regulated financial-services business.",
            "You will lead a team of staff-level builders shipping production",
            "GenAI evaluation, internal copilots, and customer-facing AI",
            "features, with direct exposure to executive stakeholders.",
        ),
    ),
    (
        "What you'll own",
        (
            "Applied AI strategy for the financial-services portfolio.",
            "Lab buildout: hiring, technical direction, and operating model.",
            "Production GenAI evaluation: offline and online.",
            "Executive stakeholder alignment across product, risk, and legal.",
            "Regulated AI delivery: audit trails, model risk, controls.",
        ),
    ),
    (
        "What we're looking for",
        (
            "Someone who has actually built and led applied AI teams in",
            "banking or financial services — not someone who has only",
            "advised on AI strategy from a consulting seat.",
            "Hands-on technical depth in GenAI evaluation, retrieval,",
            "and production LLM systems.",
            "Comfortable leading senior individual contributors and",
            "partnering with executive stakeholders in regulated contexts.",
        ),
    ),
    (
        "Patterns that aren't a fit",
        (
            "Pure research backgrounds without production AI delivery.",
            "Pure advisory backgrounds without team leadership outcomes.",
            "Engineers without regulated-industry exposure (we need someone",
            "who has navigated model risk + audit, not someone learning it).",
        ),
    ),
    (
        "Compensation and location",
        (
            "Competitive base, equity, and a meaningful bonus tied to",
            "delivery outcomes. Hybrid based in New York City; up to two",
            "remote days per week.",
        ),
    ),
)


def render_pdf(output_path: Path) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
        title=SYNTHETIC_JD_TITLE,
        author="Cloris certification synthetic fixture",
        subject="Synthetic JD fixture — sanitized test data, not a real role.",
    )

    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        spaceAfter=6,
    )
    h1_style = ParagraphStyle(
        "h1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        spaceAfter=12,
    )
    h2_style = ParagraphStyle(
        "h2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        spaceBefore=12,
        spaceAfter=6,
    )

    elements = [Paragraph(SYNTHETIC_JD_TITLE, h1_style)]
    for heading, lines in SYNTHETIC_JD_SECTIONS:
        elements.append(Paragraph(heading, h2_style))
        for line in lines:
            elements.append(Paragraph(line, body_style))
        elements.append(Spacer(1, 6))

    doc.build(elements)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PDF path (default: {DEFAULT_OUTPUT.relative_to(ROOT)})",
    )
    args = parser.parse_args()
    render_pdf(args.output)
    size = args.output.stat().st_size
    print(f"wrote {args.output.relative_to(ROOT)} ({size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
