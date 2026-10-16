from __future__ import annotations

from io import BytesIO

import pytest

from shared.source_packet import (
    SourcePacketError,
    compose_source_packet_text,
    extract_source_file_text,
    normalize_source_text,
)
from shared.source_packet_synthesis import (
    SYNTHESIS_SOURCE_CHAR_BUDGET,
    synthesize_v2_from_source_packet,
)
from shared.brief_v2_schema import validate_v2_brief


def _pdf_with_text(text: str) -> bytes:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 24 Tf 72 720 Td ({escaped}) Tj ET\n".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{idx} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"
    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode("ascii")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode("ascii")
    return bytes(out)


def _docx_with_text(text: str) -> bytes:
    from docx import Document

    doc = Document()
    for paragraph in text.split("\n"):
        doc.add_paragraph(paragraph)
    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_text_upload_extracts_readable_text() -> None:
    out = extract_source_file_text(
        filename="intake.txt",
        content=b"Staff Platform Engineer\n\nBuild internal developer systems.",
        content_type="text/plain",
        kind="intake_notes",
    )
    assert out.filename == "intake.txt"
    assert "developer systems" in out.text
    assert out.kind == "intake_notes"


def test_markdown_upload_extracts_readable_text() -> None:
    out = extract_source_file_text(
        filename="intake-notes.md",
        content=(
            b"# Head of Applied AI Lab\n\n"
            b"- Quasi-CTO for BFS applied AI.\n"
            b"- Needs agentic systems depth."
        ),
        content_type="text/markdown",
        kind="intake_notes",
    )

    assert out.filename == "intake-notes.md"
    assert out.kind == "intake_notes"
    assert "Quasi-CTO for BFS applied AI" in out.text


def test_pdf_upload_extracts_readable_text() -> None:
    out = extract_source_file_text(
        filename="head-of-applied-ai.pdf",
        content=_pdf_with_text("Head of Applied AI Lab"),
        content_type="application/pdf",
        kind="job_description",
    )

    assert out.kind == "job_description"
    assert "Head of Applied AI Lab" in out.text


def test_docx_upload_extracts_readable_text() -> None:
    out = extract_source_file_text(
        filename="intake-notes.docx",
        content=_docx_with_text("Hiring manager wants a builder, not an advisor."),
        content_type=(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
        kind="intake_notes",
    )

    assert out.kind == "intake_notes"
    assert "builder, not an advisor" in out.text


def test_compose_source_packet_groups_uploaded_files_by_kind() -> None:
    jd = extract_source_file_text(
        filename="jd.txt",
        content=b"Staff Platform Engineer\n\nOwn developer systems.",
        content_type="text/plain",
        kind="job_description",
    )
    notes = extract_source_file_text(
        filename="notes.txt",
        content=b"Needs hands-on infrastructure judgment.",
        content_type="text/plain",
        kind="intake_notes",
    )
    legacy = extract_source_file_text(
        filename="legacy.txt",
        content=b"Legacy generic upload remains readable.",
        content_type="text/plain",
    )

    text = compose_source_packet_text(files=[jd, notes, legacy])

    assert "UPLOADED JOB DESCRIPTION FILES" in text
    assert "UPLOADED INTAKE NOTES FILES" in text
    assert "UPLOADED GENERAL FILES" in text
    assert "Legacy generic upload remains readable." in text


def test_unsupported_upload_extension_raises() -> None:
    with pytest.raises(SourcePacketError) as exc:
        extract_source_file_text(
            filename="spreadsheet.xlsx",
            content=b"x",
            content_type=None,
        )
    assert exc.value.code == "unsupported_source_file"


def test_source_packet_synthesis_fallback_returns_valid_v2() -> None:
    source = compose_source_packet_text(
        job_description_text=(
            "Staff Platform Engineer\n\nOwn the developer platform, ship "
            "production services, and define reliability patterns for product teams."
        ),
        intake_notes_text="Must be hands-on and comfortable with ambiguous infrastructure work.",
    )
    result = synthesize_v2_from_source_packet(source_text=source)
    validate_v2_brief(result.v2_draft)
    assert result.source == "deterministic"
    assert result.v2_draft["role_title"] == "Staff Platform Engineer"
    assert result.v2_draft["engagement_context"] == {
        "selectivity_posture": "selective"
    }
    assert result.field_provenance


def test_deterministic_synthesis_over_budget_does_not_flag_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLORIS_DISABLE_INTAKE_LLM", "1")
    source = "x" * (SYNTHESIS_SOURCE_CHAR_BUDGET + 100)
    normalized = normalize_source_text(source)
    result = synthesize_v2_from_source_packet(source_text=source)
    assert result.source == "deterministic"
    assert result.source_truncated is False
    assert result.source_char_count == len(normalized)


def test_llm_synthesis_over_budget_flags_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CLORIS_DISABLE_INTAKE_LLM", raising=False)
    monkeypatch.setattr(
        "shared.source_packet_synthesis._has_llm_access",
        lambda: True,
    )
    llm_v2 = {
        "role_title": "Test Role",
        "capability_areas": [
            {
                "name": "Product engineering",
                "description": "Ships customer-facing systems end-to-end.",
            }
        ],
        "depth_distinction": {
            "builder_definition": "Owns architecture and ships.",
            "user_definition": "Maintains existing features.",
            "edge_case_guidance": "Borderline = full eval.",
        },
    }
    validate_v2_brief(llm_v2)
    monkeypatch.setattr(
        "shared.source_packet_synthesis.opus_llm_cached",
        lambda *_args, **_kwargs: llm_v2,
    )
    source = "x" * (SYNTHESIS_SOURCE_CHAR_BUDGET + 100)
    result = synthesize_v2_from_source_packet(source_text=source)
    assert result.source == "llm"
    assert result.source_truncated is True


def test_synthesis_no_truncation_under_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLORIS_DISABLE_INTAKE_LLM", "1")
    source = (
        "Staff Platform Engineer\n\nOwn the developer platform and ship "
        "production services."
    )
    result = synthesize_v2_from_source_packet(source_text=source)
    assert result.source_truncated is False
