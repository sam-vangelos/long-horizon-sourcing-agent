#!/usr/bin/env python3
"""Build a master FTL LatAm candidate workbook from the shared CSV/XLSX exports.

This script uses only the Python standard library. It reads:

- output/fdl-brazil-phd-candidates.xlsx
- output/github/github_brazil?_saved_candidates.csv
- output/github-colombia/github_colombia_saved_candidates.csv
- output/ftl-latam-outreach-board.md

And writes:

- output/ftl-latam-master-candidates.xlsx
- output/ftl-latam-master-candidates.csv
"""

from __future__ import annotations

import csv
import datetime as dt
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path("/Users/jordan.rivera/Projects/recruiting-tools/sourcing-agent")
OUTPUT_DIR = ROOT / "output"
XLSX_SOURCE = OUTPUT_DIR / "fdl-brazil-phd-candidates.xlsx"
GITHUB_BRAZIL_CSV = OUTPUT_DIR / "github" / "github_brazil?_saved_candidates.csv"
GITHUB_COLOMBIA_CSV = OUTPUT_DIR / "github-colombia" / "github_colombia_saved_candidates.csv"
BOARD_MD = OUTPUT_DIR / "ftl-latam-outreach-board.md"

MASTER_XLSX = OUTPUT_DIR / "ftl-latam-master-candidates.xlsx"
MASTER_CSV = OUTPUT_DIR / "ftl-latam-master-candidates.csv"

SPREADSHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT_TYPES_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
CORE_PROPS_NS = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
DC_NS = "http://purl.org/dc/elements/1.1/"
DCTERMS_NS = "http://purl.org/dc/terms/"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"


def norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def norm_key(value: str) -> str:
    text = unicodedata.normalize("NFKD", norm_text(value)).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def unique_preserve(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in values:
        value = norm_text(raw)
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def split_capabilities(values: Iterable[str]) -> List[str]:
    parts: List[str] = []
    for value in values:
        for chunk in re.split(r"\s*(?:;|\||/|\+)\s*", value or ""):
            cleaned = norm_text(chunk)
            if cleaned:
                parts.append(cleaned)
    return unique_preserve(parts)


def join_multiline(values: Iterable[str], sep: str = "\n") -> str:
    return sep.join(unique_preserve(values))


def first_nonempty(records: List[dict], fields: Iterable[str], source_order: Tuple[str, ...] = ("xlsx", "csv")) -> str:
    for source_kind in source_order:
        for record in records:
            if record["_kind"] != source_kind:
                continue
            for field in fields:
                value = norm_text(record.get(field, ""))
                if value:
                    return value
    for record in records:
        for field in fields:
            value = norm_text(record.get(field, ""))
            if value:
                return value
    return ""


def col_to_num(col: str) -> int:
    value = 0
    for char in col:
        value = value * 26 + (ord(char.upper()) - 64)
    return value


def num_to_col(n: int) -> str:
    out = []
    while n:
        n, rem = divmod(n - 1, 26)
        out.append(chr(65 + rem))
    return "".join(reversed(out)) or "A"


def xml_cell_value(cell: ET.Element, shared_strings: List[str]) -> str:
    ns = {"main": SPREADSHEET_NS}
    cell_type = cell.attrib.get("t")
    value = cell.find("main:v", ns)
    inline = cell.find("main:is", ns)
    if cell_type == "s" and value is not None:
        return shared_strings[int(value.text)]
    if cell_type == "inlineStr" and inline is not None:
        return "".join((node.text or "") for node in inline.findall(".//main:t", ns))
    if value is not None and value.text is not None:
        return value.text
    return ""


def read_xlsx_sheet(path: Path, target_sheet: str) -> List[dict]:
    ns = {"main": SPREADSHEET_NS}
    with ZipFile(path) as zf:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("main:si", ns):
                shared_strings.append("".join((node.text or "") for node in si.findall(".//main:t", ns)))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}

        target = None
        for sheet in workbook.find("main:sheets", ns):
            if sheet.attrib["name"] == target_sheet:
                rid = sheet.attrib[f"{{{REL_NS}}}id"]
                target = rel_map[rid]
                break
        if target is None:
            raise ValueError(f"Sheet {target_sheet!r} not found in {path}")

        if target.startswith("/"):
            target = target.lstrip("/")
        elif not target.startswith("xl/"):
            target = "xl/" + target

        root = ET.fromstring(zf.read(target))
        rows = []
        max_col = 0
        for row in root.findall(".//main:sheetData/main:row", ns):
            row_map: Dict[int, str] = {}
            for cell in row.findall("main:c", ns):
                ref = cell.attrib.get("r", "")
                match = re.match(r"([A-Z]+)(\d+)", ref)
                if not match:
                    continue
                col = col_to_num(match.group(1))
                max_col = max(max_col, col)
                row_map[col] = xml_cell_value(cell, shared_strings)
            rows.append([row_map.get(i, "") for i in range(1, max_col + 1)])

    header = rows[0]
    return [dict(zip(header, row)) for row in rows[1:] if any(norm_text(str(value)) for value in row)]


def parse_board(board_path: Path) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    tier_map: Dict[str, dict] = {}
    validation_map: Dict[str, dict] = {}
    current_section = ""
    order_counter = 0

    lines = board_path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        if line.startswith("## "):
            current_section = line[3:].strip()
            continue
        if current_section in {"Tier A", "Tier B", "Tier C", "Do Not Sequence"} and line.startswith("- `"):
            match = re.match(r"- `([^`]+)` (.+)", line)
            if not match:
                continue
            label = match.group(1)
            remainder = match.group(2)
            parts = remainder.split(" — ", 2)
            if len(parts) == 3:
                name, country, note = parts
            elif len(parts) == 2:
                name, note = parts
                lower_note = note.lower()
                if lower_note.startswith("brazil "):
                    country = "Brazil"
                elif lower_note.startswith("colombia "):
                    country = "Colombia"
                elif lower_note.startswith("global "):
                    country = "Global / US"
                else:
                    country = ""
            else:
                continue
            order_counter += 1
            tier_map[norm_key(name)] = {
                "display_name": name,
                "tier": current_section,
                "tier_order": {"Tier A": 1, "Tier B": 2, "Tier C": 3, "Do Not Sequence": 5}[current_section],
                "board_order": order_counter,
                "source_label": label,
                "country": country,
                "board_note": note,
            }
            continue

        if line.startswith("| ") and not line.startswith("| ---"):
            cells = [cell.strip() for cell in line.strip().split("|")[1:-1]]
            if cells and cells[0] == "Candidate":
                current_section = "Validation Table"
                continue
            if current_section == "Validation Table" and len(cells) == 5:
                candidate, label, hook, validation, notes = cells
                validation_map[norm_key(candidate)] = {
                    "validation_label": label,
                    "validation_hook": hook,
                    "validation_status": validation,
                    "validation_notes": notes,
                }

    return tier_map, validation_map


def infer_country(location: str, source_files: List[str]) -> str:
    text = norm_key(location)
    if any(term in text for term in ["brazil", "brasil", "sao paulo", "campinas", "belo horizonte", "cuiaba", "minas gerais"]):
        return "Brazil"
    if any(term in text for term in ["colombia", "bogota", "medellin", "cali", "panama"]):
        return "Colombia"
    if any(term in text for term in ["california", "mountain view", "united states", "usa"]):
        return "Global / US"
    if any("github-colombia" in file_name for file_name in source_files):
        return "Colombia"
    if any("github_brazil" in file_name or "brazil" in file_name for file_name in source_files):
        return "Brazil"
    return "Unknown"


def parse_rank_list(records: List[dict]) -> str:
    ranks = []
    for record in records:
        raw = norm_text(record.get("Rank", ""))
        if not raw:
            continue
        try:
            ranks.append(int(raw))
        except ValueError:
            continue
    return ", ".join(str(rank) for rank in sorted(set(ranks)))


def parse_csv_records(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            record = dict(row)
            record["_kind"] = "csv"
            record["_file"] = path.name
            rows.append(record)
    return rows


ALIASES = {
    norm_key("Bryan L. M. de Oliveira"): "Bryan Oliveira",
    norm_key("Bryan Lincoln M. de Oliveira"): "Bryan Oliveira",
}


def canonical_name(raw_name: str, tier_map: Dict[str, dict]) -> str:
    key = norm_key(raw_name)
    alias = ALIASES.get(key)
    if alias:
        return alias
    if key in tier_map:
        return tier_map[key]["display_name"]
    return raw_name.strip()


def build_groups(tier_map: Dict[str, dict]) -> Dict[str, List[dict]]:
    records = []
    records.extend(parse_csv_records(GITHUB_BRAZIL_CSV))
    records.extend(parse_csv_records(GITHUB_COLOMBIA_CSV))
    for row in read_xlsx_sheet(XLSX_SOURCE, "Stack-Ranked Candidates"):
        record = dict(row)
        record["_kind"] = "xlsx"
        record["_file"] = XLSX_SOURCE.name
        records.append(record)

    groups: Dict[str, List[dict]] = defaultdict(list)
    for record in records:
        raw_name = " ".join(filter(None, [record.get("First Name", "").strip(), record.get("Last Name", "").strip()])).strip()
        if not raw_name:
            raw_name = first_nonempty([record], ["Headline", "Title", "GitHub Username"], source_order=("xlsx", "csv"))
        record["_raw_name"] = raw_name
        record["_canonical_name"] = canonical_name(raw_name, tier_map)
        key = norm_key(record["_canonical_name"]) or norm_key(record.get("GitHub URL", "")) or norm_key(record.get("LinkedIn URL", ""))
        groups[key].append(record)
    return groups


def summarize_group(
    key: str,
    records: List[dict],
    tier_map: Dict[str, dict],
    validation_map: Dict[str, dict],
) -> dict:
    canon_key = norm_key(records[0]["_canonical_name"])
    board = tier_map.get(canon_key, {})
    validation = validation_map.get(canon_key, {})

    source_files = unique_preserve(record["_file"] for record in records)
    source_platforms = unique_preserve(record.get("Source", "") or ("GitHub CSV" if record["_kind"] == "csv" else "LinkedIn Workbook") for record in records)
    raw_names = unique_preserve(record["_raw_name"] for record in records)

    has_github_csv = any(record["_kind"] == "csv" for record in records)
    has_workbook_github = any(record["_kind"] == "xlsx" and norm_text(record.get("Source", "")) == "GitHub" for record in records)
    has_workbook_linkedin = any(record["_kind"] == "xlsx" and norm_text(record.get("Source", "")) == "LinkedIn" for record in records)
    github_url = first_nonempty(records, ["GitHub URL"], source_order=("csv", "xlsx"))
    linkedin_url = first_nonempty(records, ["LinkedIn URL"], source_order=("xlsx", "csv"))
    github_message_subject = first_nonempty(records, ["Outreach Subject"], source_order=("csv",))
    github_message = first_nonempty(records, ["Outreach Message"], source_order=("csv",))

    if has_github_csv and has_workbook_linkedin:
        source_label = "Both / GH-msg"
    elif has_github_csv:
        source_label = "GH-msg"
    elif has_workbook_github or github_url:
        source_label = "GH-only"
    else:
        source_label = "LI"

    if board:
        tier = board["tier"]
        tier_order = board["tier_order"]
        board_order = board["board_order"]
        board_country = board["country"]
        board_note = board["board_note"]
        board_label = board["source_label"]
    else:
        tier = "Unreviewed / hold"
        tier_order = 4
        board_order = 9999
        board_country = ""
        board_note = ""
        board_label = ""

    country = board_country or infer_country(join_multiline(record.get("Location", "") for record in records), source_files)

    if tier == "Tier A":
        recommended_action = "Full personalized GEM sequence"
    elif tier == "Tier B":
        recommended_action = "Semi-personalized / broader-net GEM sequence"
    elif tier == "Tier C":
        recommended_action = "Broad-net or reach-shot GEM sequence"
    elif tier == "Do Not Sequence":
        recommended_action = "Skip"
    else:
        recommended_action = "Review before sequencing"

    dedupe_note = ""
    if len(raw_names) > 1:
        dedupe_note = "Merged duplicate/alias records: " + "; ".join(raw_names)

    return {
        "Outreach Order": board_order,
        "Tier": tier,
        "Tier Order": tier_order,
        "Country Bucket": country,
        "Candidate Name": board.get("display_name", records[0]["_canonical_name"]),
        "Merged From Names": "\n".join(raw_names),
        "Source Label": source_label,
        "Board Source Label": board_label,
        "Source Platforms": "\n".join(source_platforms),
        "Source Files": "\n".join(source_files),
        "GitHub Provenance": "Yes" if (has_github_csv or has_workbook_github or github_url) else "No",
        "GitHub Message Available": "Yes" if github_message_subject or github_message else "No",
        "GitHub Validation": validation.get("validation_status", ""),
        "Recommended Action": recommended_action,
        "Current Company": first_nonempty(records, ["Current Company", "Company"], source_order=("xlsx", "csv")),
        "Current Title": first_nonempty(records, ["Current Title", "Title"], source_order=("xlsx", "csv")),
        "Headline / Profile Summary": first_nonempty(records, ["Headline", "Title"], source_order=("xlsx", "csv")),
        "Capability Areas": "\n".join(split_capabilities(
            [record.get("Capability Path", "") for record in records] + [record.get("Capability Area", "") for record in records]
        )),
        "Education": first_nonempty(records, ["Education"], source_order=("xlsx",)),
        "Skills / Toolchain": "\n".join(unique_preserve(
            [record.get("Key Skills", "") for record in records] + [record.get("Toolchain", "") for record in records]
        )),
        "Top Repos": "\n".join(unique_preserve(record.get("Top Repos", "") for record in records)),
        "Email": first_nonempty(records, ["Email"], source_order=("csv", "xlsx")),
        "LinkedIn URL": linkedin_url,
        "GitHub URL": github_url,
        "Website": first_nonempty(records, ["Website"], source_order=("csv",)),
        "Location (Raw)": "\n".join(unique_preserve(record.get("Location", "") for record in records)),
        "Curated Outreach Note": board_note,
        "Validation Hook": validation.get("validation_hook", ""),
        "Validation Notes": validation.get("validation_notes", ""),
        "Raw Fit Evidence": first_nonempty(records, ["Rationale", "Evaluation Summary"], source_order=("xlsx", "csv")),
        "GitHub Subject": github_message_subject,
        "GitHub Message": github_message,
        "Original Rank(s)": parse_rank_list(records),
        "Dedupe Notes": dedupe_note,
    }


def build_master_rows() -> List[dict]:
    tier_map, validation_map = parse_board(BOARD_MD)
    groups = build_groups(tier_map)
    rows = [summarize_group(key, records, tier_map, validation_map) for key, records in groups.items()]
    rows.sort(key=lambda row: (
        int(row["Tier Order"]),
        int(row["Outreach Order"]) if str(row["Outreach Order"]).isdigit() else 999999,
        row["Country Bucket"],
        row["Candidate Name"].lower(),
    ))
    return rows


def csv_export(rows: List[dict]) -> None:
    fieldnames = list(rows[0].keys())
    with MASTER_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def xml_escape_text(value: str) -> str:
    text = escape(value or "", {'"': "&quot;"})
    return text.replace("\r\n", "\n").replace("\r", "\n")


def make_inline_string_cell(ref: str, value: str, style_id: int) -> str:
    text = xml_escape_text(value)
    return f'<c r="{ref}" s="{style_id}" t="inlineStr"><is><t xml:space="preserve">{text}</t></is></c>'


def build_sheet_xml(sheet_name: str, rows: List[List[str]], widths: List[int], tier_col_idx: int | None = None) -> str:
    max_col = max((len(row) for row in rows), default=1)
    max_row = max(len(rows), 1)
    cols_xml = []
    for idx, width in enumerate(widths, start=1):
        cols_xml.append(f'<col min="{idx}" max="{idx}" width="{width}" customWidth="1"/>')

    data_rows = []
    for row_idx, row in enumerate(rows, start=1):
        cells = []
        for col_idx in range(1, max_col + 1):
            value = row[col_idx - 1] if col_idx - 1 < len(row) else ""
            ref = f"{num_to_col(col_idx)}{row_idx}"
            style_id = 1 if row_idx == 1 else 2
            if tier_col_idx is not None and row_idx > 1 and col_idx == tier_col_idx:
                tier_value = value
                style_id = {
                    "Tier A": 3,
                    "Tier B": 4,
                    "Tier C": 5,
                    "Do Not Sequence": 6,
                    "Unreviewed / hold": 7,
                }.get(tier_value, 2)
            cells.append(make_inline_string_cell(ref, value, style_id))
        data_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    auto_filter_ref = f"A1:{num_to_col(max_col)}{max_row}"
    xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="{SPREADSHEET_NS}" xmlns:r="{REL_NS}">
  <sheetViews>
    <sheetView workbookViewId="0">
      <pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>
      <selection pane="bottomLeft" activeCell="A2" sqref="A2"/>
    </sheetView>
  </sheetViews>
  <sheetFormatPr defaultRowHeight="18"/>
  <cols>{"".join(cols_xml)}</cols>
  <sheetData>{"".join(data_rows)}</sheetData>
  <autoFilter ref="{auto_filter_ref}"/>
</worksheet>
'''
    return xml


def build_summary_sheet(rows: List[dict]) -> List[List[str]]:
    counter_tier = Counter(row["Tier"] for row in rows)
    counter_source = Counter(row["Source Label"] for row in rows)
    counter_country = Counter(row["Country Bucket"] for row in rows)

    sheet_rows = [
        ["FTL LatAm Master Workbook", ""],
        ["Generated", dt.datetime.now().isoformat(timespec="seconds")],
        ["Role scope", "General FTL role across Brazil and Colombia"],
        ["Raw row count", "173"],
        ["Unique candidate count", str(len(rows))],
        ["GitHub provenance count", str(sum(1 for row in rows if row["GitHub Provenance"] == "Yes"))],
        ["", ""],
        ["Tier counts", ""],
    ]
    for tier in ["Tier A", "Tier B", "Tier C", "Unreviewed / hold", "Do Not Sequence"]:
        sheet_rows.append([tier, str(counter_tier.get(tier, 0))])
    sheet_rows.extend([["", ""], ["Source label counts", ""]])
    for label in ["Both / GH-msg", "GH-msg", "GH-only", "LI"]:
        sheet_rows.append([label, str(counter_source.get(label, 0))])
    sheet_rows.extend([["", ""], ["Country counts", ""]])
    for country, count in sorted(counter_country.items()):
        sheet_rows.append([country, str(count)])
    sheet_rows.extend([
        ["", ""],
        ["Source file", str(XLSX_SOURCE)],
        ["Source file", str(GITHUB_BRAZIL_CSV)],
        ["Source file", str(GITHUB_COLOMBIA_CSV)],
        ["Board source", str(BOARD_MD)],
    ])
    return sheet_rows


def write_xlsx(rows: List[dict]) -> None:
    master_headers = list(rows[0].keys())
    master_matrix = [master_headers] + [[str(row.get(header, "")) for header in master_headers] for row in rows]

    github_rows = [row for row in rows if row["GitHub Provenance"] == "Yes"]
    github_headers = [
        "Tier",
        "Country Bucket",
        "Candidate Name",
        "Source Label",
        "GitHub Message Available",
        "GitHub Validation",
        "Validation Hook",
        "Current Company",
        "Headline / Profile Summary",
        "Top Repos",
        "Skills / Toolchain",
        "GitHub URL",
        "GitHub Subject",
        "GitHub Message",
        "Curated Outreach Note",
        "Validation Notes",
    ]
    github_matrix = [github_headers] + [[str(row.get(header, "")) for header in github_headers] for row in github_rows]

    summary_matrix = build_summary_sheet(rows)

    master_widths = [10, 16, 14, 28, 28, 16, 16, 22, 12, 12, 14, 20, 22, 34, 28, 24, 28, 22, 22, 32, 32, 24, 20, 36, 32, 48, 28, 56, 20, 28]
    github_widths = [14, 14, 28, 16, 12, 14, 24, 20, 34, 22, 24, 32, 28, 56, 36, 36]
    summary_widths = [28, 40]

    sheet_xml = [
        build_sheet_xml("Master Candidates", master_matrix, master_widths, tier_col_idx=2),
        build_sheet_xml("GitHub Reference", github_matrix, github_widths, tier_col_idx=1),
        build_sheet_xml("Summary", summary_matrix, summary_widths, tier_col_idx=None),
    ]

    workbook_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="{SPREADSHEET_NS}" xmlns:r="{REL_NS}">
  <workbookViews><workbookView activeTab="0"/></workbookViews>
  <sheets>
    <sheet name="Master Candidates" sheetId="1" r:id="rId1"/>
    <sheet name="GitHub Reference" sheetId="2" r:id="rId2"/>
    <sheet name="Summary" sheetId="3" r:id="rId3"/>
  </sheets>
</workbook>
'''

    workbook_rels = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{PKG_REL_NS}">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet3.xml"/>
  <Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
'''

    root_rels = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{PKG_REL_NS}">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
'''

    content_types = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="{CONTENT_TYPES_NS}">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/worksheets/sheet3.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
'''

    styles_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2">
    <font><sz val="11"/><name val="Calibri"/><family val="2"/></font>
    <font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/><family val="2"/></font>
  </fonts>
    <fills count="8">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFD9EAD3"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFFFF2CC"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFFCE5CD"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFF4CCCC"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFD9D9D9"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="1">
    <border><left/><right/><top/><bottom/><diagonal/></border>
  </borders>
  <cellStyleXfs count="1">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>
  </cellStyleXfs>
  <cellXfs count="8">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="0" fillId="3" borderId="0" xfId="0" applyFill="1" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="0" fillId="4" borderId="0" xfId="0" applyFill="1" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="0" fillId="5" borderId="0" xfId="0" applyFill="1" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="0" fillId="6" borderId="0" xfId="0" applyFill="1" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="0" fillId="7" borderId="0" xfId="0" applyFill="1" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>
'''

    app_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
  <DocSecurity>0</DocSecurity>
  <ScaleCrop>false</ScaleCrop>
  <HeadingPairs>
    <vt:vector size="2" baseType="variant">
      <vt:variant><vt:lpstr>Worksheets</vt:lpstr></vt:variant>
      <vt:variant><vt:i4>3</vt:i4></vt:variant>
    </vt:vector>
  </HeadingPairs>
  <TitlesOfParts>
    <vt:vector size="3" baseType="lpstr">
      <vt:lpstr>Master Candidates</vt:lpstr>
      <vt:lpstr>GitHub Reference</vt:lpstr>
      <vt:lpstr>Summary</vt:lpstr>
    </vt:vector>
  </TitlesOfParts>
  <Company>OpenAI Codex</Company>
  <LinksUpToDate>false</LinksUpToDate>
  <SharedDoc>false</SharedDoc>
  <HyperlinksChanged>false</HyperlinksChanged>
  <AppVersion>16.0300</AppVersion>
</Properties>
'''

    timestamp = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    core_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="{CORE_PROPS_NS}" xmlns:dc="{DC_NS}" xmlns:dcterms="{DCTERMS_NS}" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="{XSI_NS}">
  <dc:title>FTL LatAm Master Candidates</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:modified>
</cp:coreProperties>
'''

    with ZipFile(MASTER_XLSX, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("docProps/app.xml", app_xml)
        zf.writestr("docProps/core.xml", core_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/styles.xml", styles_xml)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml[0])
        zf.writestr("xl/worksheets/sheet2.xml", sheet_xml[1])
        zf.writestr("xl/worksheets/sheet3.xml", sheet_xml[2])


def main() -> None:
    rows = build_master_rows()
    if not rows:
        raise SystemExit("No rows generated.")
    csv_export(rows)
    write_xlsx(rows)
    print(f"Wrote {len(rows)} unique candidates to:")
    print(f"- {MASTER_XLSX}")
    print(f"- {MASTER_CSV}")


if __name__ == "__main__":
    main()
