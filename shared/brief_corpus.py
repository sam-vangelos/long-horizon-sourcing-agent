"""Accepted-brief corpus for source-packet intake exemplar retrieval."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from shared.runtime_state.store import RuntimeStateStore


@dataclass(frozen=True)
class CorpusHit:
    brief_key: str
    title: str
    excerpt: str
    v2_json: dict[str, Any]


def install_schema(conn: sqlite3.Connection) -> None:
    """Install accepted-brief corpus tables into the shared SQLite DB."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS brief_corpus (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brief_key TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            body_text TEXT NOT NULL,
            v2_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_brief_corpus_key
        ON brief_corpus(brief_key);
        """
    )
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS brief_corpus_fts USING fts5(
            title,
            body_text,
            content='brief_corpus',
            content_rowid='id'
        );
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS brief_corpus_ai
        AFTER INSERT ON brief_corpus BEGIN
            INSERT INTO brief_corpus_fts(rowid, title, body_text)
            VALUES (new.id, new.title, new.body_text);
        END;
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS brief_corpus_ad
        AFTER DELETE ON brief_corpus BEGIN
            INSERT INTO brief_corpus_fts(brief_corpus_fts, rowid)
            VALUES('delete', old.id);
        END;
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS brief_corpus_au
        AFTER UPDATE ON brief_corpus BEGIN
            INSERT INTO brief_corpus_fts(brief_corpus_fts, rowid)
            VALUES('delete', old.id);
            INSERT INTO brief_corpus_fts(rowid, title, body_text)
            VALUES (new.id, new.title, new.body_text);
        END;
        """
    )


def index_v2_brief(
    store: RuntimeStateStore,
    *,
    brief_key: str,
    v2_json: dict[str, Any],
    title: str | None = None,
) -> None:
    """Upsert an accepted V2 brief into the retrieval corpus."""

    now = datetime.now(timezone.utc).isoformat()
    role_title = title or str(v2_json.get("role_title") or brief_key)
    body = _body_text(v2_json)
    with store.connect() as conn:
        conn.execute(
            """
            INSERT INTO brief_corpus(brief_key, title, body_text, v2_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(brief_key) DO UPDATE SET
                title = excluded.title,
                body_text = excluded.body_text,
                v2_json = excluded.v2_json,
                updated_at = excluded.updated_at
            """,
            (brief_key, role_title, body, json.dumps(v2_json), now, now),
        )


def query_corpus(
    store: RuntimeStateStore,
    *,
    source_excerpt: str,
    limit: int = 6,
) -> list[CorpusHit]:
    """Return corpus hits for a new source packet."""

    q = _sanitize_fts_query(source_excerpt)
    with store.connect() as conn:
        try:
            rows = conn.execute(
                """
                SELECT c.brief_key, c.title, c.body_text, c.v2_json
                FROM brief_corpus_fts AS f
                JOIN brief_corpus AS c ON c.id = f.rowid
                WHERE f MATCH ?
                ORDER BY bm25(f)
                LIMIT ?
                """,
                (q, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        if not rows:
            rows = conn.execute(
                """
                SELECT brief_key, title, body_text, v2_json
                FROM brief_corpus
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
    hits: list[CorpusHit] = []
    for row in rows:
        try:
            v2 = json.loads(row["v2_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            v2 = {}
        if not isinstance(v2, dict):
            v2 = {}
        body = str(row["body_text"] or "").replace("\n", " ")
        hits.append(
            CorpusHit(
                brief_key=str(row["brief_key"]),
                title=str(row["title"]),
                excerpt=body[:420],
                v2_json=v2,
            )
        )
    return hits


def build_exemplar_block(
    store: RuntimeStateStore,
    source_text: str,
    *,
    limit: int = 4,
) -> tuple[str, list[str]]:
    """Return a short exemplar block plus the corpus ids used."""

    hits = query_corpus(store, source_excerpt=source_text, limit=max(limit, 1))
    lines: list[str] = []
    used: list[str] = []
    for hit in hits[:limit]:
        used.append(hit.brief_key)
        lines.append(f"- {hit.title}: {hit.excerpt}")
    return "\n".join(lines), used


def _sanitize_fts_query(text: str) -> str:
    tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_+.-]*", text.lower())[:24]
    return " OR ".join(tokens) if tokens else "role"


def _body_text(v2: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("role_title", "role_level", "role_summary", "minimum_bar_description"):
        value = v2.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    for area in v2.get("capability_areas") or []:
        if isinstance(area, dict):
            for key in ("name", "description"):
                value = area.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
    depth = v2.get("depth_distinction")
    if isinstance(depth, dict):
        for key in ("builder_definition", "user_definition", "edge_case_guidance"):
            value = depth.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
    return "\n".join(parts)
