"""Hugging Face Hub registry client.

Retrieval uses fuzzy listing endpoints; returned artifacts are gated to exact
name-segment matches (the segment after ``/`` in the artifact id) —
retrieval-fuzzy, decision-exact. No emails appear on these API surfaces; only
normalized shapes are cached.
"""

from __future__ import annotations

import re
from datetime import timedelta
from typing import Any, Callable
from urllib.parse import quote

from github.hubs.base import BaseHubClient, normalize_repo_url

API_BASE = "https://huggingface.co/api"
USER_AGENT = "cloris-sourcing-agent/1.0 (huggingface hub client)"
MIN_REQUEST_INTERVAL_SECONDS = 1.0

CACHE_KINDS: frozenset[str] = frozenset(
    {
        "artifact_matches",
        "model_info",
    }
)

TTL_BY_KIND: dict[str, timedelta] = {
    "artifact_matches": timedelta(days=7),
    "model_info": timedelta(days=7),
}

_GITHUB_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/[\w.-]+/[\w.-]+(?:\.git)?",
    re.IGNORECASE,
)


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return None


def _name_segment(artifact_id: str) -> str | None:
    parts = artifact_id.strip().split("/", 1)
    if len(parts) != 2 or not parts[1]:
        return None
    return parts[1]


def _name_matches_token(artifact_id: str, token: str) -> bool:
    segment = _name_segment(artifact_id)
    if segment is None:
        return False
    return segment.lower() == token.strip().lower()


def _artifact_id_from_raw(raw: dict[str, Any]) -> str | None:
    for key in ("id", "modelId"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def normalize_artifact_entry(
    raw: dict[str, Any],
    *,
    kind: str,
    token: str,
) -> dict[str, Any] | None:
    """Normalize one listing entry when its name segment exactly matches token."""
    artifact_id = _artifact_id_from_raw(raw)
    if artifact_id is None:
        return None
    # Retrieval-fuzzy, decision-exact: listing may return partial matches; only
    # the name segment (after ``/``) must equal the token case-insensitively.
    if not _name_matches_token(artifact_id, token):
        return None

    author_raw = raw.get("author")
    if isinstance(author_raw, str) and author_raw.strip():
        author = author_raw.strip()
    else:
        author = artifact_id.split("/", 1)[0]

    return {
        "artifact_id": artifact_id,
        "kind": kind,
        "author": author,
        "downloads": _int_or_none(raw.get("downloads")),
        "likes": _int_or_none(raw.get("likes")),
    }


def _github_urls_in_text(text: str) -> list[str]:
    return [match.group(0) for match in _GITHUB_URL_RE.finditer(text)]


def _walk_for_github_strings(
    value: Any,
    collector: Callable[[str], None],
) -> None:
    if isinstance(value, str):
        collector(value)
    elif isinstance(value, dict):
        for nested in value.values():
            _walk_for_github_strings(nested, collector)
    elif isinstance(value, list):
        for nested in value:
            _walk_for_github_strings(nested, collector)


def github_links_from_artifact(raw: dict[str, Any]) -> list[str]:
    """Extract normalized github.com repo URLs from cardData, tags, and metadata."""
    seen: set[str] = set()
    ordered: list[str] = []

    def collect(text: str) -> None:
        for candidate in _github_urls_in_text(text):
            normalized = normalize_repo_url(candidate)
            if normalized is not None and normalized not in seen:
                seen.add(normalized)
                ordered.append(normalized)

    tags = raw.get("tags")
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, str):
                collect(tag)

    card_data = raw.get("cardData")
    if isinstance(card_data, dict):
        _walk_for_github_strings(card_data, collect)

    for key in ("description", "readme"):
        field = raw.get(key)
        if isinstance(field, str):
            collect(field)

    return ordered


class HuggingFaceHubClient(BaseHubClient):
    """Async client for the Hugging Face Hub API."""

    hub = "hf"
    user_agent = USER_AGENT
    min_request_interval_seconds = MIN_REQUEST_INTERVAL_SECONDS
    cache_kinds = CACHE_KINDS
    ttl_by_kind = TTL_BY_KIND

    def _probe_url(self) -> str:
        return f"{API_BASE}/models?limit=1"

    async def find_exact_artifacts(
        self,
        token: str,
        *,
        max_artifacts: int = 5,
    ) -> list[dict[str, Any]]:
        token_text = str(token or "").strip()
        if not token_text:
            return []

        cache_key = token_text.lower()
        cached = self._cache_get("artifact_matches", cache_key)
        if isinstance(cached, list):
            return cached

        encoded = quote(token_text, safe="")
        models_url = f"{API_BASE}/models?search={encoded}&limit=50"
        datasets_url = f"{API_BASE}/datasets?search={encoded}&limit=50"

        matches: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        models_status, models_body = await self._get_json(models_url)
        models_ok = models_status == 200 and isinstance(models_body, list)
        if models_ok:
            for entry in models_body:
                if not isinstance(entry, dict):
                    continue
                normalized = normalize_artifact_entry(entry, kind="model", token=token_text)
                if normalized is None:
                    continue
                artifact_id = normalized["artifact_id"]
                if artifact_id in seen_ids:
                    continue
                seen_ids.add(artifact_id)
                matches.append(normalized)

        datasets_status, datasets_body = await self._get_json(datasets_url)
        datasets_ok = datasets_status == 200 and isinstance(datasets_body, list)
        if datasets_ok:
            for entry in datasets_body:
                if not isinstance(entry, dict):
                    continue
                normalized = normalize_artifact_entry(entry, kind="dataset", token=token_text)
                if normalized is None:
                    continue
                artifact_id = normalized["artifact_id"]
                if artifact_id in seen_ids:
                    continue
                seen_ids.add(artifact_id)
                matches.append(normalized)

        # Cap fan-out: downstream issues one throttled request per artifact for
        # card links; without a limit, 50 models + 50 datasets per token can
        # balloon into thousands of calls on multi-token queries.
        matches.sort(
            key=lambda m: m.get("downloads") if m.get("downloads") is not None else -1,
            reverse=True,
        )
        matches = matches[:max_artifacts]

        if models_ok and datasets_ok:
            self._cache_put("artifact_matches", cache_key, matches)
        return matches

    async def get_model_card_links(self, artifact_id: str) -> list[str]:
        artifact = str(artifact_id or "").strip()
        if not artifact:
            return []

        cache_key = artifact
        cached = self._cache_get("model_info", cache_key)
        if isinstance(cached, list):
            return cached

        encoded = quote(artifact, safe="")
        models_url = f"{API_BASE}/models/{encoded}"
        _status, body = await self._get_json(models_url)
        if not isinstance(body, dict):
            datasets_url = f"{API_BASE}/datasets/{encoded}"
            _status, body = await self._get_json(datasets_url)

        if not isinstance(body, dict):
            return []

        links = github_links_from_artifact(body)
        self._cache_put("model_info", cache_key, links)
        return links
