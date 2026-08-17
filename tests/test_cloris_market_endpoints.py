"""Tests for the Phase E Slice E1 market viewer endpoints
(`GET /api/markets`, `GET /api/market/{market_key}`).

Pins the wire contract:
- Empty markets dir → empty list, NOT 500.
- A parseable on-disk artifact → MarketSummary in list + full detail.
- Unknown key → 404 with editorial error code.
- Most-recently-updated first.
- Malformed artifact silently skipped from the catalog.
- Lane R6: zero-evidence lanes omitted from detail.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloris.app import create_app


# Minimal valid artifact payload meeting `MarketIntelArtifact.from_dict`'s
# required-key check. Tests override individual fields per scenario.
def _valid_artifact_payload(
    *,
    market_key: str = "fde__nyc__ic5",
    role_title: str = "Forward Deployed Engineer",
    role_level: str = "IC5",
    geography: str = "New York",
    artifact_updated_at: str = "2026-04-12T23:12:00+00:00",
    save_rate: float = 0.0951,
    saved_count: int = 78,
    run_count: int = 3,
    lane_count: int = 2,
    talent_pool_count: int = 1,
) -> dict:
    lanes: list[dict] = []
    for i in range(lane_count):
        lanes.append(
            {
                "lane_key": f"lane_{i}",
                "domain_lane": "general",
                "novelty_bucket": "frontier",
                "status": "winning",
                "metrics": {
                    "strings_seen": 1,
                    "candidates_seen": 100 + i,
                    "saves": 20 + i,
                    "save_rate": 0.20 + 0.01 * i,
                    "facial_yes": 30,
                    "facial_no": 0,
                    "duplicates": 0,
                    "duplicate_rate": 0.0,
                },
                "first_seen_at": artifact_updated_at,
                "last_seen_at": artifact_updated_at,
                "supporting_run_refs": [
                    "linkedin:output/runs/linkedin/test/run-1"
                ],
                "dominant_anchors": [],
                "why_it_works": "This lane works because it does.",
                "recommended_action": "Keep it active.",
            }
        )
    talent_pools: list[dict] = []
    for i in range(talent_pool_count):
        talent_pools.append(
            {
                "pool_key": f"pool_{i}",
                "label": f"Pool {i}",
                "signal_strength": "high",
                "status": "active",
                "evidence_summary": "Strong saves observed.",
                "evidence_refs": [],
                "supporting_run_refs": [
                    "linkedin:output/runs/linkedin/test/run-1"
                ],
                "recommended_search_terms": [],
            }
        )
    return {
        "schema_version": 1,
        "artifact_version": 1,
        "market_identity": {
            "market_key": market_key,
            "role_title": role_title,
            "role_level": role_level,
            "geography": geography,
            "channels_seen": ["linkedin"],
            "brief_ids_seen": ["3000000007"],
            "brief_versions_seen": ["1.0"],
        },
        "freshness": {
            "artifact_updated_at": artifact_updated_at,
            "internal_data_through": artifact_updated_at,
            "external_research_through": artifact_updated_at,
            "staleness_days": 0,
        },
        "evidence_index": {},
        "aggregate_metrics": {
            "run_count": run_count,
            "saved_count": saved_count,
            "rejected_count": 58,
            "facial_yes_rate": 0.2634,
            "save_rate": save_rate,
            "candidate_volume_by_channel": {"linkedin": 820},
        },
        "channel_summaries": {},
        "lane_intelligence": lanes,
        "talent_pool_intelligence": talent_pools,
        "noise_patterns": [],
        "employer_signal_intelligence": [],
        "candidate_signal_summary": {},
        "market_thesis": {
            "summary": "Three runs sourced 820 candidates and saved 78.",
            "supply_assessment": "Accessible.",
            "competition_assessment": "Light competition.",
            "external_context": [],
        },
        "brief_recommendations": [],
        "open_questions": [],
        "retrieval_design_summary": {},
        "section_generation_metadata": {},
        "delta_since_last_run": {},
    }


def _seed_market(
    output_root: Path,
    market_key: str,
    payload: dict | None = None,
) -> Path:
    market_dir = output_root / "market_intelligence" / market_key
    market_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = market_dir / "market-intel.json"
    artifact_path.write_text(json.dumps(payload or _valid_artifact_payload(market_key=market_key)))
    return artifact_path


@pytest.fixture()
def client_with_isolated_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[TestClient, Path]:
    monkeypatch.setattr(
        "shared.output_paths.MARKET_INTELLIGENCE_ROOT",
        tmp_path / "market_intelligence",
    )
    return TestClient(create_app()), tmp_path


def test_markets_list_returns_empty_when_no_artifacts(client_with_isolated_root):
    client, _ = client_with_isolated_root
    res = client.get("/api/markets")
    assert res.status_code == 200
    body = res.json()
    assert body["slice"] == "v0-markets-list-1"
    assert body["markets"] == []


def test_markets_list_returns_summary_for_each_artifact(client_with_isolated_root):
    client, root = client_with_isolated_root
    _seed_market(
        root,
        "fde__nyc__ic5",
        _valid_artifact_payload(
            market_key="fde__nyc__ic5",
            role_title="Forward Deployed Engineer",
            geography="New York",
        ),
    )

    res = client.get("/api/markets")
    assert res.status_code == 200
    markets = res.json()["markets"]
    assert len(markets) == 1
    assert markets[0]["market_key"] == "fde__nyc__ic5"
    assert markets[0]["role_title"] == "Forward Deployed Engineer"
    assert markets[0]["aggregate_save_rate"] == pytest.approx(0.0951)
    assert markets[0]["run_count"] == 3
    assert markets[0]["saved_count"] == 78


def test_markets_list_sorted_most_recent_first(client_with_isolated_root):
    client, root = client_with_isolated_root
    _seed_market(
        root,
        "older",
        _valid_artifact_payload(
            market_key="older", artifact_updated_at="2026-01-01T00:00:00+00:00"
        ),
    )
    _seed_market(
        root,
        "newer",
        _valid_artifact_payload(
            market_key="newer", artifact_updated_at="2026-04-12T00:00:00+00:00"
        ),
    )

    res = client.get("/api/markets")
    assert res.status_code == 200
    markets = res.json()["markets"]
    assert [m["market_key"] for m in markets] == ["newer", "older"]


def test_markets_list_skips_malformed_artifact(client_with_isolated_root):
    client, root = client_with_isolated_root
    # Valid one — should appear.
    _seed_market(
        root,
        "ok",
        _valid_artifact_payload(market_key="ok"),
    )
    # Malformed: write JSON missing required fields.
    bad_dir = root / "market_intelligence" / "broken"
    bad_dir.mkdir(parents=True, exist_ok=True)
    (bad_dir / "market-intel.json").write_text('{"schema_version": 1}')

    res = client.get("/api/markets")
    assert res.status_code == 200
    keys = [m["market_key"] for m in res.json()["markets"]]
    assert keys == ["ok"]


def test_market_detail_returns_full_payload(client_with_isolated_root):
    client, root = client_with_isolated_root
    _seed_market(
        root,
        "fde__nyc__ic5",
        _valid_artifact_payload(market_key="fde__nyc__ic5", lane_count=2),
    )

    res = client.get("/api/market/fde__nyc__ic5")
    assert res.status_code == 200
    body = res.json()
    assert body["slice"] == "v0-market-detail-1"
    assert body["market_key"] == "fde__nyc__ic5"
    assert body["role_title"] == "Forward Deployed Engineer"
    assert len(body["lanes"]) == 2
    assert body["lanes"][0]["why_it_works"] == "This lane works because it does."
    assert body["market_thesis"]["summary"].startswith("Three runs")


def test_market_detail_404_for_unknown_key(client_with_isolated_root):
    client, _ = client_with_isolated_root
    res = client.get("/api/market/no-such-market")
    assert res.status_code == 404
    detail = res.json()["detail"]
    assert detail["error"] == "market_not_found"
    assert detail["market_key"] == "no-such-market"


def test_market_detail_omits_zero_evidence_lanes(
    client_with_isolated_root,
):
    """R6: a lane with zero candidates_seen + zero saves shouldn't ship."""

    client, root = client_with_isolated_root
    payload = _valid_artifact_payload(market_key="fde__nyc__ic5", lane_count=1)
    # Append a fully-empty lane that should be dropped.
    payload["lane_intelligence"].append(
        {
            "lane_key": "empty",
            "domain_lane": "general",
            "novelty_bucket": "frontier",
            "status": "tested",
            "metrics": {
                "strings_seen": 0,
                "candidates_seen": 0,
                "saves": 0,
                "save_rate": 0.0,
                "facial_yes": 0,
                "facial_no": 0,
                "duplicates": 0,
                "duplicate_rate": 0.0,
            },
            "first_seen_at": "2026-04-12T23:12:00+00:00",
            "last_seen_at": "2026-04-12T23:12:00+00:00",
            "supporting_run_refs": [
                "linkedin:output/runs/linkedin/test/run-1"
            ],
            "dominant_anchors": [],
        }
    )
    _seed_market(root, "fde__nyc__ic5", payload)

    res = client.get("/api/market/fde__nyc__ic5")
    assert res.status_code == 200
    body = res.json()
    # Only the populated lane survives.
    assert len(body["lanes"]) == 1
    assert body["lanes"][0]["lane_key"] != "empty"
