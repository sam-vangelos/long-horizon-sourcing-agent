"""Layer 2: Graph traversal — adjacency list of queries, candidates, expansions.

Builds in memory, writes session_*_graph.json once at session end.
"""

from __future__ import annotations

import json
from pathlib import Path


class GraphLayer:
    def __init__(self, output_path: Path):
        self._path = output_path
        self._nodes: dict[str, dict] = {}
        self._edges: list[dict] = []
        self._cross_refs: dict[str, list[str]] = {}  # candidate -> [query_keys]

    def register_query(self, query):
        key = f"query:{query.id}"
        self._nodes[key] = {
            "type": "query",
            "channel": query.channel,
            "name": query.name,
            "candidates_found": 0,
            "saves": [],
        }

    def record_discovery(self, username: str, query, rank: int):
        q_key = f"query:{query.id}"
        c_key = f"candidate:{username}"

        # Update query node
        if q_key in self._nodes:
            self._nodes[q_key]["candidates_found"] += 1

        # Track cross-references
        self._cross_refs.setdefault(username, [])
        if q_key not in self._cross_refs[username]:
            self._cross_refs[username].append(q_key)

        # Create/update candidate node
        if c_key not in self._nodes:
            self._nodes[c_key] = {
                "type": "candidate",
                "outcome": "pending",
                "capability_area": "",
                "confidence": 0.0,
                "discovered_via": [q_key],
            }
        else:
            if q_key not in self._nodes[c_key]["discovered_via"]:
                self._nodes[c_key]["discovered_via"].append(q_key)

        self._edges.append({
            "from": q_key,
            "to": c_key,
            "type": "discovered",
            "rank": rank,
        })

    def record_save(self, username: str, query, confidence: float, capability_area: str):
        q_key = f"query:{query.id}"
        c_key = f"candidate:{username}"

        if q_key in self._nodes:
            self._nodes[q_key]["saves"].append(username)

        if c_key in self._nodes:
            self._nodes[c_key]["outcome"] = "SAVE"
            self._nodes[c_key]["confidence"] = confidence
            self._nodes[c_key]["capability_area"] = capability_area

        # Track expansion effectiveness — if this save came from a graph_expansion query,
        # increment saves counter on the expansion node for the seed
        if query.channel == "graph_expansion":
            seed = query.query  # username stored in query field
            exp_key = f"expansion:{seed}"
            if exp_key in self._nodes:
                self._nodes[exp_key].setdefault("saves_from_expansion", 0)
                self._nodes[exp_key]["saves_from_expansion"] += 1

    def record_reject(self, username: str):
        c_key = f"candidate:{username}"
        if c_key in self._nodes:
            self._nodes[c_key]["outcome"] = "REJECT"

    def record_facial_no(self, username: str):
        c_key = f"candidate:{username}"
        if c_key in self._nodes:
            self._nodes[c_key]["outcome"] = "FACIAL_NO"

    def record_expansion_queued(self, username: str, confidence: float, capability_area: str):
        c_key = f"candidate:{username}"
        exp_key = f"expansion:{username}"
        self._nodes[exp_key] = {
            "type": "expansion",
            "seed": username,
            "confidence": confidence,
            "capability_area": capability_area,
            "yield_queries": 0,
        }
        self._edges.append({
            "from": c_key,
            "to": exp_key,
            "type": "triggered_expansion",
        })

    def record_expansion_processed(self, seeds: list[dict], new_query_count: int):
        for seed in seeds:
            exp_key = f"expansion:{seed['username']}"
            if exp_key in self._nodes:
                self._nodes[exp_key]["yield_queries"] = new_query_count

    def write(self):
        # Build cross-references list
        cross_references = []
        for candidate, query_keys in self._cross_refs.items():
            if len(query_keys) > 1:
                cross_references.append({
                    "candidate": candidate,
                    "found_via": query_keys,
                })

        # Build branch summary
        branch_summary = []
        for key, node in self._nodes.items():
            if node["type"] == "query":
                saves_count = len(node.get("saves", []))
                branch_summary.append({
                    "root": key,
                    "channel": node["channel"],
                    "candidates_found": node["candidates_found"],
                    "saves": saves_count,
                    "productive": saves_count > 0,
                })

        # Build expansion effectiveness summary
        expansion_summary = {
            "seeds": 0,
            "candidates_discovered": 0,
            "saves": 0,
        }
        for key, node in self._nodes.items():
            if node["type"] == "expansion":
                expansion_summary["seeds"] += 1
                expansion_summary["saves"] += node.get("saves_from_expansion", 0)
        # Count candidates from graph_expansion queries
        for key, node in self._nodes.items():
            if node["type"] == "query" and node["channel"] == "graph_expansion":
                expansion_summary["candidates_discovered"] += node["candidates_found"]

        graph = {
            "nodes": self._nodes,
            "edges": self._edges,
            "cross_references": cross_references,
            "branch_summary": branch_summary,
            "expansion_summary": expansion_summary,
        }

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(graph, f, indent=2)
