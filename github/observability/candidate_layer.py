"""Layer 3: Candidate outcomes — saves, close rejects, FACIAL_NO aggregation.

Writes session_*_candidates.json once at session end.
"""

from __future__ import annotations

import json
from pathlib import Path


# Keyword categories for FACIAL_NO rationale bucketing
FACIAL_NO_CATEGORIES: dict[str, list[str]] = {
    "web_developer": ["web developer", "frontend", "react", "angular", "vue", "css", "html", "next.js", "nuxt"],
    "devops_sre": ["devops", "sre", "kubernetes", "terraform", "infrastructure", "ansible", "docker only"],
    "data_analyst": ["data analyst", "tableau", "power bi", "business intelligence", "excel"],
    "student_tutorial": ["student", "tutorial", "learning", "coursework", "bootcamp", "beginner"],
    "no_relevant_repos": ["no relevant repos", "empty profile", "no public", "insufficient"],
    "generic_ml_user": ["api wrapper", "inference only", "no training", "uses but doesn't build"],
    "mobile_developer": ["mobile", "android", "ios", "flutter", "react native", "swift"],
    "game_developer": ["game", "unity", "unreal", "godot"],
    "embedded_iot": ["embedded", "iot", "arduino", "raspberry pi", "firmware"],
    "academic_only": ["academic", "coursework only", "class project", "homework"],
}


class CandidateLayer:
    def __init__(self, output_path: Path):
        self._path = output_path
        self._saves: list[dict] = []
        self._close_rejects: list[dict] = []
        self._facial_no_by_query: dict[str, dict] = {}  # query_key -> {total, categories}
        self._capability_distribution: dict[str, dict] = {}  # area -> {count, transferable, direct}

    def record_save(
        self,
        username: str,
        candidate,
        decision,
        query,
        result_rank: int = 0,
    ):
        contact_str = ", ".join(candidate.contact.emails[:3]) if candidate.contact.emails else ""
        save_entry = {
            "username": username,
            "name": candidate.user.name or username,
            "github_url": candidate.user.profile_url,
            "location": candidate.user.location,
            "decision_type": decision.decision,
            "confidence": decision.confidence,
            "capability_area": decision.path,
            "rationale": decision.rationale,
            "case_for": decision.rationale,  # in current schema, rationale serves as case_for
            "query_name": query.name,
            "query_channel": query.channel,
            "query_id": query.id,
            "result_rank": result_rank,
            "contact_emails": candidate.contact.emails,
            "contact_website": candidate.contact.website,
        }
        self._saves.append(save_entry)

        # Capability distribution
        area = decision.path or "unknown"
        match_type = "transferable" if "TRANSFERABLE" in decision.decision else "direct"
        if area not in self._capability_distribution:
            self._capability_distribution[area] = {"count": 0, "direct": 0, "transferable": 0}
        self._capability_distribution[area]["count"] += 1
        self._capability_distribution[area][match_type] += 1

    def record_close_reject(self, username: str, candidate, decision, query):
        self._close_rejects.append({
            "username": username,
            "name": candidate.user.name or username,
            "github_url": candidate.user.profile_url,
            "reason": decision.rationale,
            "capability_area": decision.path,
            "query_name": query.name,
            "confidence": decision.confidence,
        })

    def record_facial_no(self, username: str, rationale: str, query):
        q_key = f"query:{query.id}"
        if q_key not in self._facial_no_by_query:
            self._facial_no_by_query[q_key] = {"total": 0, "categories": {}}

        self._facial_no_by_query[q_key]["total"] += 1

        # Categorize
        rationale_lower = rationale.lower()
        categorized = False
        for category, keywords in FACIAL_NO_CATEGORIES.items():
            if any(kw in rationale_lower for kw in keywords):
                cats = self._facial_no_by_query[q_key]["categories"]
                cats[category] = cats.get(category, 0) + 1
                categorized = True
                break
        if not categorized:
            cats = self._facial_no_by_query[q_key]["categories"]
            cats["other"] = cats.get("other", 0) + 1

    def write(self):
        output = {
            "saves": self._saves,
            "close_rejects": self._close_rejects,
            "facial_no_patterns": self._facial_no_by_query,
            "capability_distribution": self._capability_distribution,
        }

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(output, f, indent=2)
