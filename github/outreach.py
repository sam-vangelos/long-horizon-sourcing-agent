"""Generate personalized outreach copy for saved GitHub candidates.

This module produces recruiter-ready message drafts that are STORED ONLY —
never sent autonomously. The recruiter reviews each draft and decides
whether and how to use it. All copy references the candidate's actual work
(repos, READMEs, papers) and connects it to the open role.
"""

from __future__ import annotations

from datetime import datetime, timezone

import shared.config as config
from shared.failures import ApiBudgetExhaustedError, is_api_budget_exhausted_error
from shared.llm_clients import opus_llm
from github.schemas import GitHubCandidate
from shared.schemas import OpusDecision


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

OUTREACH_SYSTEM = """\
You write short recruiting outreach messages on behalf of a technical recruiter.

Rules:
- Open with a specific reference to the candidate's work. Name a repo, describe
  what it does, or reference a README detail, paper, or website finding. Never
  open with a generic greeting or compliment.
- Connect their work to the role: explain *why* their specific experience maps
  to the team's needs. Be concrete ("your data-pipeline reliability work maps to our
  platform team's needs"), not generic ("your skills are a great fit").
- Keep the message body under 150 words.
- No flattery, buzzwords, or generic recruiting language. No "rockstar",
  "passionate", "exciting opportunity", "thrilled", or similar filler.
- If the candidate has a personal website or papers, reference something
  specific from them.
- Write in first person as the recruiter sending the message. Professional but
  human — the recruiter should sound like a real person, not a template.
- Include a subject line suitable for email or LinkedIn InMail.

Return valid JSON with exactly these keys:
{
  "subject_line": "...",
  "message": "...",
  "repo_referenced": "name of the primary repo you referenced, or empty string",
  "capability_hook": "one-sentence summary of why this person fits the role"
}
"""

OUTREACH_USER = """\
Generate a personalized outreach message for this candidate.

ROLE
- Title: {role_title}
- Summary: {role_summary}

CANDIDATE
- Name: {candidate_name}
- GitHub username: {candidate_username}

EVALUATION RESULT
- Capability area matched: {capability_area_matched}
- Evidence that triggered the save: {evaluation_evidence}
- Confidence: {evaluation_confidence}
- Evaluation summary: {evaluation_summary}

TOP REPOSITORIES (non-fork, by stars)
{repo_highlights}

README EXCERPTS
{readme_excerpts}

WEBSITE / PAPERS
{website_papers}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_outreach(
    candidate: GitHubCandidate,
    brief,
    eval_result: OpusDecision,
    max_retries: int = 2,
) -> dict:
    """Generate stored outreach copy for a saved GitHub candidate.

    Parameters
    ----------
    candidate : GitHubCandidate
        The enriched candidate record.
    brief : brief_schema.Brief or brief_loader.Brief
        The hiring brief (must expose ``role_title`` and ``role_summary``).
    eval_result : OpusDecision
        The Opus evaluation decision that triggered the save.
    max_retries : int
        Number of attempts before giving up.

    Returns
    -------
    dict
        Keys: username, subject_line, message, repo_referenced,
        capability_hook, generated_at.  Returns an empty dict on failure.
    """
    for attempt in range(max_retries):
        try:
            result = _build_and_call(candidate, brief, eval_result)
            if result and result.get("message"):
                return result
        except Exception as e:
            if is_api_budget_exhausted_error(e):
                raise ApiBudgetExhaustedError(str(e)) from e
            if attempt == max_retries - 1:
                print(f"    [OUTREACH] Failed after {max_retries} attempts for {candidate.user.username}: {e}")
    return {}


def _build_and_call(
    candidate: GitHubCandidate,
    brief,
    eval_result: OpusDecision,
) -> dict:
    """Build outreach prompt and call Opus. Returns result dict or empty dict."""
    # --- Extract repo highlights (top 3 non-fork by stars) -----------
    non_fork_repos = sorted(
        [r for r in candidate.top_repos if not r.is_fork],
        key=lambda r: r.stars,
        reverse=True,
    )[:3]

    if non_fork_repos:
        repo_lines = []
        for r in non_fork_repos:
            topics = f" | topics: {', '.join(r.topics)}" if r.topics else ""
            repo_lines.append(
                f"- {r.name} ({r.language}, {r.stars}\u2605): {r.description}{topics}"
            )
        repo_highlights = "\n".join(repo_lines)
    else:
        repo_highlights = "(no non-fork repositories available)"

    # --- README excerpts ---------------------------------------------
    readme_text = getattr(candidate, "readme_text", "") or ""
    repo_readmes = getattr(candidate, "repo_readmes", {}) or {}

    readme_parts = []
    if readme_text:
        readme_parts.append(f"Profile README:\n{readme_text[:800]}")
    if repo_readmes:
        for repo_name, content in list(repo_readmes.items())[:3]:
            readme_parts.append(f"README for {repo_name}:\n{content[:500]}")
    readme_excerpts = "\n\n".join(readme_parts) if readme_parts else "(none available)"

    # --- Papers and website ------------------------------------------
    paper_titles = getattr(candidate, "paper_titles", []) or []
    website_text = getattr(candidate, "website_text", "") or ""

    website_parts = []
    if paper_titles:
        website_parts.append("Papers: " + "; ".join(paper_titles[:5]))
    if website_text:
        website_parts.append(f"Website content:\n{website_text[:600]}")
    website_papers = "\n".join(website_parts) if website_parts else "(none discovered)"

    # --- Build the user prompt ---------------------------------------
    user_prompt = OUTREACH_USER.format(
        role_title=getattr(brief, "role_title", ""),
        role_summary=getattr(brief, "role_description", "") or getattr(brief, "role_summary", ""),
        candidate_name=candidate.user.name or candidate.user.username,
        candidate_username=candidate.user.username,
        capability_area_matched=eval_result.path,
        evaluation_evidence=eval_result.rationale,
        evaluation_confidence=eval_result.confidence,
        evaluation_summary=eval_result.rationale,
        repo_highlights=repo_highlights,
        readme_excerpts=readme_excerpts,
        website_papers=website_papers,
    )

    # --- Call Opus (reduced max_tokens to prevent rambling) ----------
    usage_context = {
        "stage": "github_outreach_generation",
        "source": "github",
        "username": candidate.user.username,
        "candidate_name": candidate.user.name or candidate.user.username,
        "role_title": getattr(brief, "role_title", ""),
        "evaluation_path": eval_result.path,
    }
    result = opus_llm(
        system_prompt=OUTREACH_SYSTEM,
        user_prompt=user_prompt,
        expect_json=True,
        max_tokens=2048,
        usage_context=usage_context,
        model_name=config.OUTREACH_MODEL_NAME,
    )

    return {
        "username": candidate.user.username,
        "subject_line": result.get("subject_line", ""),
        "message": result.get("message", ""),
        "repo_referenced": result.get("repo_referenced", ""),
        "capability_hook": result.get("capability_hook", ""),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
