from linkedin.candidate_review import CandidateReviewer
"""Regression lock for CLO-175: the employer blacklist must key on the current
employer (CLO-115 ruling). A headline mentioning a blacklisted name as a tool,
certification, or skill must never exclude; a headline naming it as the current
employer ("@ X" / "at X") must. The must-NOT-match cases below are real
headlines of candidates falsely excluded in the PAE campaign (29 of its 31
blacklist exclusions were this class) plus the Brazil run's Lucas Fernandes.
"""

from linkedin.evaluation_loop import EvaluationLoop

from types import SimpleNamespace

import pytest

from linkedin.orchestrator import Pipeline

BLACKLIST = ["Babbage", "OpenAI", "Anthropic"]


def snippet(headline: str = "", current_company: str = "") -> SimpleNamespace:
    return SimpleNamespace(headline=headline, current_company=current_company)


def match(headline: str = "", current_company: str = ""):
    return CandidateReviewer._employer_blacklist_match(
        snippet(headline=headline, current_company=current_company), BLACKLIST
    )


# ---------------------------------------------------------------- tool mentions
# Verbatim headlines from PAE's runtime DB (falsely excluded pre-fix).
@pytest.mark.parametrize(
    "headline",
    [
        # Akash M
        "Sr. Generative AI Engineer | Data Scientist | LangChain | RAG | OpenAI "
        "| Azure OpenAI | LLMs | Python | 11+ Years",
        # Nidhi V
        "AI/ML Engineer | Generative & Agentic AI (LangChain, LangGraph, RAG, "
        "Anthropic Claude, GPT-4o) | AWS & Azure | MLOps",
        # Inderjit Singh
        "Principal AI Engineer | Agentic AI Architect | GenAI Solutions | "
        "LangGraph · Azure OpenAI · RAG · MCP · Python",
        # Lucas Fernandes (Brazil run)
        "Software Engineer at CI&T · Full-Stack · Node.js · TypeScript · "
        "Svelte · Go · Anthropic CCAR-F · AWS · GCP",
    ],
)
def test_headline_tool_mention_never_excludes(headline):
    assert match(headline=headline) is None


# ------------------------------------------------------------ employer context
@pytest.mark.parametrize(
    "headline, blocked",
    [
        ("MTS @ OpenAI", "OpenAI"),
        ("MTS @OpenAI", "OpenAI"),
        ("Research Engineer at Anthropic", "Anthropic"),
        ("Security Engineer at Babbage | ex-Google", "Babbage"),
    ],
)
def test_headline_current_employer_excludes(headline, blocked):
    got = match(headline=headline)
    assert got is not None and got[0] == blocked and got[1] == "headline"


# ------------------------------------------------------------------ alumni stay
@pytest.mark.parametrize(
    "headline",
    [
        "ML Engineer | ex-OpenAI | building agents",
        "Previously at Anthropic, now founding something new",
        "Staff Engineer, formerly at OpenAI",
    ],
)
def test_headline_alumni_never_excludes(headline):
    assert match(headline=headline) is None


# ------------------------------------------------------------- current_company
def test_current_company_excludes_unchanged():
    got = match(current_company="Anthropic")
    assert got is not None and got[0] == "Anthropic" and got[1] == "current_company"


def test_current_company_word_boundary_holds():
    # "Babbage" must not fire inside another word.
    assert match(current_company="Centurion Systems") is None


def test_school_escape_retained():
    assert match(current_company="Babbage School of Engineering") is None


def test_empty_blacklist_is_none():
    assert (
        CandidateReviewer._employer_blacklist_match(snippet(headline="MTS @ OpenAI"), [])
        is None
    )


# ------------------------------------------------ alumni phrasings (CLO-330)
# A past marker anywhere in the clause before "at X" / "@ X" makes X a past
# employer. Live client slates read each of these as current employment.
ALUMNI_BLACKLIST = ["Meta", "Mercor"]


def shared_match(headline: str = "", current_company: str = ""):
    from shared.employer_blacklist import match_current_employer

    return match_current_employer(current_company, headline, ALUMNI_BLACKLIST)


@pytest.mark.parametrize(
    "headline",
    [
        "Former engineer at Meta",
        "Previously Data Lead at Meta",
        "Ex-SWE @ Meta",
        "Formerly at Mercor",
        "Prev. Account Lead at Mercor",
        "Past Head of Growth at Mercor",
        "Late of FAIR at Meta",
        "Alum: Account Lead @ Mercor",
        # The marker opens a clause longer than the lookback window.
        "Previously Senior Staff Machine Learning Infrastructure Engineer at Meta",
        "Data Scientist | Formerly Principal Data Platform Architect Lead @ Mercor",
    ],
)
def test_alumni_phrasing_is_not_current_employment(headline):
    assert shared_match(headline=headline) is None


@pytest.mark.parametrize(
    "headline, blocked",
    [
        ("Head of Data at Mercor", "Mercor"),
        ("Account Lead @ Mercor", "Mercor"),
        ("Data Lead at Mercor (ex-Google)", "Mercor"),
        ("Strategic Projects at Mercor, previously Handshake AI", "Mercor"),
        # A marker in an earlier clause belongs to that clause.
        ("ex-Google | Head of Data at Mercor", "Mercor"),
        ("Former SWE at Google; Data Lead at Mercor", "Mercor"),
        ("Previously Google - Data Lead at Mercor", "Mercor"),
        # "now" / "currently" end the past stretch inside one clause.
        ("Ex-Google now Head of Data at Mercor", "Mercor"),
        # A marker is spent on the first employer after it.
        ("Former engineer at Google and Head of Data at Mercor", "Mercor"),
    ],
)
def test_current_phrasing_still_excludes(headline, blocked):
    got = shared_match(headline=headline)
    assert got is not None and got[0] == blocked and got[1] == "headline"


def test_current_company_with_segments_still_excludes():
    got = shared_match(current_company="Mercor | Strategic Projects")
    assert got is not None and got[0] == "Mercor" and got[1] == "current_company"


def test_a_word_cut_by_the_lookback_window_is_not_a_marker():
    # 32 characters back from "at" lands inside "Transformer": "former ...".
    headline = "Transformer Inference Platform Leads at Meta"
    got = shared_match(headline=headline)
    assert got is not None and got[0] == "Meta"
