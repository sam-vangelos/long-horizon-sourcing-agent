"""P3.2 — exhaustion epochs (no brief fixture dependency)."""

from shared.schemas import SearchString
from shared.search_memory import update_search_memory

def test_search_memory_epoch_archives_on_brief_revision():
    """P3.2: a family exhausted under brief v1 starts fresh under v2 with the
    v1 counters archived as history; unchanged briefs never archive."""

    memory = {}
    v1_strings = [
        SearchString(
            id=i,
            name=f"v1 string {i}",
            boolean='("Goldman Sachs") AND ("LLM")',
            pages_reviewed=2,
            candidates_count=10,
            duplicates_count=12,
            family_key="canonical_bank_company_first",
            novelty_bucket="canonical",
            domain_lane="capital_markets",
        )
        for i in (1, 2)
    ]
    for s in v1_strings:
        setattr(s, "brief_epoch", "hash-v1")
    memory = update_search_memory(memory, "p", v1_strings)
    family = memory["families"]["canonical_bank_company_first"]
    assert family["status"] == "exhausted"
    assert family["brief_epoch"] == "hash-v1"

    # Unchanged brief: nothing archived, still exhausted.
    same_epoch = SearchString(
        id=3, name="v1 string 3", boolean='("JPMorgan") AND ("LLM")',
        candidates_count=5, duplicates_count=6,
        family_key="canonical_bank_company_first", novelty_bucket="canonical",
    )
    setattr(same_epoch, "brief_epoch", "hash-v1")
    memory = update_search_memory(memory, "p", [same_epoch])
    family = memory["families"]["canonical_bank_company_first"]
    assert family["archived_epochs"] == []
    assert family["status"] == "exhausted"

    # Brief revised: counters archive, family is fresh (active) under v2.
    v2_string = SearchString(
        id=4, name="v2 string", boolean='("Goldman Sachs") AND ("agents")',
        candidates_count=8, duplicates_count=1,
        family_key="canonical_bank_company_first", novelty_bucket="canonical",
    )
    setattr(v2_string, "brief_epoch", "hash-v2")
    memory = update_search_memory(memory, "p", [v2_string])
    family = memory["families"]["canonical_bank_company_first"]
    assert family["brief_epoch"] == "hash-v2"
    assert family["status"] == "active"
    assert family["strings_seen"] == 1
    assert len(family["archived_epochs"]) == 1
    archived = family["archived_epochs"][0]
    assert archived["brief_epoch"] == "hash-v1"
    assert archived["status"] == "exhausted"
    assert archived["strings_seen"] == 3


def test_search_memory_prior_session_suppressions_do_not_feed_exhaustion():
    """P3.2: only same-epoch overlap counts toward the 0.40 dup-rate rule."""

    memory = {}
    strings = [
        SearchString(
            id=i,
            name=f"s{i}",
            boolean='("Goldman Sachs") AND ("LLM")',
            candidates_count=10,
            duplicates_count=12,
            suppressed_prior_session_count=11,
            family_key="fam",
            novelty_bucket="edge_case",
        )
        for i in (1, 2)
    ]
    memory = update_search_memory(memory, "p", strings)
    family = memory["families"]["fam"]
    # 2 within-run duplicates vs 20 candidates — nowhere near exhausted.
    assert family["status"] == "active"
    assert family["suppressed_prior_session"] == 22
