"""F3 + F4: who the panel says this is, and which project the run may touch.

F3 — ``_panel_confirms_expected_name`` decided that a panel showed the expected
person by (a) treating ANY one-codepoint token as a truncated initial and (b)
scanning a free-text window of the panel's first twelve tokens. Both are
identity claims the source text never made: an unmarked lone letter is a whole
name in a caseless script, and a name found in the panel's BODY copy is not the
panel's name. Each produced a ``True`` that committed a succeeded /
already-present receipt with no save click.

The truth table below is derived from the INPUT SPACE the matcher accepts —
script (cased vs caseless, full-width), abbreviation marker (absent, period,
ellipsis), token-count relation, separators (space, hyphen, particles),
position of the expected name inside the panel text, and panel readability —
not from the examples in any brief.

F4 — the Recruiter tab bind compared URL SHAPE, never project identity, so once
the brief's own tab went away the next-best Recruiter tab won and the owner's
search was typed into another project's sidebar. Resume then accepted any
``/talent/hire/`` page as "we're on a search page" and re-entered the owner's
Boolean there.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from linkedin.browser import LinkedInBrowser, _panel_confirms_expected_name
from shared.governor import UNGOVERNED_FOR_TESTS
from shared.schemas import Progress, SearchString

PROJECT_ID = "test-project"
PROJECT_URL = (
    f"https://www.linkedin.com/talent/hire/{PROJECT_ID}/discover/recruiterSearch"
)
FOREIGN_URL = "https://www.linkedin.com/talent/hire/999/discover/recruiterSearch"
PROJECTLESS_URL = "https://www.linkedin.com/talent/search"


# ---------------------------------------------------------------------------
# F3 — the name matcher, driven through the real production reader
# ---------------------------------------------------------------------------

# (panel_text, expected_name, confirms)
#
# Read every row as: "the panel is showing a 'Change stage' button — is it this
# person's?" A True that is wrong commits a save receipt nobody earned; a False
# that is wrong refuses a save that would have succeeded. False is the safe
# error, so every uncertain shape below resolves False.
_NAME_TRUTH_TABLE = [
    # -- caseless scripts: shortness is not an abbreviation -------------------
    ("王 伟\nEngineer", "王小明", False),
    ("王小明\nEngineer", "王小明", True),
    ("王小…\nEngineer", "王小明", True),
    ("王小明 · 2nd\nEngineer", "王小明", True),
    ("王小明\nEngineer", "明小王", False),
    ("김 민준\nEngineer", "김민준", False),
    # -- the expected name sitting in BODY copy, not in the name -------------
    ("Ann Lin\nPrincipal at Ann Li Consulting", "Ann Li", False),
    ("Ann Lin · 2nd · Ann Li Consulting", "Ann Li", False),
    ("Ann Lin\nEngineer", "Ann Li", False),
    ("Ann Li\nEngineer", "Ann Lin", False),
    ("John Smithson\nEngineer", "John Smith", False),
    # -- abbreviation markers: period and ellipsis only ----------------------
    ("Alexander Smith\nEngineer", "A Smith", False),
    ("A Smith\nEngineer", "Alexander Smith", False),
    ("Alexander Smith\nEngineer", "A. Smith", True),
    ("A. Smith\nEngineer", "Alexander Smith", True),
    ("Biao Zhang\nEngineer", "Biao Z.", True),
    ("Biao Z.\nEngineer", "Biao Zhang", True),
    ("Biao Zh…\nEngineer", "Biao Zhang", True),
    ("Biao Zh...\nEngineer", "Biao Zhang", True),
    ("Biao Zhang\nEngineer", "Biao Zh…", True),
    ("Biao Zhang\nEngineer", "Biao Zh", False),
    # a period after a MULTI-letter token is punctuation, not an abbreviation
    ("John Smith Jrx\nEngineer", "John Smith Jr.", False),
    # -- diacritics and width fold to the same letters -----------------------
    ("José García\nEngineer", "Jose Garcia", True),
    ("Jose Garcia\nEngineer", "José García", True),
    ("Ada Lovelace\nEngineer", "Ada Lovelace", True),
    ("Ａｎｎ Ｌｉｎ\nEngineer", "Ann Lin", True),
    ("Ａｎｎ Ｌｉｎ\nEngineer", "Ann Li", False),
    ("Biao Ｚ．\nEngineer", "Biao Zhang", True),
    # -- cased non-Latin scripts DO abbreviate -------------------------------
    ("Борис Ельцин\nEngineer", "Борис Е.", True),
    ("Борис Ельцин\nEngineer", "Борис Е", False),
    # -- hyphenated surnames: a hyphen is a separator, not a letter ----------
    ("Ana García-López\nEngineer", "Ana Garcia Lopez", True),
    ("Ana Garcia Lopez\nEngineer", "Ana García-López", True),
    ("Anne-Marie Smith\nEngineer", "Anne Marie Smith", True),
    ("Ana Garcialopez\nEngineer", "Ana Garcia-Lopez", False),
    # -- name particles: present on both sides or not at all -----------------
    ("Jan van der Berg\nEngineer", "Jan van der Berg", True),
    ("Jan Van Der Berg\nEngineer", "Jan van der Berg", True),
    ("Jan Berg\nEngineer", "Jan van der Berg", False),
    ("Omar al-Rashid\nEngineer", "Omar Al Rashid", True),
    ("Maria de la Cruz\nEngineer", "Maria de la Cruz", True),
    ("Maria dela Cruz\nEngineer", "Maria de la Cruz", False),
    # -- suffixes ------------------------------------------------------------
    ("John Smith Jr.\nEngineer", "John Smith Jr.", True),
    ("John Smith Jr.\nEngineer", "John Smith", True),
    ("John Smith\nEngineer", "John Smith Jr.", False),
    ("Henry Ford III\nEngineer", "Henry Ford III", True),
    ("Henry Ford II\nEngineer", "Henry Ford III", False),
    ("Henry Ford III\nEngineer", "Henry Ford II", False),
    # -- middle names on one side only ---------------------------------------
    ("Ada Byron Lovelace\nEngineer", "Ada Byron Lovelace", True),
    ("Ada Byron Lovelace\nEngineer", "Ada Lovelace", False),
    ("Ada Lovelace\nEngineer", "Ada Byron Lovelace", False),
    ("Ada Byron Lovelace\nEngineer", "Ada B. Lovelace", True),
    # -- single-token names --------------------------------------------------
    ("Madonna\nSinger", "Madonna", True),
    ("Ann\nEngineer", "Ann Lin", False),
    # KNOWN BOUND, asserted so it stays visible: one token in order is the
    # whole evidence a single-token expected name can offer, so it confirms
    # against a name line that BEGINS with it. LinkedIn abbreviates cards to
    # "First L." (a marked initial), never to a bare first name, so in
    # production a single-token expected name is a real mononym.
    ("Ann Lin\nEngineer", "Ann", True),
    # -- panel shape ---------------------------------------------------------
    ("Ada Lovelace", "Ada Lovelace", True),
    ("\n\n   \nAda Lovelace\nEngineer", "Ada Lovelace", True),
    ("···\nAda Lovelace\nEngineer", "Ada Lovelace", True),
    # KNOWN BOUND of reading the first rendered line: a panel that leads with
    # anything but the name refuses. Fail-closed, and the direction we want
    # until a live DOM capture pins a real name selector.
    ("Engineer\nAda Lovelace", "Ada Lovelace", False),
    ("", "Ada Lovelace", False),
]


@pytest.mark.parametrize(
    ("panel_text", "expected_name", "confirms"), _NAME_TRUTH_TABLE
)
def test_panel_name_matcher_truth_table(panel_text, expected_name, confirms):
    assert _panel_confirms_expected_name(panel_text, expected_name) is confirms


def _browser_reading_panel(panel_text, *, change_stage_visible=True):
    """A LinkedInBrowser whose ONLY fake is the DOM: the real is_already_saved runs."""
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    change_stage = MagicMock()
    change_stage.is_visible = AsyncMock(return_value=change_stage_visible)
    panel = MagicMock()
    panel.inner_text = AsyncMock(
        side_effect=panel_text if isinstance(panel_text, Exception) else None,
        return_value=None if isinstance(panel_text, Exception) else panel_text,
    )
    page = MagicMock()

    def locate(selector):
        located = MagicMock()
        located.first = panel if "profile__main-container" in selector else change_stage
        return located

    page.locator.side_effect = locate
    browser._page = page
    return browser


@pytest.mark.parametrize(
    ("panel_text", "expected_name", "confirms"), _NAME_TRUTH_TABLE
)
def test_is_already_saved_matches_the_truth_table(panel_text, expected_name, confirms):
    """The production reader, not the helper: same table, through the DOM path."""
    browser = _browser_reading_panel(panel_text)
    assert asyncio.run(
        browser.is_already_saved(expected_name=expected_name)
    ) is confirms


@pytest.mark.parametrize(
    ("panel_text", "expected_name"),
    [
        (RuntimeError("panel unreadable"), "Ada Lovelace"),
        ("Ada Lovelace\nEngineer", "···"),
        ("Ada Lovelace\nEngineer", None),
        ("Ada Lovelace\nEngineer", ""),
    ],
    ids=["unreadable-panel", "nameless-expected", "no-expected-name", "empty-expected"],
)
def test_is_already_saved_falls_open_when_there_is_nothing_to_compare(
    panel_text, expected_name
):
    """A read that did not happen is not evidence of a mismatch — unchanged."""
    browser = _browser_reading_panel(panel_text)
    assert asyncio.run(browser.is_already_saved(expected_name=expected_name)) is True


def test_is_already_saved_reports_not_saved_when_the_button_is_absent():
    browser = _browser_reading_panel(
        "Ada Lovelace\nEngineer", change_stage_visible=False
    )
    assert asyncio.run(browser.is_already_saved(expected_name="Ada Lovelace")) is False


# ---------------------------------------------------------------------------
# F4 — Recruiter tab binding is a project decision
# ---------------------------------------------------------------------------


def _tab(url, *, stable=True):
    page = MagicMock()
    page.url = url
    if stable:
        page.wait_for_load_state = AsyncMock()
    else:
        page.wait_for_load_state = AsyncMock(
            side_effect=RuntimeError("target crashed")
        )
    return page


def _browser_with_tabs(pages, *, required_project=PROJECT_ID, current=None):
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser.set_required_project_id(required_project)
    ctx = MagicMock()
    ctx.pages = list(pages)
    browser._browser = MagicMock()
    browser._browser.contexts = [ctx]
    browser._input_backend = MagicMock()
    browser._input_backend.initialize = AsyncMock()
    browser._page = current if current is not None else _tab("about:blank")
    return browser


def test_require_recruiter_tab_refuses_a_foreign_project_tab():
    """Only another project's Recruiter tab is open. Binding it types the
    owner's search into the wrong pipeline, and nothing navigates afterwards to
    undo that — the caller operates the tab immediately."""
    foreign = _tab(FOREIGN_URL)
    browser = _browser_with_tabs([foreign])

    with pytest.raises(RuntimeError, match="LinkedIn Recruiter is required"):
        asyncio.run(browser.require_recruiter_tab())

    assert browser._page is not foreign
    browser._input_backend.initialize.assert_not_awaited()


def test_require_recruiter_tab_refuses_a_projectless_tab():
    """F1's asymmetry, applied to the bind: a page that names no project proves
    nothing, and unverified is a mismatch when the brief pins a project."""
    projectless = _tab(PROJECTLESS_URL)
    browser = _browser_with_tabs([projectless])

    with pytest.raises(RuntimeError, match="LinkedIn Recruiter is required"):
        asyncio.run(browser.require_recruiter_tab())

    assert browser._page is not projectless


def test_require_recruiter_tab_binds_the_brief_project_tab_past_a_foreign_one():
    owned = _tab(PROJECT_URL)
    browser = _browser_with_tabs([_tab(FOREIGN_URL), owned])

    asyncio.run(browser.require_recruiter_tab())

    assert browser._page is owned
    assert browser._project_id == PROJECT_ID


def test_require_recruiter_tab_is_unchanged_when_the_brief_pins_no_project():
    """No pin, nothing to violate: shape scoring alone still binds."""
    foreign = _tab(FOREIGN_URL)
    browser = _browser_with_tabs([foreign], required_project=None)

    asyncio.run(browser.require_recruiter_tab())

    assert browser._page is foreign


def test_connect_still_adopts_a_foreign_tab_for_run_start_to_correct():
    """Deliberately NOT symmetric with require_recruiter_tab.

    connect() is followed by run-start's project-aware navigation, so adopting
    a foreign tab there costs nothing and refusing would strand a run whose page
    was about to be fixed. Pinned here so the bind precondition stays scoped to
    the callers that operate the tab immediately.
    """
    foreign = _tab(FOREIGN_URL)
    browser = _browser_with_tabs([foreign])
    browser._page_is_live = AsyncMock(return_value=True)

    asyncio.run(browser.connect())

    assert browser._page is foreign


def test_enter_search_string_never_types_into_a_foreign_project_sidebar():
    """The production caller. `enter_search_string` runs require_recruiter_tab
    first and then drives the sidebar of whatever it bound."""
    foreign = _tab(FOREIGN_URL)
    browser = _browser_with_tabs([foreign])
    browser.go_back_to_results = AsyncMock()

    with pytest.raises(RuntimeError, match="LinkedIn Recruiter is required"):
        asyncio.run(browser.enter_search_string("(python OR golang)"))

    foreign.locator.assert_not_called()
    foreign.fill.assert_not_called()
    foreign.click.assert_not_called()


# ---------------------------------------------------------------------------
# F4 — resume must not accept "some project's Recruiter page" as its own
# ---------------------------------------------------------------------------


def _pipeline(td, *, project_id=PROJECT_ID):
    from linkedin.orchestrator import Pipeline

    with patch("linkedin.orchestrator.load_brief") as mock_brief, patch(
        "linkedin.orchestrator.init_judger"
    ), patch("linkedin.orchestrator.LinkedInBrowser"):
        brief = MagicMock()
        brief.id = "test"
        brief.linkedin_project_id = project_id
        brief.has_v2_schema = False
        brief.employer_blacklist = []
        brief.permanent_filters = {}
        brief.needs_preflight.return_value = False
        mock_brief.return_value = brief
        brief_path = Path(td) / "brief.json"
        brief_path.write_text('{"id": "test"}')
        return Pipeline(brief_path=str(brief_path), output_dir=td)


def _wire_run(p, *, result_counts):
    """The `_wire_real_run_full_page_path` stub set, for a REAL browser object.

    Everything that needs a live DOM is stubbed; the bind chain
    (enter_search_string -> require_recruiter_tab -> _bind_existing_recruiter_page)
    is left real, because that chain is what is under test.
    """
    p._run_health_summary = MagicMock(return_value={"green_but_useless": False})
    p._enrich_run_snapshot = MagicMock()
    p.browser.connect = AsyncMock()
    p.browser.disconnect = AsyncMock()
    p.browser.navigate_to_search = AsyncMock()
    p.browser.go_back_to_results = AsyncMock()
    p.browser.get_results_count = AsyncMock(side_effect=list(result_counts))
    p.browser.get_results_count_text = AsyncMock(
        side_effect=[str(value) for value in result_counts]
    )
    p.browser.go_to_next_page = AsyncMock(return_value=True)
    p._ensure_browser_healthy = AsyncMock()
    p._verify_session_geography_chips = AsyncMock()
    p._apply_session_location_filter = AsyncMock()
    p._enforce_constraint_manifest = MagicMock()
    p._load_candidate_history = MagicMock()
    p._load_search_memory = MagicMock()
    p._evaluate_variant_lifecycle = MagicMock(return_value=None)
    p._plan_variant_experiments = AsyncMock()
    p._print_session_summary = MagicMock()
    p._print_summary = MagicMock()
    p._generate_run_report = MagicMock()
    p._session_expired = MagicMock()
    p._session_expired.is_set.return_value = False


def test_resume_navigates_off_a_foreign_projects_recruiter_page():
    """`/talent/hire/` present was the whole test, so resume re-entered the
    owner's Boolean on project 999's search page."""
    with tempfile.TemporaryDirectory() as td:
        p = _pipeline(td)
        owner = SearchString(
            id=1, name="owner", boolean="one", status="in_progress", pages_reviewed=1
        )
        progress = Progress(brief_name="test", strings=[owner])
        _wire_run(p, result_counts=[0])
        p.browser.page = MagicMock(url=FOREIGN_URL)
        p.browser.enter_search_string = AsyncMock()

        asyncio.run(p._process_string(owner, progress))

        p.browser.navigate_to_search.assert_awaited_once_with(PROJECT_URL)


def test_resume_stays_put_on_the_briefs_own_recruiter_page():
    with tempfile.TemporaryDirectory() as td:
        p = _pipeline(td)
        owner = SearchString(
            id=1, name="owner", boolean="one", status="in_progress", pages_reviewed=1
        )
        progress = Progress(brief_name="test", strings=[owner])
        _wire_run(p, result_counts=[0])
        p.browser.page = MagicMock(url=PROJECT_URL)
        p.browser.enter_search_string = AsyncMock()

        asyncio.run(p._process_string(owner, progress))

        p.browser.navigate_to_search.assert_not_awaited()


def test_resume_with_an_owned_pending_review_never_touches_a_foreign_project():
    """ACCEPTANCE (F4).

    Run starts with both the brief's Recruiter tab and another project's open,
    and with a full review already owed on the owner string. The brief's tab
    then becomes unusable — a crashed renderer, which is exactly the condition
    `_bind_existing_recruiter_page` skips as "unstable". The only Recruiter tab
    left belongs to project 999.

    Nothing may be reviewed, searched or saved there. The run stops with the
    owner still in_progress, the later string still queued, and the pending
    review still owed.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _pipeline(td)
        owner = SearchString(
            id=1, name="owner", boolean="one", status="in_progress", pages_reviewed=1
        )
        later = SearchString(id=2, name="later", boolean="two", status="queued")
        progress = Progress(brief_name="test", strings=[owner, later])
        progress.save(str(p.progress_path))
        p._load_or_create_progress = MagicMock(return_value=progress)

        owned_tab = _tab(PROJECT_URL)
        foreign_tab = _tab(FOREIGN_URL)
        browser = _browser_with_tabs([owned_tab, foreign_tab], current=owned_tab)
        browser.go_back_to_results = AsyncMock()
        p.browser = browser
        p._ensure_services()
        _wire_run(p, result_counts=[25])

        # A full review already owed on the owner string, carried across the
        # process boundary the way resume carries it.
        pending = MagicMock()
        p._resume_pending_full_decisions = {"owed": "FACIAL_YES"}
        p._resume_pending_full_owner_ids = {"owed": owner.id}
        p._resume_pending_full_snippets = {"owed": pending}

        # The brief's own tab dies after run start; the health check is where a
        # live run notices.
        def _kill_owned_tab():
            owned_tab.wait_for_load_state = AsyncMock(
                side_effect=RuntimeError("target crashed")
            )

        p._ensure_browser_healthy = AsyncMock(side_effect=lambda: _kill_owned_tab())

        with pytest.raises(RuntimeError, match="LinkedIn Recruiter is required"):
            asyncio.run(p.run_full(resume=True))

        # Nothing happened on project 999's page.
        assert browser._page is not foreign_tab
        foreign_tab.locator.assert_not_called()
        foreign_tab.fill.assert_not_called()
        foreign_tab.click.assert_not_called()
        foreign_tab.evaluate.assert_not_called()
        # The obligation is still owed, and the queue is untouched.
        assert p._resume_pending_full_snippets == {"owed": pending}
        saved = json.loads(Path(td, "progress.json").read_text())
        assert [item["status"] for item in saved["strings"]] == [
            "in_progress",
            "queued",
        ]
