"""LinkedIn Recruiter's redesigned search page, against synthetic offline DOM.

The redesign opens the filter rail in a SUMMARY state (summary pills, an Edit
button, qualifications) on every load; the old facet rail only exists inside
the edit panel that Edit opens. It also moved the result count, added a
"Spotlights" carousel whose Previous/Next buttons are not pagination, put a
scope menu on every company chip, and added qualifications that silently
narrow a search. Every fixture below is modelled on the selectors recorded in
docs/linkedin-recruiter-dom-map.md for that redesign.
"""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from linkedin.browser import FacetReconciliationError, LinkedInBrowser
from shared.governor import UNGOVERNED_FOR_TESTS

SEARCH_URL = (
    "https://www.linkedin.com/talent/hire/9000000003/discover/recruiterSearch"
    "?searchContextId=ctx-1"
)


def _run_page(html: str, body, *, url: str | None = None, project_id: str | None = None):
    """Run ``body(browser, page)`` against ``html`` in an offline Chromium page.

    With ``url`` the document is served at that address through a route, so
    URL-reading code sees a real Recruiter search URL; nothing leaves the
    machine.
    """
    from rebrowser_playwright.async_api import async_playwright

    async def _main():
        pw = await async_playwright().start()
        try:
            chromium = await asyncio.wait_for(
                pw.chromium.launch(headless=True, args=["--single-process", "--no-zygote"]),
                timeout=30,
            )
            try:
                page = await chromium.new_page()
                if url:
                    async def _serve(route):
                        await route.fulfill(status=200, content_type="text/html", body=html)

                    await page.route("https://www.linkedin.com/**", _serve)
                    await page.goto(url, wait_until="domcontentloaded")
                else:
                    encoded = base64.b64encode(html.encode()).decode()
                    await page.goto(
                        f"data:text/html;base64,{encoded}", wait_until="domcontentloaded"
                    )
                browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
                browser._page = page
                if project_id:
                    browser.set_required_project_id(project_id)
                return await body(browser, page)
            finally:
                await chromium.close()
        finally:
            await pw.stop()

    return asyncio.run(asyncio.wait_for(_main(), timeout=120))


def _pill(value: str, *, label: str | None = None) -> str:
    shown = label if label is not None else value
    return (
        f'<li data-test-facet-pills-item><span class="facet-pill">'
        f"<span data-test-pill-label>{shown}</span>"
        f'<button data-test-pill-dismiss aria-label="Remove {shown}"></button>'
        f"</span></li>"
    )


_EDIT_PANEL = (
    '<div data-test-filter-edit-panel>'
    '<button data-test-filter-edit-panel-back type="button">Back</button>'
    '<div class="search-facet-wrapper facet-locations"><ul>'
    + _pill("United States")
    + "</ul></div>"
    '<div class="search-facet-wrapper facet-companies"><ul></ul></div>'
    '<div class="search-facet-wrapper facet-titles"><ul></ul></div>'
    '<div class="keywords-facet"><button aria-label="Edit Profile keywords or boolean">'
    "Profile keywords</button></div>"
    "</div>"
)

_RESULTS = (
    '<span data-test-profile-list-num-custom class="profile-list__header-info-text">'
    "916 results</span>"
    '<ol class="profile-list"><li><article class="profile-list-item">'
    '<div class="lockup__title"><a href="/talent/profile/p1">Card One</a></div>'
    "</article></li></ol>"
)

# The summary state as every page load renders it: no facet wrapper, no chip,
# no keyword control. Edit swaps the summary for the edit panel a beat later.
_SUMMARY_RAIL = f"""
<aside class="left-rail">
  <div id="summary">
    <ul data-test-filter-summary-pills>
      <li><span data-test-filter-summary-pill="BING_GEO_SWR">
        <span data-test-filter-summary-pill-label>United States</span></span></li>
      <li><span data-test-filter-summary-pill="companies">
        <span data-test-filter-summary-pill-label>AfterQuery</span>
        <span data-test-filter-summary-pill-count>+3</span></span></li>
    </ul>
    <button data-test-filter-summary-edit type="button">Edit</button>
    <ul id="qualifications">QUALIFICATIONS</ul>
  </div>
</aside>
{_RESULTS}
<script>
  window.editClicks = 0;
  window.removedQualifications = [];
  const rail = document.querySelector('aside.left-rail');
  rail.addEventListener('click', event => {{
    if (event.target.closest('button[data-test-filter-summary-edit]')) {{
      window.editClicks += 1;
      setTimeout(() => {{
        document.querySelector('#summary').style.display = 'none';
        if (!document.querySelector('[data-test-filter-edit-panel]')) {{
          rail.insertAdjacentHTML('beforeend', {_EDIT_PANEL!r});
        }} else {{
          document.querySelector('[data-test-filter-edit-panel]').style.display = '';
        }}
      }}, 150);
      return;
    }}
    if (event.target.closest('button[data-test-filter-edit-panel-back]')) {{
      document.querySelector('[data-test-filter-edit-panel]').remove();
      document.querySelector('#summary').style.display = '';
      return;
    }}
    const remove = event.target.closest('button[data-test-qualification-item-remove]');
    if (remove && !window.qualificationsStick) {{
      const row = remove.closest('li[data-test-qualifications-summary-row]');
      window.removedQualifications.push(
        row.querySelector('button[data-test-qualification-item-text]').textContent.trim());
      setTimeout(() => row.remove(), 80);
    }}
  }});
</script>
"""


def _summary_page(*qualifications: str, stick: bool = False) -> str:
    rows = "".join(
        "<li data-test-qualifications-summary-row><div data-test-qualification-item>"
        f"<button data-test-qualification-item-text>{text}</button>"
        '<button data-test-qualification-item-remove aria-label="Remove qualification">'
        "x</button></div></li>"
        for text in qualifications
    )
    html = _SUMMARY_RAIL.replace("QUALIFICATIONS", rows)
    if stick:
        html = html.replace("window.removedQualifications = [];",
                            "window.removedQualifications = []; window.qualificationsStick = true;")
    return html


_LEGACY_RAIL = f"""
<aside class="left-rail">
  <div class="search-facet-wrapper facet-locations"><ul>{_pill("United States")}</ul></div>
  <div class="search-facet-wrapper facet-companies"><ul>{_pill("Mercor")}</ul></div>
  <div class="search-facet-wrapper facet-titles"><ul></ul></div>
</aside>
<div class="search-query-summary__title">93 results</div>
"""


# ---------------------------------------------------------------------------
# a. the rail opens in SUMMARY; the readers open the edit panel first
# ---------------------------------------------------------------------------


def test_summary_rail_is_opened_before_the_chip_readers_read_it():
    async def _body(browser, page):
        assert await browser.read_applied_location_chips() == ["United States"]
        assert await browser.read_applied_company_chips() == []
        assert await page.evaluate("window.editClicks") == 1
        # Idempotent: a second read finds the panel open and clicks nothing.
        assert await browser.read_applied_title_chips() == []
        assert await page.evaluate("window.editClicks") == 1

    _run_page(_summary_page(), _body)


def test_summary_rail_no_longer_fails_the_session_geography_gate(tmp_path, monkeypatch):
    from linkedin.geography_gate import GeographyGateDeps, GeographyGateService
    from shared import config

    monkeypatch.setattr(config, "LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED", True)

    async def _body(browser, page):
        brief = SimpleNamespace(permanent_filters={"Location": "United States"})
        gate = GeographyGateService(
            GeographyGateDeps(
                get_browser=lambda: browser,
                get_brief_obj=lambda: brief,
                log_path=tmp_path / "run_log.jsonl",
                stats={},
            )
        )
        await gate._verify_session_geography_chips()
        assert await page.evaluate("window.editClicks") == 1

    _run_page(_summary_page(), _body)


def test_legacy_rail_makes_the_edit_panel_helper_a_no_op():
    async def _body(browser, page):
        browser._ghost_click_locator = AsyncMock(
            side_effect=AssertionError("the legacy rail has nothing to open")
        )
        await browser.ensure_filter_edit_panel_open()
        assert await browser.read_applied_location_chips() == ["United States"]
        assert await browser.read_applied_company_chips() == ["Mercor"]
        browser._ghost_click_locator.assert_not_awaited()

    _run_page(_LEGACY_RAIL, _body)


def test_an_edit_panel_that_never_opens_fails_closed():
    stuck = _summary_page().replace("window.editClicks += 1;", "window.editClicks += 1; return;")

    async def _body(browser, page):
        with pytest.raises(FacetReconciliationError) as raised:
            await browser.ensure_filter_edit_panel_open(timeout_ms=1500)
        assert raised.value.gate == "filter_edit_panel_unopened"
        # A chip reader treats the same failure as an unreadable rail.
        assert await browser.read_applied_location_chips() is None

    _run_page(stuck, _body)


# ---------------------------------------------------------------------------
# b. summary pills and qualification rows are not unattributed rail pills
# ---------------------------------------------------------------------------


_EDIT_RAIL_WITH_SUMMARY_AND_QUALIFICATION = f"""
<aside class="left-rail">
  <ul data-test-filter-summary-pills>
    <li data-test-facet-pills-item><span data-test-filter-summary-pill="companies">
      AfterQuery +3</span></li>
  </ul>
  <ul><li data-test-qualifications-summary-row>
    <div data-test-qualification-item><ul>
      <li data-test-facet-pills-item><button data-test-pill-dismiss
        aria-label="Remove Currently working at AfterQuery">x</button></li>
    </ul></div>
  </li></ul>
  <div class="custom-filters"><ul>
    <li data-test-facet-pills-item><button data-test-pill-dismiss
      aria-label="Remove Engineering Manager - LLM Delivery">x</button></li>
  </ul></div>
  {_EDIT_PANEL}
</aside>
"""


def test_summary_pills_and_qualifications_do_not_trip_the_unattributed_pill_check():
    async def _body(browser, page):
        assert await browser.read_applied_location_chips() == ["United States"]
        assert await browser.read_applied_company_chips() == []

    _run_page(_EDIT_RAIL_WITH_SUMMARY_AND_QUALIFICATION, _body)


def test_an_unattributed_pill_inside_the_edit_panel_still_fails_closed():
    html = _EDIT_RAIL_WITH_SUMMARY_AND_QUALIFICATION.replace(
        '<div class="keywords-facet">',
        '<ul><li data-test-facet-pills-item><button data-test-pill-dismiss '
        'aria-label="Remove Rotated">x</button></li></ul><div class="keywords-facet">',
        1,
    )

    async def _body(browser, page):
        assert await browser.read_applied_location_chips() is None

    _run_page(html, _body)


# ---------------------------------------------------------------------------
# c. the result count moved to span[data-test-profile-list-num-custom]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [("916 results", 916), ("1 result", 1), ("1K+ results", 1000), ("2,345 results", 2345)],
)
def test_result_count_reads_the_profile_list_header(text, expected):
    html = (
        '<span data-test-profile-list-num-custom class="profile-list__header-info-text">'
        f"{text}</span>"
    )

    async def _body(browser, page):
        assert await browser.get_results_count() == expected
        assert await browser._peek_results_count_text()

    _run_page(html, _body)


def test_result_count_keeps_the_old_summary_title_as_a_fallback():
    async def _body(browser, page):
        assert await browser.get_results_count() == 93

    _run_page(_LEGACY_RAIL, _body)


# ---------------------------------------------------------------------------
# d. the Spotlights carousel is not pagination
# ---------------------------------------------------------------------------


_CAROUSEL_AND_PAGINATION = """
<section class="spotlights">
  <button aria-label="Previous" disabled>&lsaquo;</button>
  <button aria-label="Next" onclick="window.clicks.push('carousel')">&rsaquo;</button>
</section>
<div class="mini-pagination">
  <a class="mini-pagination__quick-link" href="#" aria-label="Go to previous page 1"
     onclick="window.clicks.push('previous'); return false;">&lsaquo;</a>
  <a class="mini-pagination__quick-link" href="#" aria-label="Go to next page 3"
     onclick="window.clicks.push('pagination'); advance(); return false;">&rsaquo;</a>
</div>
<span data-test-profile-list-num-custom>916 results</span>
<ol class="profile-list"><li><article class="profile-list-item">
  <div class="lockup__title"><a id="first" href="/talent/profile/p26">Page Two First</a></div>
</article></li></ol>
<script>
  window.clicks = [];
  function advance() {
    setTimeout(() => { document.querySelector('#first').textContent = 'Page Three First'; }, 100);
  }
</script>
"""


def test_a_disabled_carousel_previous_is_not_proof_of_page_one():
    async def _body(browser, page):
        assert await browser.results_page_offset() == 25

    _run_page(
        _CAROUSEL_AND_PAGINATION,
        _body,
        url=SEARCH_URL + "&start=25&uiOrigin=PAGINATION",
    )


def test_page_one_is_read_from_the_url_not_the_carousel():
    async def _body(browser, page):
        assert await browser.results_page_offset() == 0

    _run_page(_CAROUSEL_AND_PAGINATION, _body, url=SEARCH_URL + "&uiOrigin=FACET_SEARCH")


def test_page_local_recovery_refuses_a_reanchor_that_did_not_land_on_page_one():
    """The carousel's disabled Previous used to prove page one on every page."""
    from linkedin.search_intelligence import bootstrap_experiment_state
    from linkedin.search_surface import SearchSurface
    from shared.schemas import SearchString

    owner = SearchString(id=7, name="Recorded", boolean="data AND evaluation", block="A")
    state = bootstrap_experiment_state(owner)

    async def _body(browser, page):
        browser.navigate_to_search = AsyncMock()
        surface = SearchSurface.__new__(SearchSurface)
        surface.context = SimpleNamespace(browser=browser, brief_obj=SimpleNamespace(raw={}))
        surface.allocator = SimpleNamespace(require_experiment=lambda _id: state)
        surface.geography = SimpleNamespace(
            reset_location_applied=MagicMock(),
            _apply_session_location_filter=AsyncMock(),
        )
        surface._get_project_url = lambda: SEARCH_URL
        surface._check_page_recovery_health = AsyncMock()
        surface._check_page_recovery_authority = AsyncMock()
        surface._apply_opening_search = AsyncMock()
        surface._probe_card_slot_count_after_hydration = AsyncMock(return_value=1)
        surface._recovery_top_identity = AsyncMock(return_value="p26")
        surface._advance_recovery_page = AsyncMock(return_value=True)
        surface._variant_id_for_search_string = lambda _owner: "root"
        surface._remember_recovery_page = AsyncMock()
        surface.recovery_top_identity = "p26"
        surface.last_opening_facet_receipt = None

        assert await surface._reanchor_recovery_page(owner, 1, "root") is False
        surface._remember_recovery_page.assert_not_awaited()

    _run_page(
        _CAROUSEL_AND_PAGINATION,
        _body,
        url=SEARCH_URL + "&start=25&uiOrigin=PAGINATION",
    )


def test_next_page_fallback_clicks_pagination_never_the_carousel():
    async def _body(browser, page):
        assert await browser.go_to_next_page() is True
        assert await page.evaluate("window.clicks") == ["pagination"]

    _run_page(_CAROUSEL_AND_PAGINATION, _body)


# ---------------------------------------------------------------------------
# e/f. company chip scope: read back, keep, and set CURRENT under an allowlist
# ---------------------------------------------------------------------------


def _scoped_company_rail(*names: str, labels: dict[str, str] | None = None,
                         scope_works: bool = True) -> str:
    labels = labels or {}
    pills = "".join(
        f'<li data-test-facet-pills-item data-company="{name}"><span class="facet-pill" '
        f'aria-label="Can have, {labels.get(name, name)}">'
        f"<span data-test-pill-label>{labels.get(name, name)}</span>"
        f'<button data-test-pill-actions-dropdown-trigger '
        f'aria-label="Dropdown menu for updating {name}">...</button>'
        f'<button data-test-pill-dismiss aria-label="Remove {labels.get(name, name)}">x</button>'
        "</span></li>"
        for name in names
    )
    actions = "".join(
        f'<div role="menuitemradio" data-test-pill-action="{action}" '
        f'aria-checked="{"true" if action in ("include", "CURRENT_OR_PAST") else "false"}">'
        f"{action}</div>"
        for action in ("must-have", "include", "exclude",
                       "CURRENT_OR_PAST", "CURRENT", "PAST", "PAST_NOT_CURRENT")
    )
    return f"""
<aside class="left-rail"><div data-test-filter-edit-panel>
  <div class="search-facet-wrapper facet-locations"><ul>{_pill("United States")}</ul></div>
  <div class="search-facet-wrapper facet-companies"><ul>{pills}</ul></div>
  <div class="search-facet-wrapper facet-titles"><ul></ul></div>
</div></aside>
<div id="pill-menu" role="menu" style="display:none">{actions}</div>
<script>
  window.facetInteractions = [];
  window.scopeWorks = {str(scope_works).lower()};
  const SUFFIX = {{CURRENT: ' (Current)', PAST: ' (Past)',
                   PAST_NOT_CURRENT: ' (Past not Current)', CURRENT_OR_PAST: ''}};
  const menu = document.querySelector('#pill-menu');
  let menuFor = null;
  document.addEventListener('click', event => {{
    const trigger = event.target.closest('button[data-test-pill-actions-dropdown-trigger]');
    if (trigger) {{
      menuFor = trigger.closest('li');
      menu.style.display = 'block';
      return;
    }}
    const action = event.target.closest('[data-test-pill-action]');
    if (action && menuFor) {{
      const scope = action.getAttribute('data-test-pill-action');
      const name = menuFor.getAttribute('data-company');
      window.facetInteractions.push(`scope:${{name}}:${{scope}}`);
      if (window.scopeWorks && scope in SUFFIX) {{
        const label = name + SUFFIX[scope];
        menuFor.querySelector('[data-test-pill-label]').textContent = label;
        menuFor.querySelector('.facet-pill').setAttribute('aria-label', `Can have, ${{label}}`);
        menuFor.querySelector('button[data-test-pill-dismiss]').setAttribute('aria-label', `Remove ${{label}}`);
      }}
      menu.style.display = 'none';
      menuFor = null;
      return;
    }}
    const dismiss = event.target.closest('button[data-test-pill-dismiss]');
    if (dismiss) {{
      const item = dismiss.closest('li[data-test-facet-pills-item]');
      window.facetInteractions.push(`remove:${{item.getAttribute('data-company') || 'location'}}`);
      item.remove();
    }}
  }});
</script>
"""


def test_split_facet_scope_reads_every_scope_suffix():
    from linkedin.browser import split_facet_scope

    assert split_facet_scope("AfterQuery (Current)") == ("AfterQuery", "CURRENT")
    assert split_facet_scope("AfterQuery (Past)") == ("AfterQuery", "PAST")
    assert split_facet_scope("AfterQuery (Past not Current)") == (
        "AfterQuery", "PAST_NOT_CURRENT")
    assert split_facet_scope("AfterQuery (Current or Past)") == (
        "AfterQuery", "CURRENT_OR_PAST")
    assert split_facet_scope("Scale AI") == ("Scale AI", None)


def test_a_current_scoped_chip_reads_back_as_its_company_with_the_scope_kept_apart():
    html = _scoped_company_rail("AfterQuery", labels={"AfterQuery": "AfterQuery (Current)"})

    async def _body(browser, page):
        assert await browser.read_applied_company_chips() == ["AfterQuery"]
        assert await browser.read_applied_company_chip_scopes() == {"AfterQuery": "CURRENT"}

    _run_page(html, _body)


def test_reconcile_keeps_a_current_scoped_chip_it_asked_for():
    html = _scoped_company_rail("AfterQuery", labels={"AfterQuery": "AfterQuery (Current)"})

    async def _body(browser, page):
        browser.apply_company_filter = AsyncMock(side_effect=AssertionError("retyped"))
        browser.apply_title_filter = AsyncMock(side_effect=AssertionError("retyped"))
        await browser.reconcile_facets(companies={"AfterQuery"}, titles=set())
        assert await page.evaluate("window.facetInteractions") == []
        assert browser.last_facet_reconciliation["after"]["companies"] == ["AfterQuery"]

    _run_page(html, _body)


def test_an_allowlist_brief_asks_for_current_company_scope():
    from linkedin.competitor_scope import company_facet_scope

    allowlist = {"facet_groups": [["Mercor"], ["AfterQuery", "Handshake"]]}
    assert company_facet_scope(SimpleNamespace(raw={"employer_allowlist": allowlist})) == "CURRENT"
    assert company_facet_scope(SimpleNamespace(raw={})) is None
    assert company_facet_scope(SimpleNamespace(raw={"employer_allowlist": {"bad": 1}})) is None


def test_reconcile_sets_current_on_every_company_chip_when_the_scope_is_pinned():
    html = _scoped_company_rail("AfterQuery", "Mercor")

    async def _body(browser, page):
        browser.set_company_facet_scope("CURRENT")
        await browser.reconcile_facets(companies={"AfterQuery", "Mercor"}, titles=set())
        assert await browser.read_applied_company_chip_scopes() == {
            "AfterQuery": "CURRENT",
            "Mercor": "CURRENT",
        }
        assert sorted(await page.evaluate("window.facetInteractions")) == [
            "scope:AfterQuery:CURRENT",
            "scope:Mercor:CURRENT",
        ]
        receipt = browser.last_facet_reconciliation
        assert receipt["failure_gate"] == ""
        assert receipt["unapplied"]["companies"] == []
        assert receipt["company_scope"] == {
            "requested": "CURRENT",
            "observed": {"AfterQuery": "CURRENT", "Mercor": "CURRENT"},
        }
        # A verify-only pass checks the scope too, and it holds.
        await browser.reconcile_facets(
            companies={"AfterQuery", "Mercor"}, titles=set(), verify_only=True
        )

    _run_page(html, _body)


def test_a_scope_that_will_not_read_back_is_a_failed_facet_never_current_or_past():
    html = _scoped_company_rail("AfterQuery", scope_works=False)

    async def _body(browser, page):
        browser.set_company_facet_scope("CURRENT")
        await browser.reconcile_facets(companies={"AfterQuery"}, titles=set())
        receipt = browser.last_facet_reconciliation
        assert receipt["unapplied"]["companies"] == ["AfterQuery"]
        assert receipt["scope_unverified"] == {"companies": ["AfterQuery"]}
        assert receipt["after"]["companies"] == []
        # The Current-or-Past chip did not stay on the rail.
        assert await browser.read_applied_company_chips() == []
        assert "remove:AfterQuery" in await page.evaluate("window.facetInteractions")

    _run_page(html, _body)


def _company_chip_rail(*, label: str | None, pill_aria: str, dismiss_aria: str) -> str:
    """One AfterQuery chip shaped like the 2026-10-15 live capture.

    ``label`` is the visible ``[data-test-pill-label]`` text; None leaves the
    node out entirely. The two aria-labels are set independently, because live
    they go stale apart from the visible label.
    """
    shown = "" if label is None else f"<span data-test-pill-label>{label}</span>"
    chip = (
        f'<li data-test-facet-pills-item><span class="facet-pill" '
        f'aria-label="{pill_aria}" title="AfterQuery (Current)">{shown}'
        f'<button data-test-pill-dismiss aria-label="{dismiss_aria}">x</button>'
        "</span></li>"
    )
    return f"""
<aside class="left-rail"><div data-test-filter-edit-panel>
  <div class="search-facet-wrapper facet-locations"><ul>{_pill("United States")}</ul></div>
  <div class="search-facet-wrapper facet-companies"><ul>{chip}</ul></div>
  <div class="search-facet-wrapper facet-titles"><ul></ul></div>
</div></aside>
"""


def test_a_default_chip_with_a_stale_current_aria_label_reads_current_or_past():
    """Live: scope set back to default, visible label bare, dismiss still "(Current)"."""
    html = _company_chip_rail(
        label="AfterQuery",
        pill_aria="Can have, AfterQuery (Current)",
        dismiss_aria="Remove AfterQuery (Current)",
    )

    async def _body(browser, page):
        assert await browser.read_applied_company_chips() == ["AfterQuery"]
        assert await browser.read_applied_company_chip_scopes() == {
            "AfterQuery": "CURRENT_OR_PAST"
        }
        # The dangerous direction: a pinned CURRENT must not verify on this rail.
        browser.set_company_facet_scope("CURRENT")
        with pytest.raises(FacetReconciliationError) as raised:
            await browser.reconcile_facets(
                companies={"AfterQuery"}, titles=set(), verify_only=True
            )
        assert raised.value.gate == "company_scope_mismatch"

    _run_page(html, _body)


def test_a_current_visible_label_reads_current_whatever_the_aria_labels_say():
    html = _company_chip_rail(
        label="AfterQuery (Current)",
        pill_aria="Can have, AfterQuery",
        dismiss_aria="Remove AfterQuery",
    )

    async def _body(browser, page):
        assert await browser.read_applied_company_chip_scopes() == {
            "AfterQuery": "CURRENT"
        }

    _run_page(html, _body)


def test_a_chip_without_a_label_node_still_reads_its_dismiss_aria_suffix():
    html = _company_chip_rail(
        label=None,
        pill_aria="Can have, AfterQuery (Current). Press backspace to remove",
        dismiss_aria="Remove AfterQuery (Current)",
    )

    async def _body(browser, page):
        assert await browser.read_applied_company_chips() == ["AfterQuery"]
        assert await browser.read_applied_company_chip_scopes() == {
            "AfterQuery": "CURRENT"
        }

    _run_page(html, _body)


def test_a_blank_visible_label_makes_the_company_scope_unreadable():
    """A label node with no text is neither a suffix nor proof of the default."""
    html = _company_chip_rail(
        label="",
        pill_aria="Can have, AfterQuery (Current)",
        dismiss_aria="Remove AfterQuery (Current)",
    )

    async def _body(browser, page):
        assert await browser.read_applied_company_chip_scopes() is None

    _run_page(html, _body)


def test_verify_only_refuses_a_rail_whose_company_scope_reverted():
    html = _scoped_company_rail("AfterQuery")

    async def _body(browser, page):
        browser.set_company_facet_scope("CURRENT")
        with pytest.raises(FacetReconciliationError) as raised:
            await browser.reconcile_facets(
                companies={"AfterQuery"}, titles=set(), verify_only=True
            )
        assert raised.value.gate == "company_scope_mismatch"

    _run_page(html, _body)


# ---------------------------------------------------------------------------
# g. qualifications are cleared before any search, or the opening fails closed
# ---------------------------------------------------------------------------


def test_a_leftover_qualification_is_removed_before_the_search_surface_is_ready():
    html = _summary_page(
        "Currently working at AfterQuery, Handshake, Mercor, or Surge AI",
        "Has led a data annotation program",
    )

    async def _body(browser, page):
        await browser.assert_search_surface_ready()
        assert await page.evaluate("window.removedQualifications") == [
            "Currently working at AfterQuery, Handshake, Mercor, or Surge AI",
            "Has led a data annotation program",
        ]
        assert browser.last_removed_qualifications == [
            "Currently working at AfterQuery, Handshake, Mercor, or Surge AI",
            "Has led a data annotation program",
        ]
        assert await page.locator("[data-test-qualification-item]").count() == 0
        # The rail is left in the edit panel, ready for the facets.
        assert await browser.read_applied_location_chips() == ["United States"]

    _run_page(html, _body, url=SEARCH_URL, project_id="9000000003")


def test_a_qualification_that_will_not_go_fails_the_surface_closed():
    html = _summary_page("Currently working at AfterQuery", stick=True)

    async def _body(browser, page):
        with pytest.raises(FacetReconciliationError) as raised:
            await browser.assert_search_surface_ready()
        assert raised.value.gate == "qualification_removal_failed"

    _run_page(html, _body, url=SEARCH_URL, project_id="9000000003")


def test_the_opening_logs_each_removed_qualification(tmp_path):
    import json

    from linkedin.search_surface import SearchSurface

    surface = SearchSurface.__new__(SearchSurface)
    browser = SimpleNamespace(last_removed_qualifications=["Currently working at AfterQuery"])
    surface.context = SimpleNamespace(browser=browser, log_path=tmp_path / "run_log.jsonl")
    surface._log_removed_qualifications(search_string_id=9)
    events = [
        json.loads(line)
        for line in (tmp_path / "run_log.jsonl").read_text().splitlines()
    ]
    assert [(e["event"], e["string_id"], e["qualification"]) for e in events] == [
        ("search_qualification_removed", 9, "Currently working at AfterQuery")
    ]
