"""Browser-boundary tests for the LinkedIn candidate-note primitive."""

from __future__ import annotations

import asyncio
import json
import random
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from linkedin.browser import (
    PROFILE_PANEL_SELECTOR,
    _PROFILE_PANEL_SCOPE,
    LinkedInBrowser,
)
from linkedin.input_backends import TypingResult, build_note_typing_plan
from shared.governor import UNGOVERNED_FOR_TESTS


# The note is the evaluation alone as of 2026-09-21: no marker, no date, no
# string id, no labels. The duplicate probe is the note's OWN first line.
NOTE_TEXT = (
    "Senior GenAI data scientist with recent hands-on LLM fine-tuning work.\n\n"
    "At Dataverso her own bullets state PEFT/QLoRA fine-tuning on Llama2."
)


class _DOM:
    def __init__(
        self,
        *,
        notes=(),
        trigger_text="More actions for Ada Lovelace",
        trigger_count=1,
        menu_item_count=1,
        menu_expands=True,
        collapsed=False,
        profile_body_expander=False,
        expand_persists=False,
        readable=True,
        panel_text: str | None = None,
        readback: str | None = None,
        visibility_id="ember6086-note-visbility-option-PROJECT",
        persist=True,
    ):
        self.events: list[str] = []
        self.notes = list(notes)
        self.trigger_text = trigger_text
        self.trigger_count = trigger_count
        self.menu_item_count = menu_item_count
        self.menu_expands = menu_expands
        self.menu_open = False
        self.collapsed = collapsed
        self.expanded = not collapsed
        self.profile_body_expander = profile_body_expander
        self.expand_persists = expand_persists
        self.readable = readable
        self.panel_text = panel_text
        self.readback = readback
        self.visibility_id = visibility_id
        self.persist = persist
        self.composer_open = False
        self.typed = ""
        self.hovered = None
        self.page = _Page(self)

    def visible_notes(self) -> list[str]:
        return self.notes if self.expanded else self.notes[:1]

    def press(self, key: str) -> None:
        self.events.append(key)
        if key == "notes_expand":
            if not self.expand_persists:
                self.expanded = True
        elif key == "more_actions":
            if self.menu_expands:
                self.menu_open = True
        elif key == "menu_add_note":
            self.menu_open = False
            self.composer_open = True
        elif key == "Escape":
            self.menu_open = False
        elif key == "cancel":
            self.composer_open = False
        elif key == "submit":
            if self.persist:
                self.composer_open = False
                self.notes.append(self.typed)


class _Mouse:
    def __init__(self, dom: _DOM):
        self.dom = dom

    async def down(self):
        return None

    async def up(self):
        self.dom.press(self.dom.hovered)


class _Page:
    def __init__(self, dom: _DOM):
        self.dom = dom
        self.url = "https://www.linkedin.com/talent/recruiterSearch/profile/ada"
        self.mouse = _Mouse(dom)
        self.keyboard = _Keyboard(dom)

    async def wait_for_function(self, *args, **kwargs):
        # LinkedIn's CSP forbids 'unsafe-eval'; wait_for_function evaluates a
        # string as JavaScript in the page and raises on EVERY call. Measured
        # live 2026-08-22. Reproduced here so reintroducing it fails loudly.
        raise RuntimeError(
            "EvalError: Evaluating a string as JavaScript violates the "
            "following Content Security Policy directive"
        )

    def locator(self, selector: str):
        mapping = {
            LinkedInBrowser._NOTE_TITLE: "title",
            LinkedInBrowser._NOTE_EXPAND_PRIMARY: "notes_expand",
            LinkedInBrowser._NOTE_EXPAND_FALLBACK: "notes_expand",
            _PROFILE_PANEL_SCOPE
            + " [data-test-expandable-list-expand-btn]": "panel_expands",
            LinkedInBrowser._NOTE_MORE_ACTIONS_TRIGGER_PRIMARY: "more_actions",
            LinkedInBrowser._NOTE_ADD_MENU_ITEM_PRIMARY: "menu_add_note",
            LinkedInBrowser._NOTE_FORM_PRIMARY: "form",
            LinkedInBrowser._NOTE_FORM_FALLBACK: "form",
            PROFILE_PANEL_SELECTOR: "panel",
        }
        if selector not in mapping:
            raise AssertionError(f"Unexpected page selector: {selector}")
        return _Locator(self.dom, mapping[selector])

    async def wait_for_timeout(self, _milliseconds: int):
        return None

    async def wait_for_function(self, _expression: str, *, arg: str, timeout: int):
        assert arg == LinkedInBrowser._NOTE_MORE_ACTIONS_TRIGGER_PRIMARY
        if not self.dom.menu_open:
            raise RuntimeError("menu did not open")
        return True


class _Locator:
    def __init__(self, dom: _DOM, key: str):
        self.dom = dom
        self.key = key

    @property
    def first(self):
        return self

    def locator(self, selector: str):
        if self.key == "form":
            mapping = {
                LinkedInBrowser._NOTE_TEXTAREA_PRIMARY: "textarea",
                LinkedInBrowser._NOTE_TEXTAREA_FALLBACK: "textarea",
                LinkedInBrowser._NOTE_VISIBILITY_CHECKED: "visibility",
                LinkedInBrowser._NOTE_SUBMIT: "submit",
                LinkedInBrowser._NOTE_CANCEL: "cancel",
            }
            if selector in mapping:
                return _Locator(self.dom, mapping[selector])
        raise AssertionError(f"Unexpected nested selector: {self.key} -> {selector}")

    async def count(self):
        if self.key == "notes_expand":
            # Live behaviour: the notes expander stays MOUNTED whenever the
            # list exists and merely goes hidden when there is nothing to
            # expand. Presence must not be read as "collapsed".
            return int(bool(self.dom.notes))
        if self.key == "panel_expands":
            return int(self.dom.profile_body_expander) + int(
                self.dom.expand_persists
                or (self.dom.collapsed and not self.dom.expanded)
            )
        if self.key == "more_actions":
            return self.dom.trigger_count
        if self.key == "menu_add_note":
            return self.dom.menu_item_count if self.dom.menu_open else 0
        if self.key == "panel":
            return 1
        if self.key == "form":
            return int(self.dom.composer_open)
        return 1

    async def is_visible(self, *, timeout: int = 5000):
        if self.key == "notes_expand":
            # Only actually visible while something remains collapsed.
            return bool(
                self.dom.expand_persists
                or (self.dom.collapsed and not self.dom.expanded)
            )
        if self.key == "form":
            return bool(self.dom.composer_open)
        if self.key == "menu_add_note":
            return bool(self.dom.menu_open) and bool(self.dom.menu_item_count)
        return True

    async def wait_for(self, *, state: str, timeout: int):
        if state != "visible" or (
            self.key == "form" and not self.dom.composer_open
        ):
            raise RuntimeError("not visible")

    async def inner_text(self, *, timeout: int):
        if self.key == "title":
            return f"Notes ({len(self.dom.notes)})"
        if self.key == "more_actions":
            return self.dom.trigger_text
        assert self.key == "panel"
        self.dom.events.append("read_panel")
        if not self.dom.readable:
            raise RuntimeError("panel unreadable")
        return (
            self.dom.panel_text
            if self.dom.panel_text is not None
            else "\n".join(self.dom.visible_notes())
        )

    async def text_content(self, *, timeout: int):
        assert self.key == "more_actions"
        return self.dom.trigger_text

    async def get_attribute(self, name: str):
        if self.key == "more_actions" and name == "aria-expanded":
            return str(self.dom.menu_open).lower()
        if self.key == "visibility" and name == "id":
            return self.dom.visibility_id
        raise AssertionError(f"Unexpected attribute: {self.key}.{name}")

    async def input_value(self, *, timeout: int):
        assert self.key == "textarea"
        self.dom.events.append("readback")
        return self.dom.typed if self.dom.readback is None else self.dom.readback

    async def is_enabled(self, *, timeout: int):
        assert self.key == "submit"
        return bool(self.dom.typed)

    async def scroll_into_view_if_needed(self, *, timeout: int):
        return None

    async def hover(self, *, timeout: int):
        self.dom.hovered = self.key


class _Keyboard:
    def __init__(self, dom: _DOM):
        self.dom = dom

    async def press(self, key: str):
        self.dom.press(key)


class _InputBackend:
    def __init__(self, dom: _DOM, *, change_identity_after_type=False):
        self.dom = dom
        self.change_identity_after_type = change_identity_after_type

    async def click_locator(self, _page, _locator, *, about_to_commit=None):
        return False

    async def type_text(self, _page, _locator, text, *, plan):
        self.dom.events.append("type")
        self.dom.typed = text
        if self.change_identity_after_type:
            self.dom.page.url = (
                "https://www.linkedin.com/talent/recruiterSearch/profile/grace"
            )
        return TypingResult(
            transport="test",
            duration_ms=1,
            typo_count=plan.typo_count,
            used_correction=plan.used_correction,
        )


def _browser(dom: _DOM, *, change_identity_after_type=False) -> LinkedInBrowser:
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._page = dom.page
    browser._input_backend = _InputBackend(
        dom,
        change_identity_after_type=change_identity_after_type,
    )
    return browser


def _run(awaitable):
    with patch("linkedin.browser.asyncio.sleep", new=AsyncMock()):
        return asyncio.run(awaitable)


def test_happy_path_submits_once_and_confirms_the_note():
    """A successful invocation presses submit exactly once and confirms persistence."""
    dom = _DOM()
    browser = _browser(dom)

    assert _run(browser.write_candidate_note(NOTE_TEXT, expected_name="Ada Lovelace"))
    assert dom.events.count("more_actions") == 1
    assert dom.events.count("menu_add_note") == 1
    assert dom.events.count("submit") == 1
    assert (
        dom.events.index("more_actions")
        < dom.events.index("menu_add_note")
        < dom.events.index("submit")
    )
    assert NOTE_TEXT in dom.notes
    assert browser._last_note_failure_reason is None


@pytest.mark.parametrize("trigger_count", [0, 2])
def test_more_actions_trigger_requires_exactly_one_match(trigger_count):
    dom = _DOM(trigger_count=trigger_count)
    browser = _browser(dom)

    assert not _run(
        browser.write_candidate_note(NOTE_TEXT, expected_name="Ada Lovelace")
    )
    assert "more_actions" not in dom.events
    assert "menu_add_note" not in dom.events
    assert "submit" not in dom.events
    assert browser._last_note_failure_reason == "note_button_not_found"
    assert dom.events.count("submit") <= 1


@pytest.mark.parametrize("menu_item_count", [0, 2])
def test_add_note_menu_item_requires_exactly_one_match(menu_item_count):
    dom = _DOM(menu_item_count=menu_item_count)
    browser = _browser(dom)

    assert not _run(
        browser.write_candidate_note(NOTE_TEXT, expected_name="Ada Lovelace")
    )
    assert dom.events.count("more_actions") == 1
    assert "menu_add_note" not in dom.events
    assert dom.events.count("Escape") == 1
    assert "submit" not in dom.events
    assert browser._last_note_failure_reason == "note_menu_item_not_found"


def test_menu_that_never_opens_is_dismissed_without_submit():
    dom = _DOM(menu_expands=False)
    browser = _browser(dom)

    assert not _run(
        browser.write_candidate_note(NOTE_TEXT, expected_name="Ada Lovelace")
    )
    assert dom.events.count("more_actions") == 1
    assert "menu_add_note" not in dom.events
    assert dom.events.count("Escape") == 1
    assert "submit" not in dom.events
    assert browser._last_note_failure_reason == "note_menu_item_not_found"


def test_existing_note_with_the_same_opening_short_circuits_without_submit():
    """A note already opening with this evaluation is idempotent success."""
    dom = _DOM(notes=[NOTE_TEXT])
    browser = _browser(dom)

    assert _run(browser.write_candidate_note(NOTE_TEXT, expected_name="Ada Lovelace"))
    assert "more_actions" not in dom.events
    assert "menu_add_note" not in dom.events
    assert "submit" not in dom.events
    assert browser._last_note_failure_reason == "note_already_present"


def test_collapsed_notes_expand_before_marker_read():
    """A collapsed list is expanded before panel text is read."""
    dom = _DOM(notes=["Human note", NOTE_TEXT], collapsed=True)
    browser = _browser(dom)

    assert _run(browser.write_candidate_note(NOTE_TEXT, expected_name="Ada Lovelace"))
    assert dom.events.index("notes_expand") < dom.events.index("read_panel")
    assert "submit" not in dom.events


def test_profile_body_expander_does_not_make_notes_probe_ambiguous():
    """Only the notes expander participates in the marker probe."""
    dom = _DOM(
        notes=["Human note", NOTE_TEXT],
        collapsed=True,
        profile_body_expander=True,
    )
    browser = _browser(dom)
    panel_wide = dom.page.locator(
        _PROFILE_PANEL_SCOPE + " [data-test-expandable-list-expand-btn]"
    )

    assert _run(panel_wide.count()) == 2
    assert _run(browser.write_candidate_note(NOTE_TEXT, expected_name="Ada Lovelace"))
    assert dom.events.index("notes_expand") < dom.events.index("read_panel")
    assert "submit" not in dom.events


def test_unreadable_notes_refuse_before_compose():
    """Probe None refuses rather than spending unreadable as marker-absent."""
    dom = _DOM(notes=["Human note"], readable=False)
    browser = _browser(dom)

    assert not _run(
        browser.write_candidate_note(NOTE_TEXT, expected_name="Ada Lovelace")
    )
    assert "more_actions" not in dom.events
    assert "menu_add_note" not in dom.events
    assert "type" not in dom.events
    assert "submit" not in dom.events
    assert browser._last_note_failure_reason == "note_probe_failed"


def test_partially_rendered_notes_refuse_before_compose():
    """A list that cannot fully expand is tri-state unreadable, not absent."""
    dom = _DOM(
        notes=["First", "Second"],
        collapsed=True,
        expand_persists=True,
    )
    browser = _browser(dom)

    assert not _run(
        browser.write_candidate_note(NOTE_TEXT, expected_name="Ada Lovelace")
    )
    assert "more_actions" not in dom.events
    assert "menu_add_note" not in dom.events
    assert "submit" not in dom.events
    assert browser._last_note_failure_reason == "note_probe_failed"


def test_zero_notes_skips_panel_read_and_expand():
    """Notes (0) is provably absent without touching panel text or expansion."""
    dom = _DOM(collapsed=True)
    browser = _browser(dom)

    assert _run(browser.write_candidate_note(NOTE_TEXT, expected_name="Ada Lovelace"))
    probe_events = dom.events[: dom.events.index("more_actions")]
    assert "notes_expand" not in probe_events
    assert "read_panel" not in probe_events
    assert dom.events.count("submit") == 1


def test_positive_count_marker_in_panel_text_short_circuits():
    """A panel-wide probe match is idempotent success without composition."""
    dom = _DOM(notes=["Human note"], panel_text=f"Profile\n{NOTE_TEXT}")
    browser = _browser(dom)

    assert _run(browser.write_candidate_note(NOTE_TEXT, expected_name="Ada Lovelace"))
    assert "read_panel" in dom.events
    assert "more_actions" not in dom.events
    assert "menu_add_note" not in dom.events
    assert "submit" not in dom.events


def test_persisting_expand_control_refuses_before_compose():
    """A still-collapsed list is unknown, never marker-absent."""
    dom = _DOM(notes=["Human note"], collapsed=True, expand_persists=True)
    browser = _browser(dom)

    assert not _run(
        browser.write_candidate_note(NOTE_TEXT, expected_name="Ada Lovelace")
    )
    assert "notes_expand" in dom.events
    assert "read_panel" not in dom.events
    assert "more_actions" not in dom.events
    assert "menu_add_note" not in dom.events
    assert "submit" not in dom.events
    assert browser._last_note_failure_reason == "note_probe_failed"


def test_successful_expansion_without_marker_proceeds_to_write():
    """An expanded marker-free panel allows composition and one submit."""
    dom = _DOM(notes=["Human note"], collapsed=True)
    browser = _browser(dom)

    assert _run(browser.write_candidate_note(NOTE_TEXT, expected_name="Ada Lovelace"))
    assert dom.events.index("notes_expand") < dom.events.index("read_panel")
    assert dom.events.count("submit") == 1
    assert NOTE_TEXT in dom.notes


def test_unreadable_panel_text_refuses_before_compose():
    """An unreadable profile panel remains probe-unknown and fails closed."""
    dom = _DOM(notes=["Human note"], readable=False)
    browser = _browser(dom)

    assert not _run(
        browser.write_candidate_note(NOTE_TEXT, expected_name="Ada Lovelace")
    )
    assert "read_panel" in dom.events
    assert "more_actions" not in dom.events
    assert "menu_add_note" not in dom.events
    assert "submit" not in dom.events
    assert browser._last_note_failure_reason == "note_probe_failed"


def test_more_actions_name_mismatch_refuses_before_click():
    """The identity-bearing trigger text must match the expected identity."""
    dom = _DOM(trigger_text="More actions for Grace Hopper")
    browser = _browser(dom)

    assert not _run(
        browser.write_candidate_note(NOTE_TEXT, expected_name="Ada Lovelace")
    )
    assert "more_actions" not in dom.events
    assert "menu_add_note" not in dom.events
    assert "submit" not in dom.events
    assert browser._last_note_failure_reason == "note_button_name_mismatch"


def test_more_actions_name_match_normalizes_rendering_whitespace():
    dom = _DOM(
        trigger_text=(
            "\n      More actions for Helena Duarte Vieira\n"
            "      \n      \n    \n  \n"
        )
    )
    browser = _browser(dom)

    assert _run(
        browser.write_candidate_note(
            NOTE_TEXT,
            expected_name="Helena Duarte Vieira",
        )
    )
    assert dom.events.count("submit") == 1


@pytest.mark.parametrize(
    ("expected_name", "trigger_text"),
    [
        ("Jose Garcia", "More actions for José García"),
        ("Tao Z.", "More actions for Tao Zhang"),
    ],
)
def test_more_actions_accepts_equivalent_panel_name(expected_name, trigger_text):
    dom = _DOM(trigger_text=trigger_text)
    browser = _browser(dom)

    assert _run(browser.write_candidate_note(NOTE_TEXT, expected_name=expected_name))
    assert dom.events.count("submit") == 1


def test_pre_submit_exception_cancels_without_submit():
    dom = _DOM()
    browser = _browser(dom)
    browser._input_backend.type_text = AsyncMock(
        side_effect=RuntimeError("typing failed")
    )

    assert not _run(
        browser.write_candidate_note(NOTE_TEXT, expected_name="Ada Lovelace")
    )
    assert "cancel" in dom.events
    assert "submit" not in dom.events
    assert browser._last_note_failure_reason == "note_not_submitted"


def test_textarea_mismatch_cancels_without_submit():
    """A corrupted textarea read-back is cancelled before submit."""
    dom = _DOM(readback="corrupted")
    browser = _browser(dom)

    assert not _run(
        browser.write_candidate_note(NOTE_TEXT, expected_name="Ada Lovelace")
    )
    assert "cancel" in dom.events
    assert "submit" not in dom.events
    assert browser._last_note_failure_reason == "note_text_mismatch"


def test_non_project_visibility_cancels_without_submit():
    """A non-project default visibility is observed and refused, never changed."""
    dom = _DOM(visibility_id="ember6086-note-visbility-option-OWNER")
    browser = _browser(dom)

    assert not _run(
        browser.write_candidate_note(NOTE_TEXT, expected_name="Ada Lovelace")
    )
    assert "cancel" in dom.events
    assert "submit" not in dom.events
    assert browser._last_note_failure_reason == "note_visibility_not_project"


def test_before_commit_runs_before_submit():
    """The external guard runs at the commit boundary before the submit press."""
    dom = _DOM()
    browser = _browser(dom)

    def before_commit():
        dom.events.append("before_commit")

    assert _run(
        browser.write_candidate_note(
            NOTE_TEXT,
            expected_name="Ada Lovelace",
            before_commit=before_commit,
        )
    )
    assert dom.events.index("before_commit") < dom.events.index("submit")
    assert dom.events.count("before_commit") == 1
    assert dom.events.count("submit") == 1


def test_unconfirmed_submit_is_never_retried():
    """A failed persistence confirmation never causes a second submit press."""
    dom = _DOM(persist=False)
    browser = _browser(dom)

    assert not _run(
        browser.write_candidate_note(NOTE_TEXT, expected_name="Ada Lovelace")
    )
    assert dom.events.count("submit") == 1
    assert "cancel" in dom.events
    assert browser._last_note_failure_reason == "note_not_persisted"


def test_panel_identity_change_mid_sequence_aborts():
    """A profile identity swap aborts before any irreversible note submit."""
    dom = _DOM()
    browser = _browser(dom, change_identity_after_type=True)

    with pytest.raises(RuntimeError, match="profile identity mismatch"):
        _run(browser.write_candidate_note(NOTE_TEXT, expected_name="Ada Lovelace"))
    assert "submit" not in dom.events
    assert browser._last_note_failure_reason == "note_panel_identity_mismatch"


def test_primitive_never_calls_fill():
    """Note entry uses the input backend, with no direct textarea mutation."""
    dom = _DOM()
    browser = _browser(dom)

    assert _run(browser.write_candidate_note(NOTE_TEXT, expected_name="Ada Lovelace"))
    assert "type" in dom.events


def test_note_typing_plan_pauses_longer_after_sentences_than_words():
    """Prose cadence gives sentence boundaries a longer pause than word gaps."""
    plan = build_note_typing_plan("One two. Three", rng=random.Random(1))
    pauses = [step.delay_seconds for step in plan.steps if step.kind == "pause"]

    assert len(pauses) == 2
    assert pauses[1] > pauses[0]


def _write_smoke_state(tmp_path):
    profile_url = "https://www.linkedin.com/talent/profile/ada"
    verdicts = [
        {
            "stage": "full",
            "decision": "SAVE",
            "path": "direct_experience",
            "confidence": 0.9,
            "rationale": rationale,
            "candidate_name": "Ada Lovelace",
            "profile_url": profile_url,
        }
        for rationale in ("Older judgment.", "Latest real judgment.")
    ]
    snippets = [
        {
            "name": "Ada Lovelace",
            "profile_url": profile_url,
            "source_string_id": 12,
            "source_string_name": "builders",
        }
    ]
    (tmp_path / "final_judgments.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in verdicts)
    )
    (tmp_path / "snippets.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in snippets)
    )
    return profile_url


def test_smoke_dry_run_uses_latest_projection_without_browser(tmp_path, capsys):
    from tools import smoke_candidate_note

    profile_url = _write_smoke_state(tmp_path)
    with patch.object(
        smoke_candidate_note,
        "_run_live",
        side_effect=AssertionError("dry-run touched browser path"),
    ) as live:
        result = smoke_candidate_note.main(
            [
                "--profile-url",
                profile_url,
                "--project-id",
                "9000000010",
                "--state-dir",
                str(tmp_path),
                "--dry-run",
            ]
        )

    output = capsys.readouterr().out
    assert result == 0
    assert "Latest real judgment." in output
    assert "Older judgment." not in output
    assert "string #" not in output
    assert "Cloris" not in output
    assert "note_chars:" in output
    assert "estimated_typing_seconds:" in output
    live.assert_not_called()


class _SmokeAnchor:
    def __init__(self, href: str, *, visible: bool = True):
        self.href = href
        self.visible = visible

    async def get_attribute(self, name: str):
        assert name == "href"
        return self.href

    async def is_visible(self, *, timeout: int):
        return self.visible


class _SmokeLinks:
    def __init__(self, links):
        self.links = links

    async def count(self):
        return len(self.links)

    def nth(self, index: int):
        return self.links[index]


def test_smoke_anchor_lookup_uses_exact_recruiter_identity_fragment():
    from tools.smoke_candidate_note import _find_exact_profile_anchor

    exact = _SmokeAnchor(
        "/talent/hire/9000000010/manage/all/profile/ada?tracking=one"
    )
    page = SimpleNamespace(
        locator=lambda selector: _SmokeLinks(
            [
                _SmokeAnchor("/company/profile/ada"),
                _SmokeAnchor("/talent/recruiterSearch/profile/grace"),
                exact,
            ]
        )
    )

    assert asyncio.run(_find_exact_profile_anchor(page, "ada")) is exact


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"identity": "grace"}, "profile identity precondition failed"),
        ({"panel_count": 0}, "profile panel precondition failed"),
        ({"panel_buttons": 0}, "more-actions trigger precondition failed"),
        ({"panel_buttons": 2}, "more-actions trigger precondition failed"),
        ({"title_count": 0}, "notes title precondition failed"),
        ({"title": "Notes"}, "notes title precondition failed"),
    ],
)
def test_smoke_refuses_every_hard_precondition(override, message):
    from tools.smoke_candidate_note import _check_preconditions

    values = {
        "identity": "ada",
        "panel_count": 1,
        "global_buttons": 1,
        "panel_buttons": 1,
        "title_count": 1,
        "title": "Notes (2)",
        **override,
    }
    title = MagicMock()
    title.inner_text = AsyncMock(return_value=values["title"])
    titles = MagicMock()
    titles.count = AsyncMock(return_value=values["title_count"])
    titles.first = title
    panel_buttons = MagicMock()
    panel_buttons.count = AsyncMock(return_value=values["panel_buttons"])
    panel = MagicMock()
    panel.wait_for = AsyncMock()
    panel.locator.side_effect = lambda selector: (
        panel_buttons
        if "data-test-more-actions-trigger" in selector
        else titles
    )
    panels = MagicMock()
    panels.count = AsyncMock(return_value=values["panel_count"])
    panels.first = panel
    global_buttons = MagicMock()
    global_buttons.count = AsyncMock(return_value=values["global_buttons"])
    page = MagicMock()
    def _page_locator(selector):
        if "data-test-more-actions-trigger" not in selector:
            return panels
        # Page-wide count is informational; the panel-scoped count is the gate.
        return panel_buttons if ":is(" in selector else global_buttons

    page.locator.side_effect = _page_locator
    browser = SimpleNamespace(
        current_profile_identity_fragment=lambda: values["identity"],
        page=page,
    )

    with pytest.raises(RuntimeError, match=message):
        asyncio.run(_check_preconditions(browser, "ada"))


def test_smoke_reads_the_smallest_complete_landed_note():
    from tools.smoke_candidate_note import _read_landed_note

    expected = NOTE_TEXT
    descendants = MagicMock()
    descendants.evaluate_all = AsyncMock(
        return_value=[
            # The probe-bearing but INCOMPLETE fragment: the note's own first
            # line, which the rail renders as its own element.
            NOTE_TEXT.splitlines()[0],
            expected,
            f"Profile header\n{expected}\nEdit note",
        ]
    )
    expand = MagicMock()
    expand.count = AsyncMock(return_value=0)
    panel = MagicMock()
    panel.locator.side_effect = lambda selector: (
        descendants if selector == "*" else expand
    )
    panels = MagicMock()
    panels.count = AsyncMock(return_value=1)
    panels.first = panel
    page = MagicMock()
    # The reader now resolves the notes expander at PAGE level with the
    # primitive's notes-scoped selectors (a profile-body expander also matches
    # the generic attribute), so route those to the expander mock.
    page.locator.side_effect = lambda selector: (
        expand if "expand" in selector else panels
    )
    browser = SimpleNamespace(page=page)

    assert asyncio.run(_read_landed_note(browser, NOTE_TEXT)) == expected


def test_live_smoke_calls_production_primitive_and_flags_landed_mismatch(capsys):
    from tools import smoke_candidate_note

    class _Governor:
        def can_start_session(self):
            return True, "ok"

        start_session = MagicMock()
        check_profile_open_or_raise = MagicMock()
        record_profile_open = MagicMock()
        end_session = MagicMock()

    panel = MagicMock()
    panel.wait_for = AsyncMock()
    panels = MagicMock()
    panels.first = panel
    page = MagicMock()
    page.goto = AsyncMock()
    page.wait_for_timeout = AsyncMock()
    page.locator.return_value = panels

    class _Browser:
        instance = None

        def __init__(self, *, governor):
            self.governor = governor
            self.page = page
            self._last_note_failure_reason = None
            self.write_candidate_note = AsyncMock(return_value=True)
            self._ghost_click_locator = AsyncMock()
            _Browser.instance = self

        @staticmethod
        def _profile_url_fragment(url):
            return LinkedInBrowser._profile_url_fragment(url)

        set_required_project_id = MagicMock()
        connect = AsyncMock()
        disconnect = AsyncMock()

    decision = SimpleNamespace(candidate_name="Ada Lovelace")
    with patch("linkedin.browser.LinkedInBrowser", _Browser), patch(
        "shared.governor.SessionGovernor", _Governor
    ), patch.object(
        smoke_candidate_note,
        "_find_exact_profile_anchor",
        new=AsyncMock(return_value=object()),
    ), patch.object(
        smoke_candidate_note,
        "_check_preconditions",
        new=AsyncMock(return_value="Notes (1)"),
    ), patch.object(
        smoke_candidate_note,
        "_diagnostic_read",
        new=AsyncMock(side_effect=["Notes (2)", "Senior GenAI data scien"]),
    ):
        result = asyncio.run(
            smoke_candidate_note._run_live(
                profile_url="https://www.linkedin.com/talent/profile/ada",
                project_id="9000000010",
                decision=decision,
                note_text=NOTE_TEXT,
            )
        )

    assert result == 1
    _Browser.instance.write_candidate_note.assert_awaited_once_with(
        NOTE_TEXT,
        expected_name="Ada Lovelace",
    )
    assert _Browser.instance.governor.record_profile_open.call_count == 1
    output = capsys.readouterr().out
    assert 'landed_note: "Senior GenAI data scien"' in output
    assert "MISMATCH: LANDED NOTE DOES NOT MATCH INTENDED NOTE" in output


def test_note_write_never_uses_wait_for_function():
    """LOCK (live 2026-08-22): LinkedIn's CSP forbids 'unsafe-eval', so
    page.wait_for_function raises on every call. Using it made a healthy menu
    report as note_menu_item_not_found on every single write."""
    dom = _DOM()
    browser = _browser(dom)

    # The fake page's wait_for_function raises the real CSP error, so a happy
    # path that still succeeds proves the primitive never calls it.
    assert _run(browser.write_candidate_note(NOTE_TEXT, expected_name="Ada Lovelace"))
    assert dom.events.count("submit") == 1
    assert browser._last_note_failure_reason is None


def test_confirmation_succeeds_while_the_composer_remains_open():
    """LOCK (live 2026-08-22): LinkedIn leaves a fresh EMPTY composer mounted
    after a successful submit. Requiring the form to have closed reported a
    note that demonstrably landed as note_not_confirmed."""
    dom = _DOM()
    browser = _browser(dom)
    # persist the note but keep the composer mounted, exactly as LinkedIn does
    original_press = dom.press

    def press(key: str) -> None:
        if key == "submit":
            dom.events.append(key)
            if dom.persist:
                dom.notes.append(dom.typed)
            return  # composer deliberately stays open
        original_press(key)

    dom.press = press

    assert _run(browser.write_candidate_note(NOTE_TEXT, expected_name="Ada Lovelace"))
    assert dom.composer_open is True
    assert browser._last_note_failure_reason is None


def test_hidden_expand_control_is_never_clicked():
    """LOCK (live 2026-08-22): the notes expander stays MOUNTED BUT HIDDEN when
    there is nothing to expand. Ghost-clicking a hidden element raises, which
    made the primitive refuse to recognise its OWN note (note_probe_failed)."""
    dom = _DOM(notes=[NOTE_TEXT])
    browser = _browser(dom)

    assert _run(browser.write_candidate_note(NOTE_TEXT, expected_name="Ada Lovelace"))
    assert "notes_expand" not in dom.events
    assert "submit" not in dom.events
    assert browser._last_note_failure_reason == "note_already_present"


def test_note_with_a_different_opening_is_not_a_duplicate():
    """Only this evaluation's own opening counts: a colleague's note is not it."""
    dom = _DOM(notes=["Spoke to Ada last quarter about the platform role."])
    browser = _browser(dom)

    assert _run(browser.write_candidate_note(NOTE_TEXT, expected_name="Ada Lovelace"))
    assert dom.events.count("submit") == 1
    assert NOTE_TEXT in dom.notes
    assert browser._last_note_failure_reason is None


def test_probe_matches_through_rendered_line_wrapping():
    """The rail re-wraps note text; both sides are whitespace-normalized."""
    wrapped = NOTE_TEXT.splitlines()[0].replace(" ", "\n   ", 3)
    dom = _DOM(notes=["Human note"], panel_text=f"Notes\n{wrapped}")
    browser = _browser(dom)

    assert _run(browser.write_candidate_note(NOTE_TEXT, expected_name="Ada Lovelace"))
    assert "submit" not in dom.events
    assert browser._last_note_failure_reason == "note_already_present"


@pytest.mark.parametrize("note_text", ["", "   \n\n  "])
def test_empty_note_text_refuses_before_touching_the_dom(note_text):
    """An empty probe must never be spent as 'the note is already there'."""
    dom = _DOM(notes=["Human note"])
    browser = _browser(dom)

    assert not _run(
        browser.write_candidate_note(note_text, expected_name="Ada Lovelace")
    )
    assert dom.events == []
    assert browser._last_note_failure_reason == "empty_note"
