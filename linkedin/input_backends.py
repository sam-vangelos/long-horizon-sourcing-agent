"""Session-scoped input backends for concurrent and away-from-keyboard modes."""

from __future__ import annotations

import asyncio
import ctypes
import ctypes.util
import math
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

from decoy.actions._utils import human_scroll
from shared import config
from shared.human_timing import human_delay_correlated

if TYPE_CHECKING:
    from rebrowser_playwright.async_api import Locator, Page


def normalize_input_mode(mode: str | None) -> str:
    raw = (mode or "concurrent").strip().lower().replace("-", "_")
    aliases = {
        "concurrent": "concurrent",
        "synthetic": "concurrent",
        "ghostcursor": "concurrent",
        "ghost_cursor": "concurrent",
        "away": "away",
        "takeover": "away",
        "away_from_keyboard": "away",
        "afk": "away",
    }
    if raw not in aliases:
        raise ValueError(f"Unsupported input mode: {mode}")
    return aliases[raw]


@dataclass(frozen=True)
class TypingStep:
    kind: str
    value: str = ""
    delay_seconds: float = 0.0
    source_index: int | None = None
    is_correction: bool = False


@dataclass(frozen=True)
class TypingPlan:
    steps: list[TypingStep]
    typo_positions: tuple[int, ...] = field(default_factory=tuple)

    @property
    def typo_count(self) -> int:
        return len(self.typo_positions)

    @property
    def used_correction(self) -> bool:
        return bool(self.typo_positions)


@dataclass(frozen=True)
class TypingResult:
    transport: str
    duration_ms: int
    typo_count: int
    used_correction: bool
    fallback_char_count: int = 0


_BOOLEAN_OPERATORS = {"AND", "OR", "NOT"}
_THINKING_PAUSE_MIN_INDEX = 18


def _random_alpha_replacement(ch: str, rng: random.Random) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    if ch.isupper():
        alphabet = alphabet.upper()
    candidates = [candidate for candidate in alphabet if candidate != ch]
    return rng.choice(candidates)


def _char_delay(rng: random.Random) -> float:
    return rng.uniform(
        config.LINKEDIN_SEARCH_TYPING_CHAR_MIN_SECONDS,
        config.LINKEDIN_SEARCH_TYPING_CHAR_MAX_SECONDS,
    )


_HOME_ROW_KEYS = frozenset("asdfjkl;ASDFJKL;")
_DWELL_FLOOR_SECONDS = 0.03
_DWELL_CEILING_SECONDS = 0.30
_DWELL_LOG_MEDIAN_SECONDS = 0.09
_DWELL_LOG_SIGMA = 0.35
_HOME_ROW_DWELL_MULTIPLIER = 0.8
_FAST_BIGRAM_OVERLAP_PROBABILITY = 0.25


def _sample_key_dwell(ch: str) -> float:
    """Sample key-down hold duration (seconds) from a right-skewed log-normal."""
    dwell = random.lognormvariate(math.log(_DWELL_LOG_MEDIAN_SECONDS), _DWELL_LOG_SIGMA)
    if ch == " " or ch in _HOME_ROW_KEYS:
        dwell *= _HOME_ROW_DWELL_MULTIPLIER
    return max(_DWELL_FLOOR_SECONDS, min(_DWELL_CEILING_SECONDS, dwell))


def _is_fast_bigram(prev_ch: str, next_ch: str) -> bool:
    return (
        prev_ch.isalpha()
        and next_ch.isalpha()
        and prev_ch.lower() != next_ch.lower()
    )


def build_boolean_typing_plan(text: str, *, rng: random.Random | None = None) -> TypingPlan:
    rng = rng or random.Random()
    eligible_positions: list[int] = []
    operator_pause_after: set[int] = set()
    closing_pause_after: set[int] = set()

    token_start = 0
    while token_start < len(text):
        if text[token_start].isalpha():
            token_end = token_start
            while token_end < len(text) and text[token_end].isalpha():
                token_end += 1
            token = text[token_start:token_end]
            if len(token) >= 5 and token.upper() not in _BOOLEAN_OPERATORS:
                eligible_positions.extend(range(token_start, token_end))
            if token.upper() in _BOOLEAN_OPERATORS:
                operator_pause_after.add(token_end - 1)
            token_start = token_end
            continue
        token_start += 1

    for idx, ch in enumerate(text):
        if ch not in (")", '"', "'"):
            continue
        next_ch = text[idx + 1] if idx + 1 < len(text) else ""
        if not next_ch or next_ch.isspace() or next_ch in ")(":
            closing_pause_after.add(idx)

    typo_positions: list[int] = []
    if len(text) >= 25 and eligible_positions:
        if len(text) < 60:
            if rng.random() < config.LINKEDIN_SEARCH_TYPING_MEDIUM_TYPO_PROBABILITY:
                typo_positions.append(rng.choice(eligible_positions))
        else:
            if rng.random() < config.LINKEDIN_SEARCH_TYPING_LONG_TYPO_PROBABILITY:
                first = rng.choice(eligible_positions)
                typo_positions.append(first)
                if (
                    rng.random() < config.LINKEDIN_SEARCH_TYPING_SECOND_TYPO_PROBABILITY
                    and len(typo_positions) < config.LINKEDIN_SEARCH_TYPING_MAX_TYPOS
                ):
                    candidates = [
                        idx for idx in eligible_positions if abs(idx - first) >= 20
                    ]
                    if candidates:
                        typo_positions.append(rng.choice(candidates))

    typo_positions = sorted(typo_positions[: config.LINKEDIN_SEARCH_TYPING_MAX_TYPOS])

    thought_pause_after: int | None = None
    if len(text) >= 45 and rng.random() < 0.6:
        candidates = [
            idx
            for idx, ch in enumerate(text)
            if ch.isspace()
            and _THINKING_PAUSE_MIN_INDEX <= idx <= max(_THINKING_PAUSE_MIN_INDEX, len(text) - 10)
        ]
        if candidates:
            thought_pause_after = rng.choice(candidates)

    typo_position_set = set(typo_positions)
    slowdown_remaining = 0
    steps: list[TypingStep] = []

    for idx, ch in enumerate(text):
        if idx in typo_position_set:
            wrong_char = _random_alpha_replacement(ch, rng)
            steps.append(
                TypingStep(
                    kind="char",
                    value=wrong_char,
                    delay_seconds=_char_delay(rng),
                    source_index=idx,
                )
            )
            steps.append(
                TypingStep(
                    kind="backspace",
                    delay_seconds=_char_delay(rng),
                    source_index=idx,
                )
            )
            steps.append(
                TypingStep(
                    kind="char",
                    value=ch,
                    delay_seconds=_char_delay(rng),
                    source_index=idx,
                    is_correction=True,
                )
            )
            slowdown_remaining = 4
        else:
            delay = _char_delay(rng)
            if slowdown_remaining > 0:
                delay += rng.uniform(0.02, 0.04)
                slowdown_remaining -= 1
            steps.append(
                TypingStep(
                    kind="char",
                    value=ch,
                    delay_seconds=delay,
                    source_index=idx,
                )
            )

        if idx in operator_pause_after or idx in closing_pause_after:
            steps.append(
                TypingStep(
                    kind="pause",
                    delay_seconds=rng.uniform(
                        config.LINKEDIN_SEARCH_TYPING_OPERATOR_PAUSE_MIN_SECONDS,
                        config.LINKEDIN_SEARCH_TYPING_OPERATOR_PAUSE_MAX_SECONDS,
                    ),
                    source_index=idx,
                )
            )
        if thought_pause_after is not None and idx == thought_pause_after:
            steps.append(
                TypingStep(
                    kind="pause",
                    delay_seconds=rng.uniform(
                        config.LINKEDIN_SEARCH_TYPING_THOUGHT_PAUSE_MIN_SECONDS,
                        config.LINKEDIN_SEARCH_TYPING_THOUGHT_PAUSE_MAX_SECONDS,
                    ),
                    source_index=idx,
                )
            )

    return TypingPlan(steps=steps, typo_positions=tuple(typo_positions))


class InputBackend:
    """Abstract browser input backend."""

    mode = "concurrent"
    status_label = "uninitialized"

    async def initialize(self, page: "Page") -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def shutdown(self) -> None:
        return None

    async def click_selector(self, page: "Page", selector: str) -> bool:
        raise NotImplementedError

    async def move_selector(self, page: "Page", selector: str) -> bool:
        raise NotImplementedError

    async def click_locator(
        self,
        page: "Page",
        locator: "Locator",
        *,
        about_to_commit: Callable[[], None] | None = None,
    ) -> bool:
        raise NotImplementedError

    async def scroll(self, page: "Page", delta_y: int, *, channel: str = "scroll") -> int:
        raise NotImplementedError

    async def press_key(self, page: "Page", key: str) -> bool:
        raise NotImplementedError

    async def press_combo(self, page: "Page", combo: str) -> bool:
        raise NotImplementedError

    async def type_text(
        self,
        page: "Page",
        locator: "Locator",
        text: str,
        *,
        plan: TypingPlan,
    ) -> TypingResult:
        raise NotImplementedError


class ConcurrentInputBackend(InputBackend):
    """Current synthetic-input mode: Playwright + ghost cursor."""

    mode = "concurrent"

    def __init__(self):
        self._cursor = None
        self.status_label = "ghost-cursor unavailable"

    async def initialize(self, page: "Page") -> None:
        try:
            from python_ghost_cursor.playwright_async import create_cursor

            self._cursor = create_cursor(page)
            self.status_label = "ghost-cursor active"
        except Exception as e:  # pragma: no cover - depends on optional dependency
            self._cursor = None
            self.status_label = f"ghost-cursor unavailable: {e}"

    async def click_selector(self, page: "Page", selector: str) -> bool:
        if self._cursor:
            try:
                await asyncio.wait_for(self._cursor.click(selector), timeout=5.0)
                return True
            except Exception:
                pass
        return False

    async def move_selector(self, page: "Page", selector: str) -> bool:
        if self._cursor:
            try:
                await asyncio.wait_for(self._cursor.move(selector), timeout=5.0)
                return True
            except Exception:
                pass
        return False

    async def click_locator(
        self,
        page: "Page",
        locator: "Locator",
        *,
        about_to_commit: Callable[[], None] | None = None,
    ) -> bool:
        if self._cursor:
            try:
                handle = await locator.element_handle(timeout=3000)
                if handle:
                    await asyncio.wait_for(self._cursor.move(handle), timeout=5.0)
            except Exception:
                return False
            if handle:
                if about_to_commit is not None:
                    about_to_commit()
                await page.mouse.down()
                await page.mouse.up()
                await asyncio.sleep(random.random() * 2)
                return True
        return False

    async def scroll(self, page: "Page", delta_y: int, *, channel: str = "scroll") -> int:
        return await human_scroll(page, delta_y, channel=channel)

    async def press_key(self, page: "Page", key: str) -> bool:
        await page.keyboard.press(key)
        return True

    async def press_combo(self, page: "Page", combo: str) -> bool:
        await page.keyboard.press(combo)
        return True

    async def _type_char_atomic(
        self,
        page: "Page",
        ch: str,
        *,
        fallback_char_count: int,
    ) -> int:
        try:
            await page.keyboard.type(ch, delay=0)
        except Exception:
            await page.keyboard.insert_text(ch)
            fallback_char_count += 1
        return fallback_char_count

    async def _type_char_with_dwell(
        self,
        page: "Page",
        ch: str,
        *,
        skip_down: bool,
        next_char: str | None,
        fallback_char_count: int,
    ) -> tuple[int, bool]:
        """Type one character with sampled dwell; optionally overlap the next key down."""
        dwell = _sample_key_dwell(ch)
        overlap_scheduled = False
        self_down = skip_down
        try:
            if not skip_down:
                await page.keyboard.down(ch)
                self_down = True
            await asyncio.sleep(dwell)
            if (
                next_char is not None
                and _is_fast_bigram(ch, next_char)
                and random.random() < _FAST_BIGRAM_OVERLAP_PROBABILITY
            ):
                try:
                    await page.keyboard.down(next_char)
                    overlap_scheduled = True
                except Exception:
                    overlap_scheduled = False
            await page.keyboard.up(ch)
            self_down = False
        except Exception:
            if self_down:
                try:
                    await page.keyboard.up(ch)
                except Exception:
                    pass
                self_down = False
            else:
                fallback_char_count = await self._type_char_atomic(
                    page,
                    ch,
                    fallback_char_count=fallback_char_count,
                )
        return fallback_char_count, overlap_scheduled

    async def type_text(
        self,
        page: "Page",
        locator: "Locator",
        text: str,
        *,
        plan: TypingPlan,
    ) -> TypingResult:
        start = time.perf_counter()
        fallback_char_count = 0
        steps = plan.steps
        overlap_next_down = False
        for step_index, step in enumerate(steps):
            if step.kind == "pause":
                overlap_next_down = False
                await asyncio.sleep(step.delay_seconds)
                continue
            if step.kind == "backspace":
                overlap_next_down = False
                await page.keyboard.press("Backspace")
            elif step.kind == "char":
                if not config.LINKEDIN_TYPING_DWELL_ENABLED:
                    try:
                        await page.keyboard.type(step.value, delay=0)
                    except Exception:
                        await page.keyboard.insert_text(step.value)
                        fallback_char_count += 1
                else:
                    next_char: str | None = None
                    next_index = step_index + 1
                    if next_index < len(steps) and steps[next_index].kind == "char":
                        next_char = steps[next_index].value
                    fallback_char_count, overlap_next_down = await self._type_char_with_dwell(
                        page,
                        step.value,
                        skip_down=overlap_next_down,
                        next_char=next_char,
                        fallback_char_count=fallback_char_count,
                    )
            if step.delay_seconds > 0:
                await asyncio.sleep(step.delay_seconds)
        return TypingResult(
            transport="playwright_keyboard",
            duration_ms=int((time.perf_counter() - start) * 1000),
            typo_count=plan.typo_count,
            used_correction=plan.used_correction,
            fallback_char_count=fallback_char_count,
        )


class _CGPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


class _CoreGraphicsBridge:
    """Tiny ctypes bridge for the subset of CoreGraphics we need."""

    kCGHIDEventTap = 0
    kCGEventLeftMouseDown = 1
    kCGEventLeftMouseUp = 2
    kCGEventMouseMoved = 5
    kCGEventKeyDown = 10
    kCGEventKeyUp = 11
    kCGMouseButtonLeft = 0
    kCGScrollEventUnitPixel = 1

    def __init__(self):
        if sys.platform != "darwin":
            raise RuntimeError("Away mode currently supports macOS only.")

        app_services = ctypes.util.find_library("ApplicationServices")
        core_foundation = ctypes.util.find_library("CoreFoundation")
        if not app_services or not core_foundation:
            raise RuntimeError("CoreGraphics frameworks not available on this machine.")

        self._cg = ctypes.CDLL(app_services)
        self._cf = ctypes.CDLL(core_foundation)

        self._cg.CGEventCreate.argtypes = [ctypes.c_void_p]
        self._cg.CGEventCreate.restype = ctypes.c_void_p

        self._cg.CGEventGetLocation.argtypes = [ctypes.c_void_p]
        self._cg.CGEventGetLocation.restype = _CGPoint

        self._cg.CGEventCreateMouseEvent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            _CGPoint,
            ctypes.c_uint32,
        ]
        self._cg.CGEventCreateMouseEvent.restype = ctypes.c_void_p

        self._cg.CGEventCreateKeyboardEvent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint16,
            ctypes.c_bool,
        ]
        self._cg.CGEventCreateKeyboardEvent.restype = ctypes.c_void_p

        self._cg.CGEventCreateScrollWheelEvent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_int32,
        ]
        self._cg.CGEventCreateScrollWheelEvent.restype = ctypes.c_void_p

        self._cg.CGEventPost.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
        self._cg.CGEventPost.restype = None

        self._cf.CFRelease.argtypes = [ctypes.c_void_p]
        self._cf.CFRelease.restype = None

    def _release(self, ref: int | None) -> None:
        if ref:
            self._cf.CFRelease(ref)

    def current_location(self) -> tuple[float, float]:
        event = self._cg.CGEventCreate(None)
        if not event:
            return 0.0, 0.0
        try:
            point = self._cg.CGEventGetLocation(event)
            return point.x, point.y
        finally:
            self._release(event)

    def post_mouse(self, event_type: int, x: float, y: float) -> None:
        event = self._cg.CGEventCreateMouseEvent(
            None,
            event_type,
            _CGPoint(x, y),
            self.kCGMouseButtonLeft,
        )
        if not event:
            raise RuntimeError("Failed to create CoreGraphics mouse event.")
        try:
            self._cg.CGEventPost(self.kCGHIDEventTap, event)
        finally:
            self._release(event)

    def post_key(self, key_code: int, is_down: bool) -> None:
        event = self._cg.CGEventCreateKeyboardEvent(None, key_code, is_down)
        if not event:
            raise RuntimeError("Failed to create CoreGraphics keyboard event.")
        try:
            self._cg.CGEventPost(self.kCGHIDEventTap, event)
        finally:
            self._release(event)

    def post_scroll(self, delta_y: int) -> None:
        # CoreGraphics uses positive values for up and negative for down. Our
        # browser helpers use positive delta_y for down to match Playwright.
        event = self._cg.CGEventCreateScrollWheelEvent(
            None,
            self.kCGScrollEventUnitPixel,
            1,
            int(-delta_y),
        )
        if not event:
            raise RuntimeError("Failed to create CoreGraphics scroll event.")
        try:
            self._cg.CGEventPost(self.kCGHIDEventTap, event)
        finally:
            self._release(event)


class AwayInputBackend(InputBackend):
    """Real macOS takeover mode using CoreGraphics events."""

    mode = "away"
    KEY_CODES = {
        "Enter": 36,
        "Escape": 53,
        "Tab": 48,
        "Space": 49,
        "Backspace": 51,
        "Meta": 55,
        "Shift": 56,
    }
    CHAR_KEY_CODES = {
        "a": (0, False),
        "s": (1, False),
        "d": (2, False),
        "f": (3, False),
        "h": (4, False),
        "g": (5, False),
        "z": (6, False),
        "x": (7, False),
        "c": (8, False),
        "v": (9, False),
        "b": (11, False),
        "q": (12, False),
        "w": (13, False),
        "e": (14, False),
        "r": (15, False),
        "y": (16, False),
        "t": (17, False),
        "1": (18, False),
        "2": (19, False),
        "3": (20, False),
        "4": (21, False),
        "6": (22, False),
        "5": (23, False),
        "=": (24, False),
        "9": (25, False),
        "7": (26, False),
        "-": (27, False),
        "8": (28, False),
        "0": (29, False),
        "]": (30, False),
        "o": (31, False),
        "u": (32, False),
        "[": (33, False),
        "i": (34, False),
        "p": (35, False),
        "l": (37, False),
        "j": (38, False),
        "'": (39, False),
        "k": (40, False),
        ";": (41, False),
        "\\": (42, False),
        ",": (43, False),
        "/": (44, False),
        "n": (45, False),
        "m": (46, False),
        ".": (47, False),
        " ": (49, False),
        "`": (50, False),
        '"': (39, True),
        "(": (25, True),
        ")": (29, True),
        ":": (41, True),
        "+": (24, True),
        "&": (26, True),
    }

    def __init__(self):
        self._cg = _CoreGraphicsBridge()
        self.status_label = "CoreGraphics takeover active"

    async def initialize(self, page: "Page") -> None:
        return None

    async def click_selector(self, page: "Page", selector: str) -> bool:
        locator = page.locator(selector).first
        return await self.click_locator(page, locator)

    async def move_selector(self, page: "Page", selector: str) -> bool:
        locator = page.locator(selector).first
        point = await self._locator_screen_point(page, locator)
        if point is None:
            return False
        await self._move_mouse(point[0], point[1], channel="os_pointer")
        return True

    async def click_locator(
        self,
        page: "Page",
        locator: "Locator",
        *,
        about_to_commit: Callable[[], None] | None = None,
    ) -> bool:
        point = await self._locator_screen_point(page, locator)
        if point is None:
            return False

        await self._move_mouse(point[0], point[1], channel="os_pointer")
        await asyncio.sleep(
            human_delay_correlated(random.uniform(0.04, 0.14), channel="os_pointer")
        )
        if about_to_commit is not None:
            about_to_commit()
        self._cg.post_mouse(_CoreGraphicsBridge.kCGEventLeftMouseDown, point[0], point[1])
        await asyncio.sleep(
            human_delay_correlated(random.uniform(0.03, 0.09), channel="os_pointer")
        )
        self._cg.post_mouse(_CoreGraphicsBridge.kCGEventLeftMouseUp, point[0], point[1])
        return True

    async def scroll(self, page: "Page", delta_y: int, *, channel: str = "scroll") -> int:
        if delta_y == 0:
            return 0

        direction = 1 if delta_y > 0 else -1
        remaining = abs(delta_y)
        events = 0
        while remaining > 0:
            chunk = min(remaining, max(20, int(random.lognormvariate(math.log(85), 0.35))))
            self._cg.post_scroll(chunk * direction)
            events += 1
            remaining -= chunk
            await asyncio.sleep(
                human_delay_correlated(0.03, spread=0.45, channel=f"{channel}_micro")
            )
        return events

    async def press_key(self, page: "Page", key: str) -> bool:
        key_code = self.KEY_CODES.get(key)
        if key_code is None:
            return False
        self._cg.post_key(key_code, True)
        await asyncio.sleep(
            human_delay_correlated(random.uniform(0.03, 0.08), channel="os_key")
        )
        self._cg.post_key(key_code, False)
        return True

    async def press_combo(self, page: "Page", combo: str) -> bool:
        parts = [part.strip() for part in combo.split("+") if part.strip()]
        if not parts:
            return False
        modifiers: list[int] = []
        main_key_code: int | None = None
        for part in parts:
            lowered = part.lower()
            if lowered in {"meta", "command", "cmd"}:
                modifiers.append(self.KEY_CODES["Meta"])
                continue
            if lowered == "shift":
                modifiers.append(self.KEY_CODES["Shift"])
                continue
            if len(part) == 1 and part.isalpha():
                char_spec = self.CHAR_KEY_CODES.get(part.lower())
                if char_spec is None:
                    return False
                main_key_code = char_spec[0]
                continue
            key_code = self.KEY_CODES.get(part)
            if key_code is None:
                return False
            main_key_code = key_code
        if main_key_code is None:
            return False
        self._press_with_modifiers(main_key_code, modifiers)
        return True

    async def type_text(
        self,
        page: "Page",
        locator: "Locator",
        text: str,
        *,
        plan: TypingPlan,
    ) -> TypingResult:
        start = time.perf_counter()
        fallback_char_count = 0
        for step in plan.steps:
            if step.kind == "pause":
                await asyncio.sleep(step.delay_seconds)
                continue
            if step.kind == "backspace":
                self._post_key_tap(self.KEY_CODES["Backspace"])
            elif step.kind == "char":
                if not self._type_char(step.value):
                    await page.keyboard.insert_text(step.value)
                    fallback_char_count += 1
            if step.delay_seconds > 0:
                await asyncio.sleep(step.delay_seconds)
        return TypingResult(
            transport="coregraphics_keyboard",
            duration_ms=int((time.perf_counter() - start) * 1000),
            typo_count=plan.typo_count,
            used_correction=plan.used_correction,
            fallback_char_count=fallback_char_count,
        )

    async def _locator_screen_point(
        self,
        page: "Page",
        locator: "Locator",
    ) -> tuple[float, float] | None:
        try:
            await locator.scroll_into_view_if_needed(timeout=3000)
        except Exception:
            pass

        box = await locator.bounding_box()
        if not box:
            return None

        # box coordinates are VIEWPORT-relative; screenX/screenY locate the
        # browser WINDOW, whose top edge sits above the viewport by the height
        # of the tab strip and omnibox. Adding the two directly (the behavior
        # until 2026-08-03) aimed every OS-level click and move about 80-90px
        # high, which is why away mode has never worked. Measure the offset from
        # the page rather than assuming a constant, since it varies with zoom,
        # the bookmarks bar, and any extension chrome.
        metrics = await page.evaluate("""() => ({
            screenX: window.screenX,
            screenY: window.screenY,
            outerWidth: window.outerWidth,
            outerHeight: window.outerHeight,
            innerWidth: window.innerWidth,
            innerHeight: window.innerHeight
        })""")
        chrome_height = max(
            0.0, float(metrics["outerHeight"]) - float(metrics["innerHeight"])
        )
        # Side chrome is symmetric when present at all, so half the horizontal
        # difference is the left border.
        chrome_left = max(
            0.0,
            (float(metrics["outerWidth"]) - float(metrics["innerWidth"])) / 2.0,
        )
        return (
            float(metrics["screenX"])
            + chrome_left
            + float(box["x"])
            + float(box["width"]) / 2.0,
            float(metrics["screenY"])
            + chrome_height
            + float(box["y"])
            + float(box["height"]) / 2.0,
        )

    async def _move_mouse(self, end_x: float, end_y: float, *, channel: str) -> None:
        start_x, start_y = self._cg.current_location()
        dx = end_x - start_x
        dy = end_y - start_y
        distance = math.hypot(dx, dy)

        if distance < 1.0:
            self._cg.post_mouse(_CoreGraphicsBridge.kCGEventMouseMoved, end_x, end_y)
            return

        steps = max(8, min(28, int(distance / random.uniform(18.0, 32.0))))
        duration = min(0.95, max(0.18, 0.10 + distance / 1400.0 + random.uniform(0.08, 0.18)))

        ctrl1 = (
            start_x + dx * random.uniform(0.18, 0.32) + random.uniform(-24.0, 24.0),
            start_y + dy * random.uniform(0.18, 0.32) + random.uniform(-24.0, 24.0),
        )
        ctrl2 = (
            start_x + dx * random.uniform(0.62, 0.82) + random.uniform(-24.0, 24.0),
            start_y + dy * random.uniform(0.62, 0.82) + random.uniform(-24.0, 24.0),
        )

        for step in range(1, steps + 1):
            t = step / steps
            inv = 1.0 - t
            x = (
                inv ** 3 * start_x
                + 3 * inv ** 2 * t * ctrl1[0]
                + 3 * inv * t ** 2 * ctrl2[0]
                + t ** 3 * end_x
            )
            y = (
                inv ** 3 * start_y
                + 3 * inv ** 2 * t * ctrl1[1]
                + 3 * inv * t ** 2 * ctrl2[1]
                + t ** 3 * end_y
            )
            self._cg.post_mouse(_CoreGraphicsBridge.kCGEventMouseMoved, x, y)
            await asyncio.sleep(
                human_delay_correlated(
                    max(0.004, duration / steps),
                    spread=0.35,
                    channel=f"{channel}_micro",
                )
            )

    def _post_key_tap(self, key_code: int, *, ch: str = "a") -> None:
        self._cg.post_key(key_code, True)
        if config.LINKEDIN_TYPING_DWELL_ENABLED:
            time.sleep(_sample_key_dwell(ch))
        self._cg.post_key(key_code, False)

    def _press_with_modifiers(self, key_code: int, modifiers: list[int]) -> None:
        for modifier in modifiers:
            self._cg.post_key(modifier, True)
        self._cg.post_key(key_code, True)
        self._cg.post_key(key_code, False)
        for modifier in reversed(modifiers):
            self._cg.post_key(modifier, False)

    def _type_char(self, ch: str) -> bool:
        if len(ch) != 1:
            return False
        if ch.isalpha():
            spec = self.CHAR_KEY_CODES.get(ch.lower())
            if spec is None:
                return False
            key_code, _ = spec
            modifiers = [self.KEY_CODES["Shift"]] if ch.isupper() else []
            self._press_with_modifiers(key_code, modifiers)
            return True
        spec = self.CHAR_KEY_CODES.get(ch)
        if spec is None:
            return False
        key_code, needs_shift = spec
        modifiers = [self.KEY_CODES["Shift"]] if needs_shift else []
        self._press_with_modifiers(key_code, modifiers)
        return True


def create_input_backend(mode: str | None) -> InputBackend:
    normalized = normalize_input_mode(mode)
    if normalized == "concurrent":
        return ConcurrentInputBackend()
    return AwayInputBackend()
