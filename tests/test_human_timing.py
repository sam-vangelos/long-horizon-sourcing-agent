"""Tests for correlated timing streams."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.human_timing import human_delay_correlated, reset_human_timing


def test_human_delay_correlated_stays_within_bounds():
    reset_human_timing()
    random.seed(42)

    for _ in range(100):
        delay = human_delay_correlated(3.0, channel="bounds")
        assert 0.9 <= delay <= 12.0


def test_human_delay_correlated_channels_are_independent():
    reset_human_timing()
    random.seed(7)

    first_a = human_delay_correlated(2.0, channel="a")
    second_a = human_delay_correlated(2.0, channel="a")

    reset_human_timing()
    random.seed(7)

    first_a_replayed = human_delay_correlated(2.0, channel="a")
    first_b = human_delay_correlated(2.0, channel="b")

    assert first_a == first_a_replayed
    assert second_a != first_b
