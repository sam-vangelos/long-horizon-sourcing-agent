import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from github.observability.console import mask_email
from github.observability.observer import SessionObserver


@pytest.mark.parametrize(
    ("email", "expected"),
    [
        ("jane.doe@example.com", "j***@example.com"),
        ("j@example.com", "j***@example.com"),
        ("", ""),
        (None, None),
        ("not-an-email", "not-an-email"),
    ],
)
def test_mask_email(email, expected):
    assert mask_email(email) == expected


def test_observer_save_masks_email_on_console(tmp_path, capsys):
    observer = SessionObserver("session", tmp_path, MagicMock())
    observer.console._ts = lambda: "12:34"
    observer.graph = MagicMock()
    observer.candidates = MagicMock()
    observer.metrics = MagicMock()
    candidate = SimpleNamespace(
        contact=SimpleNamespace(emails=["jane.doe@example.com"]),
        user=SimpleNamespace(name="Jane Doe"),
    )
    decision = SimpleNamespace(confidence=0.95, decision="SAVE", path="maintainer")
    query = SimpleNamespace(id=1)

    observer.on_save("janedoe", candidate, decision, query)

    output = capsys.readouterr().out
    assert "j***@example.com" in output
    assert "jane.doe@example.com" not in output
    assert not re.search(r"\b[\w.+-]+@[\w-]+\.", output)
