"""Scoped adapters for effects formerly imported once by EvaluationLoop."""
from contextlib import ExitStack, contextmanager
from unittest.mock import patch


@contextmanager
def patch_evaluation_delay(*args, **kwargs):
    """Keep one delay spy across the concrete execution owners."""
    with ExitStack() as stack:
        spy = stack.enter_context(patch("linkedin.candidate_review.human_delay_correlated", *args, **kwargs))
        for module in ("page_review", "search_surface", "string_driver"):
            stack.enter_context(patch(f"linkedin.{module}.human_delay_correlated", new=spy))
        yield spy
