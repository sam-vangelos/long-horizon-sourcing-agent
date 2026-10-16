"""The new loop and old day-cycle import path share exact exception classes."""
import importlib
import pickle

import pytest


@pytest.mark.structural
@pytest.mark.parametrize("name", [
    "GeographyRegimeError", "PageRenderFailedError", "PanelRecoveryError",
    "TransientPaginationError", "SurfaceOwnershipLostError", "ProjectContextMismatchError",
])
def test_loop_errors_keep_public_identity_and_pickle_contract(name):
    declared = getattr(importlib.import_module("linkedin.evaluation_errors"), name)
    public = getattr(importlib.import_module("linkedin.orchestrator"), name)
    assert declared is public
    assert public.__module__ == "linkedin.orchestrator"
    original = public("recorded failure")
    restored = pickle.loads(pickle.dumps(original))
    assert type(restored) is public
    assert restored.args == original.args


@pytest.mark.structural
def test_facade_keeps_the_brief_loader_reexport():
    """tools/rejudge_review_flagged.py imports load_brief through the facade."""
    import linkedin.orchestrator as facade
    from shared.brief_loader import load_brief

    assert facade.load_brief is load_brief
