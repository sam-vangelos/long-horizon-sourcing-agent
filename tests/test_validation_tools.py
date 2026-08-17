from tools.check_repo_hygiene import is_incidental_tracked_path
from tools.run_validation import VALIDATION_PROFILES, build_pytest_command


def test_build_pytest_command_for_default_profile_excludes_dataset_band():
    command = build_pytest_command("default")
    assert command[:3] == command[:1] + ["-m", "pytest"]
    assert "--ignore=tests/test_fde_iteration_dataset.py" in command


def test_validation_profiles_document_current_non_blocking_status():
    assert VALIDATION_PROFILES["default"].known_non_blocking == ()
    assert VALIDATION_PROFILES["full"].known_excluded == ()


def test_is_incidental_tracked_path_flags_expected_junk():
    assert is_incidental_tracked_path(".DS_Store") is True
    assert is_incidental_tracked_path("docs/__pycache__/thing.pyc") is True
    assert is_incidental_tracked_path("config/brief-role.bak-20260414-120000.json") is True
    assert is_incidental_tracked_path("config/brief-role.json") is False
