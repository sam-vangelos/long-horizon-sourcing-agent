"""tools/flag_ledger.py: the parser, and the four ways a flag offends.

Everything here runs against a planted config, never the real one — the real
one is the thing the ledger is pointed at in `make hygiene`, and a test that
asserted its current contents would just restate it.
"""
from __future__ import annotations

import datetime as dt

import pytest

from tools import flag_ledger


FIXTURE_CONFIG = '''\
# Clean: dated gate, still in the future.
# gate: CLO-1 by 2099-01-01
ALPHA_ENABLED: bool = _env_flag("ALPHA_ENABLED", "false")
# A deliberate switch owes no date.
# gate: by-design — per-campaign configuration
BETA_ENABLED: bool = _env_flag(
    "BETA_ENABLED", "false"
)
# Nothing here says what would make it safe to turn on.
GAMMA_ENABLED: bool = _env_flag("GAMMA_ENABLED", "false")
# gate: CLO-9 by 2020-01-01
DELTA_ENABLED: bool = _env_flag("DELTA_ENABLED", "false")

# A blank line BELOW ends this block, so it is not EPSILON's comment and the
# gate in it does not count for anything.
# gate: by-design — belongs to no flag

EPSILON_ENABLED: bool = _env_flag("EPSILON_ENABLED", "false")
# gate: by-design — kill switch
ZETA_ENABLED: bool = _env_flag("ZETA_ENABLED", "true")
'''

READER = """\
from shared import config

if config.ALPHA_ENABLED and config.BETA_ENABLED and config.GAMMA_ENABLED:
    pass
if config.DELTA_ENABLED or config.EPSILON_ENABLED:
    pass
"""


@pytest.fixture
def root(tmp_path):
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared" / "config.py").write_text(FIXTURE_CONFIG)
    (tmp_path / "linkedin").mkdir()
    (tmp_path / "linkedin" / "reader.py").write_text(READER)
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "launch_planted.sh").write_text(
        "#!/bin/bash\nexport EPSILON_ENABLED=true\nexport GAMMA_ENABLED=false\n"
    )
    return tmp_path


def _by_name(root):
    return {flag.name: flag for flag in flag_ledger.build(root)}


def test_parser_reads_defaults_and_the_comment_block_above(root):
    flags = _by_name(root)
    assert set(flags) == {
        "ALPHA_ENABLED", "BETA_ENABLED", "GAMMA_ENABLED",
        "DELTA_ENABLED", "EPSILON_ENABLED", "ZETA_ENABLED",
    }
    assert flags["ZETA_ENABLED"].default_on is True
    assert flags["ALPHA_ENABLED"].default_on is False
    # A blank line ends the block, so EPSILON inherits neither the stray
    # comment above it nor the gate line inside it.
    assert flags["EPSILON_ENABLED"].comment == ""
    assert flags["EPSILON_ENABLED"].gate == ""


def test_a_multiline_env_flag_is_parsed_like_any_other(root):
    beta = _by_name(root)["BETA_ENABLED"]
    assert beta.default_on is False
    assert beta.gate == "by-design — per-campaign configuration"
    assert beta.readers


def test_a_launcher_export_is_what_makes_a_dark_flag_lit(root):
    flags = _by_name(root)
    assert flags["EPSILON_ENABLED"].launchers == ["launch_planted.sh"]
    # `export GAMMA_ENABLED=false` is not an enable.
    assert flags["GAMMA_ENABLED"].launchers == []


def test_missing_gate_expired_gate_and_dead_flag_are_the_offenders(root):
    found = flag_ledger.offenders(flag_ledger.build(root), dt.date(2026, 9, 20))
    named = {line.split(":")[0]: line for line in found}
    assert set(named) == {"GAMMA_ENABLED", "DELTA_ENABLED", "ZETA_ENABLED"}
    assert "names no gate" in named["GAMMA_ENABLED"]
    assert "gate expired 2020-01-01 (CLO-9)" in named["DELTA_ENABLED"]
    assert "dead flag" in named["ZETA_ENABLED"]
    # A dated gate that has not come due, a by-design gate, and a launcher
    # export each keep their flag off the list.
    assert not {"ALPHA_ENABLED", "BETA_ENABLED", "EPSILON_ENABLED"} & set(named)


def test_a_gate_only_expires_after_its_date(root):
    flags = flag_ledger.build(root)
    on_the_day = flag_ledger.offenders(flags, dt.date(2020, 1, 1))
    assert not any(line.startswith("DELTA_ENABLED") for line in on_the_day)
    assert any(
        line.startswith("DELTA_ENABLED")
        for line in flag_ledger.offenders(flags, dt.date(2020, 1, 2))
    )


def test_cli_exits_1_on_offenders_and_0_on_a_clean_tree(root, capsys):
    assert flag_ledger.main(["--root", str(root)]) == 1
    assert "[flag-ledger] GAMMA_ENABLED" in capsys.readouterr().out

    config = root / "shared" / "config.py"
    config.write_text(
        FIXTURE_CONFIG.replace(
            '# Nothing here says what would make it safe to turn on.',
            '# gate: CLO-2 by 2099-01-01',
        )
        .replace("# gate: CLO-9 by 2020-01-01", "# gate: CLO-9 by 2099-01-01")
        .replace('ZETA_ENABLED: bool = _env_flag("ZETA_ENABLED", "true")', "")
    )
    assert flag_ledger.main(["--root", str(root)]) == 0
    assert "flag ledger: 5 flags, 5 default-off, 0 offenders" in capsys.readouterr().out


def test_list_prints_every_flag_with_its_launchers_and_gate(root, capsys):
    flag_ledger.main(["--root", str(root), "--list"])
    out = capsys.readouterr().out
    assert "CLO-1 by 2099-01-01" in out
    assert "launch_planted.sh" in out
    assert "(no launcher)" in out
    assert "(no gate)" in out
