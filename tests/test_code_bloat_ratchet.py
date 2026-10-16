"""Offline detector fixtures and the content-keyed bloat ratchet."""
from pathlib import Path

import pytest


@pytest.mark.parametrize('kind,files', [
    ('carried_history', {'shared/a.py': '# Sam ruled on 2026-08-01; kept for a revert.\nVALUE = 1\n'}),
    ('narration', {'shared/a.py': 'def load_items():\n    """Load items."""\n    return []\n'}),
    ('stale_references', {'shared/a.py': '# See shared/missing.py:999 for the contract.\nVALUE = 1\n'}),
    ('duplicated_prose', {p: '# One shared paragraph explains the setup.\n# A second line describes its configuration.\n# The third line completes this paragraph.\nVALUE = 1\n' for p in ('shared/a.py', 'shared/b.py')}),
    ('defensive_layering', {'shared/a.py': 'from dataclasses import dataclass\n@dataclass\nclass Item:\n    name: str\ndef read(item: Item):\n    return getattr(item, "name", None)\n'}),
    ('dead_choice', {'shared/config.py': 'import os\nFLAG = os.getenv("FLAG", "true").lower() == "true"\n', 'tools/launch_a.sh': 'export FLAG=true\n', 'tools/launch_b.sh': 'export FLAG=true\n'}),
    ('duplicate_helpers', {p: 'def normalize(value):\n    value = value.strip()\n    return value.lower()\n' for p in ('shared/a.py', 'shared/b.py')}),
    ('test_theater', {'tests/test_a.py': 'def test_value():\n    value = 3\n    assert value == 3\n'}),
    ('essay_scripts', {'tools/launch_a.sh': '# Explains a setting.\n# Explains another setting.\n# Explains yet another setting.\nexport FLAG=true\n'}),
])
def test_planted_violation(tmp_path, kind, files):
    for name, source in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    from tools.lint_code_bloat import scan
    assert any(f['class'] == kind for f in scan(tmp_path)['findings'])


def test_ratchet():
    from tools.lint_code_bloat import ROOT, baseline_keys, scan
    current = baseline_keys(scan(ROOT))
    baseline = set((ROOT / 'tools/code_bloat_baseline.txt').read_text().splitlines())
    assert not current - baseline, f'new findings: {sorted(current - baseline)}'
    assert not baseline - current, f'stale entries: {sorted(baseline - current)}'


def test_keys_survive_line_movement(tmp_path):
    from tools.lint_code_bloat import scan
    path = tmp_path / 'shared/a.py'
    path.parent.mkdir()
    source = '# Sam ruled on 2026-08-01.\nVALUE = 1\n'
    path.write_text(source)
    before = {f['key'] for f in scan(tmp_path)['findings']}
    path.write_text('\n\n' + source)
    assert before == {f['key'] for f in scan(tmp_path)['findings']}


def test_scope_and_string_data_are_not_comments(tmp_path):
    from tools.lint_code_bloat import scan
    files = {'shared/a.py': 'PROMPT = "Sam ruled on 2026-08-01"\n',
             'researcher/a.py': '# Sam ruled on 2026-08-01\n',
             'shared/runtime_state/researcher.py': '# Sam ruled on 2026-08-01\n',
             'tests/test_researcher_a.py': 'def test_a():\n    x = 1\n    assert x == 1\n'}
    for name, source in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    assert scan(tmp_path)['findings'] == []


def test_shell_embedded_comments_and_expect_are_static(tmp_path):
    from tools.lint_code_bloat import scan
    (tmp_path / 'tools').mkdir()
    (tmp_path / 'shared').mkdir()
    (tmp_path / 'shared/config.py').write_text('FLAG = _env_flag("FLAG", "false")\n')
    (tmp_path / 'tools/launch_a.sh').write_text("export FLAG=false\npython - <<'PYEOF'\n# Sam ruled on 2026-08-01\nEXPECT = {'FLAG': False}\nraise RuntimeError('never execute')\nPYEOF\n")
    report = scan(tmp_path)
    assert report['knobs'][0]['expects'] == {'tools/launch_a.sh': False}
    assert report['files']['tools/launch_a.sh']['comment_lines'] == 1
    assert any(f['class'] == 'carried_history' for f in report['findings'])


def test_cli_rejects_new_and_stale_entries(tmp_path, capsys):
    from tools.lint_code_bloat import main
    (tmp_path / 'shared').mkdir()
    (tmp_path / 'tools').mkdir()
    path = tmp_path / 'shared/a.py'
    path.write_text('# Sam ruled on 2026-08-01\nVALUE = 1\n')
    args = ['--root', str(tmp_path)]
    assert main(args) == 1
    assert main([*args, '--baseline']) == 0
    assert main(args) == 0
    path.write_text('VALUE = 1\n')
    assert main(args) == 1
    assert main([*args, '--baseline']) == 0


def test_equivalence_accepts_comments_and_rejects_behavior(tmp_path):
    from tools.check_code_bloat_equivalence import check
    check('VALUE = 1\n', '# Contract.\nVALUE = 1\n', 'settings.py', tmp_path)
    with pytest.raises(AssertionError, match='AST changed'):
        check('VALUE = 1\n', 'VALUE = 2\n', 'settings.py', tmp_path)
    shell = "export FLAG=false\npython - <<'PYEOF'\nEXPECT = {'FLAG': False}\nPYEOF\n"
    check(shell, '# Contract.\n' + shell, 'tools/launch_ciso.sh', tmp_path)
    with pytest.raises(AssertionError, match='text changed'):
        check(shell, shell.replace('FLAG=false', 'FLAG=true'), 'tools/launch_ciso.sh', tmp_path)


@pytest.mark.parametrize('expression', ['value or []', 'value or {}'])
def test_empty_container_leads(tmp_path, expression):
    from tools.lint_code_bloat import scan
    (tmp_path / 'shared').mkdir()
    (tmp_path / 'shared/a.py').write_text(f'def read(value):\n    return {expression}\n')
    matches = [f for f in scan(tmp_path)['findings'] if f['class'] == 'defensive_layering']
    assert len(matches) == 1
    assert matches[0]['score'] == 0


def test_equivalence_accepts_other_python_modules_and_relative_imports(tmp_path):
    from tools.check_code_bloat_equivalence import check
    package = tmp_path / 'sample_package'
    package.mkdir()
    (package / '__init__.py').write_text('')
    (package / 'values.py').write_text('VALUE = 7\n')
    source = 'from .values import VALUE\nfrom dataclasses import dataclass\n@dataclass\nclass Entry:\n    value: int\nENTRY = Entry(VALUE)\n'
    check(source, '# Required contract.\n' + source, 'sample_package/settings.py', tmp_path)


def test_equivalence_rejects_docstring_derived_public_value_change(tmp_path):
    from tools.check_code_bloat_equivalence import check
    with pytest.raises(AssertionError, match='public attribute'):
        check('"""one"""\nVALUE = __doc__\n', '"""two"""\nVALUE = __doc__\n', 'settings.py', tmp_path)


def test_equivalence_import_cannot_start_network_or_subprocess(tmp_path):
    from tools.check_code_bloat_equivalence import check
    import subprocess
    for source in ('import socket\nsocket.create_connection(("127.0.0.1", 9222))\n',
                   'import subprocess\nsubprocess.run(["echo", "forbidden"])\n'):
        with pytest.raises(subprocess.CalledProcessError):
            check(source, '# Contract.\n' + source, 'settings.py', tmp_path)


def test_shell_proof_allows_absent_expect_but_preserves_quoted_and_heredoc_data(tmp_path):
    from tools.check_code_bloat_equivalence import check
    shell = '#!/bin/bash\nexport FLAG=false\necho "line one\n# data\nline three"\ncat <<EOF\n# data\nEOF\n'
    check(shell, '# Contract.\n' + shell, 'tools/other.sh', tmp_path)
    with pytest.raises(AssertionError):
        check(shell, shell.replace('# data', '# changed'), 'tools/other.sh', tmp_path)


@pytest.mark.parametrize('definition', ['def read(value=__doc__):\n    return value\n',
                                         'class Entry:\n    value = __doc__\n'])
def test_equivalence_checks_docstring_derived_callable_state(tmp_path, definition):
    from tools.check_code_bloat_equivalence import check
    with pytest.raises(AssertionError, match='public attribute'):
        check('"""one"""\n' + definition, '"""two"""\n' + definition, 'settings.py', tmp_path)


def test_equivalence_import_cannot_write_to_the_repository(tmp_path):
    from tools.check_code_bloat_equivalence import check
    import subprocess
    source = f'from pathlib import Path\nPath({str(tmp_path / "unexpected.txt")!r}).write_text("no")\n'
    with pytest.raises(subprocess.CalledProcessError):
        check(source, '# Contract.\n' + source, 'settings.py', tmp_path)
    assert not (tmp_path / 'unexpected.txt').exists()


def test_equivalence_allows_temporary_import_output(tmp_path):
    from tools.check_code_bloat_equivalence import check
    source = 'from pathlib import Path\nDESTINATION = Path.home() / "allowed.txt"\nDESTINATION.write_text("same")\n'
    check(source, '# Contract.\n' + source, 'settings.py', tmp_path)
    with pytest.raises(ValueError, match='Python or shell'):
        check('', '', 'settings.txt', tmp_path)


def test_equivalence_handles_empty_docstrings(tmp_path):
    from tools.check_code_bloat_equivalence import check
    check('"""Settings."""\nVALUE = 1\n', '""""""\nVALUE = 1\n', 'settings.py', tmp_path)


def test_equivalence_checks_values_read_from_the_module_file(tmp_path):
    from tools.check_code_bloat_equivalence import check
    before = 'from pathlib import Path\nSOURCE = Path(__file__).read_text()\n'
    after = '# Changed documentation.\n' + before
    (tmp_path / 'settings.py').write_text(after)
    with pytest.raises(AssertionError, match='public attribute'):
        check(before, after, 'settings.py', tmp_path)
