#!/usr/bin/env python3
"""Verify comment-only changes against a git revision without running a launcher."""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def executable_ast(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and ast.get_docstring(node) is not None:
            node.body = node.body[1:]
    return ast.dump(tree, include_attributes=False)


def public_values(source, root, env, file="shared/config.py"):
    snapshot = Path(env['HOME']).parent / 'tree'
    for directory in (snapshot, Path(env['HOME'])):
        if directory.exists():
            shutil.rmtree(directory)
    Path(env['HOME']).mkdir()
    def mirror(original, target, parts):
        target.mkdir()
        if original.exists():
            for child in original.iterdir():
                if child.name not in {parts[0], '__pycache__'}:
                    (target / child.name).symlink_to(child, target_is_directory=child.is_dir())
        if len(parts) > 1:
            mirror(original / parts[0], target / parts[0], parts[1:])
    mirror(Path(root), snapshot, Path(file).parts)
    (snapshot / file).write_text(source)
    code = r'''import dataclasses, enum, importlib
import json, logging, os, pickle, re, sys, types
from pathlib import Path
root, relative, temporary = map(Path, sys.argv[1:])
temporary = temporary.resolve()
path = root / relative
parts = relative.with_suffix("").parts
name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
sys.path.insert(0, str(root))
def audit(event, args):
    if event.startswith(("subprocess.", "os.exec", "os.spawn")) or event in {"socket.connect", "socket.getaddrinfo", "socket.bind", "socket.sendto", "os.system", "os.posix_spawn", "os.fork", "pty.spawn"}:
        raise RuntimeError("offline import forbids " + event)
    if event in {"os.mkdir", "os.remove", "os.rmdir", "os.rename", "os.link", "os.symlink", "os.truncate"}:
        paths = args[:2] if event in {"os.rename", "os.link", "os.symlink"} else args[:1]
        if any(not Path(p).resolve().is_relative_to(temporary) for p in paths if not isinstance(p, int)):
            raise RuntimeError("offline import forbids filesystem mutation outside its temporary directory")
    if event == "sqlite3.connect" and str(args[0]) != ":memory:" and not Path(args[0]).resolve().is_relative_to(temporary):
        raise RuntimeError("offline import forbids opening a non-temporary database")
    if event == "open":
        filename, mode, flags = args
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        if writing and not isinstance(filename, int) and not Path(filename).resolve().is_relative_to(temporary):
            raise RuntimeError("offline import forbids writes outside its temporary directory")
sys.addaudithook(audit)
module = importlib.import_module(name)
active = set()
def stable(value):
    kind = type(value).__module__ + "." + type(value).__qualname__
    if value is None or isinstance(value, (str, int, float, bool)):
        return [kind, value]
    if isinstance(value, bytes):
        return [kind, value.hex()]
    if isinstance(value, types.ModuleType):
        return ["module", value.__name__]
    if isinstance(value, Path):
        return [kind, str(value)]
    if isinstance(value, enum.Enum):
        return [kind, value.name, stable(value.value)]
    if isinstance(value, re.Pattern):
        return [kind, value.pattern, value.flags]
    if isinstance(value, logging.Logger):
        return [kind, value.name, value.level, value.disabled, value.propagate]
    identity = id(value)
    if identity in active:
        return ["cycle", kind]
    active.add(identity)
    try:
        if isinstance(value, (staticmethod, classmethod)):
            return [kind, stable(value.__func__)]
        if isinstance(value, property):
            return [kind, stable(value.fget), stable(value.fset), stable(value.fdel)]
        if callable(value) and hasattr(value, "__module__") and hasattr(value, "__qualname__"):
            label = ["callable", value.__module__, value.__qualname__]
            if value.__module__ == name:
                if isinstance(value, type):
                    label.append({k: stable(v) for k, v in sorted(vars(value).items()) if not k.startswith("_")})
                elif isinstance(value, types.FunctionType):
                    label.extend([stable(value.__defaults__), stable(value.__kwdefaults__)])
                elif hasattr(value, "__dict__"):
                    label.append(stable(vars(value)))
            return label
        if isinstance(value, dict):
            return [kind, sorted([[stable(k), stable(v)] for k, v in value.items()], key=repr)]
        if isinstance(value, (list, tuple, set, frozenset)):
            values = [stable(v) for v in value]
            return [kind, sorted(values, key=repr) if isinstance(value, (set, frozenset)) else values]
        if dataclasses.is_dataclass(value):
            return [kind, {f.name: stable(getattr(value, f.name)) for f in dataclasses.fields(value)}]
        if hasattr(value, "__dict__"):
            return [kind, {k: stable(v) for k, v in sorted(vars(value).items())}]
        return [kind, pickle.dumps(value, protocol=4).hex()]
    finally:
        active.remove(identity)
print(json.dumps({k: stable(v) for k, v in vars(module).items() if not k.startswith("_")}, sort_keys=True))
'''
    result = subprocess.run([sys.executable, '-c', code, str(snapshot), file, str(snapshot.parent)],
                            text=True, capture_output=True, check=True, cwd=snapshot, env=env, timeout=60)
    return json.loads(result.stdout.splitlines()[-1])


def expect_dicts(source):
    result = []
    for match in re.finditer(r'<<-?\s*[\'\"]?(\w+)[\'\"]?[^\n]*\n(.*?)\n\1\s*(?:\n|$)', source, re.S):
        try:
            tree = ast.parse(match[2])
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'EXPECT' for t in node.targets):
                result.append(ast.literal_eval(node.value))
    return result


def shell_literals(source):
    bodies = []
    pattern = r"(?m)^([^\n]*?)<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n(.*?)\n\2\s*(?:\n|$)"
    def replace(match):
        command, delimiter, body = match.groups()
        if re.search(r'python|\$[\{]?(?:PY|PYTHON)[\}]?(?:["\s]|$)', command):
            import io
            import tokenize
            tokens = tokenize.generate_tokens(io.StringIO(body).readline)
            bodies.append([(t.type, t.string) for t in tokens if t.type not in {tokenize.COMMENT, tokenize.NL}])
        else:
            bodies.append(body)
        return command + '<<' + delimiter + '\n' + delimiter + '\n'
    outer = re.sub(pattern, replace, source, flags=re.S)
    return list(shlex.shlex(outer, posix=True)), bodies


def check(before, after, file, root):
    if Path(file).is_absolute() or '..' in Path(file).parts:
        raise ValueError('expected a repository-relative file')
    if Path(file).suffix == '.py':
        assert executable_ast(before) == executable_ast(after), 'executable Python AST changed'
        with tempfile.TemporaryDirectory(prefix='bloat-config-') as temp:
            home = str(Path(temp) / 'home')
            env = {'PATH': os.defpath, 'HOME': home, 'PYTHONDONTWRITEBYTECODE': '1',
                   'CLORIS_USER_DATA_DIR': home, 'TMPDIR': home, 'PYTHON_DOTENV_DISABLED': '1'}
            old, new = (public_values(source, root, env, file) for source in (before, after))
        assert old == new, 'public attribute values changed'
        print(f'{file}: {len(old)} public attributes identical; executable AST identical; scrubbed imports passed')
    elif Path(file).suffix == '.sh':
        code = lambda s: [l for l in s.splitlines() if l.strip() and not l.lstrip().startswith('#')]
        assert code(before) == code(after), 'non-comment shell/heredoc text changed'
        assert shell_literals(before) == shell_literals(after), 'quoted or heredoc data changed'
        exports = lambda s: sorted(l.strip() for l in s.splitlines() if l.lstrip().startswith('export '))
        assert exports(before) == exports(after), 'exports changed'
        old, new = expect_dicts(before), expect_dicts(after)
        assert old == new, 'EXPECT dictionaries changed'
        recovery = lambda s: [l for l in s.splitlines() if 'LINKEDIN_PAGE_LOCAL_RECOVERY_ENABLED' in l]
        assert recovery(before) == recovery(after), 'page-local recovery changed'
        for source in (before, after):
            subprocess.run(['bash', '-n'], input=source, text=True, check=True)
        print(f'{file}: bash -n passed twice; {len(exports(after))} exports identical; {sum(map(len,new))} EXPECT entries identical; page-local recovery untouched; non-comment text identical')

    else:
        raise ValueError('expected a Python or shell file')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', required=True)
    parser.add_argument('file', nargs='+', help='repository-relative Python or shell files')
    args = parser.parse_args()
    for file in args.file:
        path = ROOT / file
        if not path.resolve().is_relative_to(ROOT) or path.suffix not in {'.py', '.sh'}:
            parser.error('expected a Python or shell file inside this repository')
        before = subprocess.check_output(['git', 'show', f'{args.before}:{file}'], cwd=ROOT, text=True)
        try:
            check(before, path.read_text(), file, ROOT)
        except subprocess.CalledProcessError as exc:
            parser.exit(1, f'{file}: import proof failed\n{exc.stderr}')


if __name__ == '__main__':
    main()
