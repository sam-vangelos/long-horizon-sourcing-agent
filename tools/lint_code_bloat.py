#!/usr/bin/env python3
"""Offline bloat findings; --json reports leads, --baseline records calibrated rules."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import tokenize

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ('carried_history', 'narration', 'stale_references', 'duplicated_prose',
           'defensive_layering', 'dead_choice', 'duplicate_helpers', 'test_theater', 'essay_scripts')
# Calibrate against the recorded random draws in plans/code-bloat-audit.md.
RATCHET_CLASSES = {'carried_history', 'duplicated_prose', 'essay_scripts'}
HISTORY = re.compile(r'\b(?:20\d{2}-\d\d-\d\d|(?:CLO|ALC)-\d+|Sam|ruling|wave|slice|REVERTED|superseded)\b|kept for a revert|\b(?:previously|used to|formerly)\b', re.I)
REFERENCE = re.compile(r'(?<![\w/])((?:[\w.-]+/)+[\w.-]+\.(?:py|md|sh))(?::(\d+|[A-Za-z_]\w*(?:\.\w+)*))?')
EXCLUDED = ('cloris/', 'attic/', 'designer/', 'exec_search/', 'researcher/')


def excluded(path):
    return (path.startswith(EXCLUDED) or path == 'shared/runtime_state/researcher.py'
            or Path(path).name.startswith(('researcher-', 'test_researcher_')))


def words(text):
    return re.findall(r'[a-z0-9]+', re.sub(r'([a-z])([A-Z])', r'\1 \2', text).lower())


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def scopes(tree):
    result = []
    def visit(node, prefix=''):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f'{prefix}.{child.name}' if prefix else child.name
                result.append((child.lineno, child.end_lineno, name, child))
                visit(child, name)
            else:
                visit(child, prefix)
    visit(tree)
    return result


def symbol_at(ranges, line):
    matches = [r for r in ranges if r[0] <= line <= r[1]]
    return min(matches, key=lambda r: r[1] - r[0])[2] if matches else '<module>'


def prose_blocks(source, tree=None):
    lines = source.splitlines()
    comments, docs = set(), set()
    blocks = []
    if tree is not None:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        comment_tokens = [t for t in tokens if t.type == tokenize.COMMENT]
        current = []
        for token in comment_tokens:
            line = token.start[0]
            comments.add(line)
            if current and line != current[-1][0] + 1:
                blocks.append((current[0][0], current[-1][0], '\n'.join(t for _, t in current), 'comment'))
                current = []
            current.append((line, token.string.lstrip('#').strip()))
        if current:
            blocks.append((current[0][0], current[-1][0], '\n'.join(t for _, t in current), 'comment'))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and ast.get_docstring(node):
                doc = node.body[0]
                docs.update(range(doc.lineno, doc.end_lineno + 1))
                blocks.append((doc.lineno, doc.end_lineno, ast.get_docstring(node, clean=False), 'docstring'))
    else:
        # Shell comments only: here-doc payloads are code/data, not shell commentary.
        heredoc = None
        current = []
        for line, text in enumerate(lines, 1):
            if heredoc:
                if text.strip() == heredoc:
                    heredoc = None
                continue
            match = re.search(r'<<-?\s*[\'\"]?(\w+)[\'\"]?', text)
            if match:
                heredoc = match[1]
            if text.lstrip().startswith('#') and not text.startswith('#!'):
                comments.add(line)
                current.append((line, text.lstrip().lstrip('#').strip()))
            elif current:
                blocks.append((current[0][0], current[-1][0], '\n'.join(t for _, t in current), 'comment'))
                current = []
        if current:
            blocks.append((current[0][0], current[-1][0], '\n'.join(t for _, t in current), 'comment'))
    if tree is None:
        for match in re.finditer(r'<<-?\s*[\'\"]?(\w+)[\'\"]?[^\n]*\n(.*?)\n\1\s*(?:\n|$)', source, re.S):
            try:
                embedded = ast.parse(match[2])
            except SyntaxError:
                continue
            offset = source[:match.start(2)].count('\n')
            inner, inner_comments, inner_docs = prose_blocks(match[2], embedded)
            blocks.extend((a+offset, b+offset, t, k) for a, b, t, k in inner)
            comments.update(i+offset for i in inner_comments)
            docs.update(i+offset for i in inner_docs)
    return sorted(blocks), comments, docs


def scan(root=ROOT):
    root = Path(root)
    paths = set()
    for area in ('linkedin', 'shared', 'github', 'exa', 'market_intelligence', 'tests'):
        paths.update((root / area).rglob('*.py'))
    paths.update((root / 'tools').glob('*.py'))
    paths.update((root / 'tools').glob('launch_*.sh'))
    sources, trees, ranges, blocks_by_file, metrics = {}, {}, {}, {}, {}
    findings = {}
    def emit(kind, path, line, text, reason, cost=1, **extra):
        symbol = extra.pop('symbol', symbol_at(ranges.get(path, []), line))
        key = f'{path}:{symbol}:{kind}:{digest(text)}'
        findings[key] = dict(key=key, **{'class': kind}, file=path, symbol=symbol,
                             line=line, evidence=text, reason=reason, cost=cost, **extra)

    for path in sorted(paths):
        rel = path.relative_to(root).as_posix()
        if excluded(rel):
            continue
        source = path.read_text()
        sources[rel] = source
        tree = ast.parse(source, filename=rel) if path.suffix == '.py' else None
        trees[rel] = tree
        ranges[rel] = scopes(tree) if tree else []
        blocks, comments, docs = prose_blocks(source, tree)
        blocks_by_file[rel] = blocks
        lines = source.splitlines()
        metrics[rel] = dict(lines=len(lines), comment_lines=len(comments), docstring_lines=len(docs),
                            prose_lines=len(comments | docs),
                            code_lines=sum(bool(s.strip()) and i not in comments | docs for i, s in enumerate(lines, 1)),
                            long_comment_blocks=sum(end - start + 1 >= 8 for start, end, _, k in blocks if k == 'comment'))
        if rel.startswith('tests/'):
            continue
        for start, end, text, kind in blocks:
            if HISTORY.search(text):
                emit('carried_history', rel, start, text, 'history marker in comment/docstring', end - start + 1)
            if kind == 'docstring':
                symbol = symbol_at(ranges[rel], start).split('.')[-1]
                if words(text) == words(symbol):
                    emit('narration', rel, start, text, 'docstring repeats identifier', end - start + 1)
            elif start == end and end < len(lines):
                prose = set(words(text)) - {'the', 'a', 'an', 'to', 'of', 'return', 'set', 'get'}
                code = set(words(lines[end]))
                if len(prose) >= 2 and prose <= code:
                    emit('narration', rel, start, text, 'comment words repeat next statement', 1)
            for match in REFERENCE.finditer(text):
                target, anchor = match.groups()
                if excluded(target):
                    continue
                target_path = root / target
                if not target_path.is_file():
                    archived = sorted(p.relative_to(root).as_posix() for p in (root / 'plans/archive').rglob(Path(target).name)) if target.startswith('plans/') else []
                    emit('stale_references', rel, start, match[0], 'archived path' if archived else 'missing path', targets=archived)
                elif anchor:
                    target_source = target_path.read_text()
                    if anchor.isdigit():
                        if int(anchor) < 1 or int(anchor) > len(target_source.splitlines()):
                            emit('stale_references', rel, start, match[0], 'line outside file')
                        else:
                            emit('stale_references', rel, start, match[0], 'line anchor needs semantic review', target_line=target_source.splitlines()[int(anchor)-1])
                    elif target.endswith('.py'):
                        names = {r[2] for r in scopes(ast.parse(target_source))}
                        if anchor not in names:
                            emit('stale_references', rel, start, match[0], 'symbol absent')
        if rel.startswith('tools/') and 0 < metrics[rel]['code_lines'] < metrics[rel]['prose_lines']:
            emit('essay_scripts', rel, 1, rel, 'prose lines exceed nonblank code lines', metrics[rel]['prose_lines'])

    paragraphs = defaultdict(list)
    for rel, blocks in blocks_by_file.items():
        if rel.startswith('tests/'):
            continue
        for start, end, text, _ in blocks:
            offset = 0
            for paragraph in text.split('\n\n'):
                if len(paragraph.splitlines()) >= 3 and len(words(paragraph)) >= 15:
                    paragraphs[' '.join(words(paragraph))].append((rel, start + offset, paragraph))
                offset += len(paragraph.splitlines()) + 1
    doc_paragraphs = defaultdict(set)
    doc_metrics = {}
    for area in ('docs', 'plans'):
        for path in sorted((root / area).rglob('*.md')):
            rel = path.relative_to(root).as_posix()
            if excluded(rel):
                continue
            source = path.read_text()
            doc_metrics[rel] = len(source.splitlines())
            for paragraph in re.split(r'\n\s*\n', source):
                normalized = ' '.join(words(re.sub(r'^\s*#+\s?', '', paragraph, flags=re.M)))
                if normalized in paragraphs:
                    doc_paragraphs[normalized].add(rel)
    for normalized, copies in paragraphs.items():
        if len({p for p, _, _ in copies}) < 2:
            continue
        for rel, line, text in copies:
            emit('duplicated_prose', rel, line, text, 'paragraph repeated across files', len(text.splitlines()),
                 copies=sorted({p for p, _, _ in copies}), documents=sorted(doc_paragraphs[normalized]))

    helpers = defaultdict(list)
    config_reads = defaultdict(list)
    for rel, tree in trees.items():
        if tree is None:
            continue
        if rel.startswith('tests/'):
            for start, end, name, node in ranges[rel]:
                if not name.split('.')[-1].startswith('test_') or not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                source = '\n'.join(sources[rel].splitlines()[start-1:end])
                nodes = list(ast.walk(node))
                reads_text = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'read_text' for n in nodes)
                source_read = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ('getsource', 'get_source') for n in nodes) or (reads_text and (any(isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.endswith(('.py', '.sh')) for n in nodes) or any(isinstance(n, ast.Attribute) and n.attr == '__file__' for n in nodes)))
                constants = {n.targets[0].id: n.value.value for n in node.body if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)}
                tautology = any(isinstance(n, ast.Assert) and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name) and n.test.left.id in constants and len(n.test.ops) == 1 and isinstance(n.test.ops[0], ast.Eq) and isinstance(n.test.comparators[0], ast.Constant) and constants[n.test.left.id] == n.test.comparators[0].value for n in nodes)
                mock_assert = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr.startswith('assert_called') and any(isinstance(a, ast.Attribute) and a.attr == 'return_value' for a in ast.walk(n)) for n in nodes)
                if source_read or tautology or mock_assert:
                    emit('test_theater', rel, start, ast.dump(node, include_attributes=False), 'source-text assertion' if source_read else 'assigned literal asserted' if tautology else 'mock return asserted as argument', end - start + 1, source=source, score=3 if tautology else 2 if mock_assert else 1, symbol=name)
            continue
        declared = {node.name: {n.target.id for n in node.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)} for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == 'config':
                config_reads[n.attr].append(f'{rel}:{n.lineno}')
        for start, end, name, node in ranges[rel]:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = node.body[1:] if ast.get_docstring(node) else node.body
            if len(body) >= 2:
                # ponytail: exact AST bodies only; alpha-renaming can conflate unrelated contracts.
                normalized = ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)
                helpers[normalized].append((rel, start, end, name))
            annotations = {a.arg: ast.unparse(a.annotation) for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs) if a.annotation}
            for n in ast.walk(node):
                reason, score = None, 0
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ('getattr', 'hasattr') and len(n.args) >= 2:
                    obj, field = n.args[:2]
                    if isinstance(obj, ast.Name) and isinstance(field, ast.Constant):
                        known = field.value in declared.get(annotations.get(obj.id), set())
                        reason, score = ('declared field guarded', 2) if known else ('dynamic field guard; boundary unknown', 0)
                elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'isinstance' and len(n.args) == 2:
                    obj, typ = n.args
                    if isinstance(obj, ast.Name) and annotations.get(obj.id) == ast.unparse(typ):
                        reason, score = 'annotation repeated at runtime', 1
                elif isinstance(n, ast.BoolOp) and isinstance(n.op, ast.Or) and ((isinstance(n.values[-1], ast.Dict) and not n.values[-1].keys) or (isinstance(n.values[-1], ast.List) and not n.values[-1].elts)):
                    reason = 'empty-container fallback; nullability unknown'
                elif isinstance(n, ast.ExceptHandler) and isinstance(n.type, ast.Name) and n.type.id == 'Exception' and all(isinstance(b, ast.Pass) for b in n.body):
                    reason = 'broad exception swallowed; raising behavior unknown'
                if reason:
                    emit('defensive_layering', rel, n.lineno, ast.unparse(n), reason, score=score, symbol=name)
    for normalized, copies in helpers.items():
        if len({p for p, *_ in copies}) >= 2:
            for rel, start, end, name in copies:
                emit('duplicate_helpers', rel, start, normalized, 'identical body AST across modules', end - start + 1,
                     symbol=name, copies=[f'{p}:{n}' for p, _, _, n in copies])

    launchers = {p: s for p, s in sources.items() if p.startswith('tools/launch_') and p.endswith('.sh')}
    exports, expects = {}, {}
    for rel, source in launchers.items():
        exports[rel] = {m[1]: m[2].strip().strip('\'\"') for m in re.finditer(r'^\s*export\s+([A-Z][A-Z0-9_]*)=(.*?)(?:\s+#.*)?$', source, re.M)}
        expects[rel] = {}
        for match in re.finditer(r'<<-?\s*[\'\"]?(\w+)[\'\"]?[^\n]*\n(.*?)\n\1\s*(?:\n|$)', source, re.S):
            try:
                embedded = ast.parse(match[2])
            except SyntaxError:
                continue
            for n in ast.walk(embedded):
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'EXPECT' for t in n.targets):
                    try:
                        expects[rel].update(ast.literal_eval(n.value))
                    except (ValueError, TypeError):
                        pass  # Report unresolved expressions through the source evidence, never execute them.
    launcher_markers = defaultdict(list)
    for rel, source in launchers.items():
        lines = source.splitlines()
        for i, line in enumerate(lines):
            match = re.match(r'export ([A-Z][A-Z0-9_]*)=', line)
            if match and re.search(r'\b(?:inert|do not flip|kept live for a revert)\b', '\n'.join(lines[max(0, i-18):i]), re.I):
                launcher_markers[match[1]].append(f'{rel}:{i+1}')
    knobs = []
    tree = trees.get('shared/config.py')
    if tree:
        for n in tree.body:
            if not isinstance(n, (ast.Assign, ast.AnnAssign)):
                continue
            target = n.targets[0] if isinstance(n, ast.Assign) else n.target
            if not isinstance(target, ast.Name) or n.value is None:
                continue
            for call in ast.walk(n.value):
                if not (isinstance(call, ast.Call) and ((isinstance(call.func, ast.Attribute) and call.func.attr in ('getenv', 'get')) or (isinstance(call.func, ast.Name) and call.func.id in ('_optional', '_env_flag', '_env_choice'))) and len(call.args) >= 2 and isinstance(call.args[0], ast.Constant)):
                    continue
                try:
                    default = ast.literal_eval(call.args[1])
                except (ValueError, TypeError):
                    continue
                env = call.args[0].value
                values = {p: e[env] for p, e in exports.items() if env in e}
                row = dict(name=target.id, env=env, default=default, exports=values, expects={p: e[target.id] for p, e in expects.items() if target.id in e}, reads=config_reads[target.id])
                knobs.append(row)
                preceding = '\n'.join(sources['shared/config.py'].splitlines()[max(0, n.lineno-18):n.end_lineno])
                uniform = bool(launchers) and len(values) == len(launchers) and all(v == str(default) for v in values.values())
                inert = re.search(r'\b(?:inert|do not flip|kept for a revert)\b', preceding, re.I)
                if uniform or inert or launcher_markers[env]:
                    emit('dead_choice', 'shared/config.py', n.lineno, ast.unparse(n), 'all launchers equal raw environment default' if uniform else 'inert/revert marker near knob/export', symbol=target.id, knob=row, marker_exports=launcher_markers[env])
                break
    return dict(findings=sorted(findings.values(), key=lambda f: f['key']), files=metrics,
                documents=doc_metrics, knobs=knobs, launchers=sorted(launchers),
                ratchet_classes=sorted(RATCHET_CLASSES), counts=dict(Counter(f['class'] for f in findings.values())))


def baseline_keys(report):
    return {f['key'] for f in report['findings'] if f['class'] in RATCHET_CLASSES}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    report = scan(args.root)
    path = args.root / 'tools/code_bloat_baseline.txt'
    current = baseline_keys(report)
    if args.baseline:
        path.write_text('\n'.join(sorted(current)) + ('\n' if current else ''))
    previous = set(path.read_text().splitlines()) if path.exists() else set()
    report['new'] = sorted(current - previous)
    report['stale'] = sorted(previous - current)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"[code-bloat] {len(report['findings'])} findings; {len(current)} calibrated; {len(report['new'])} new; {len(report['stale'])} stale")
    return int(bool(report['new'] or report['stale']))


if __name__ == '__main__':
    raise SystemExit(main())
