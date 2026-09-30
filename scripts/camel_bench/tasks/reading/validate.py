"""Check every reading task: answers verbatim in context, ambiguity, positions, token estimate, summary keys.

Run anywhere: python validate.py  (exits 1 on any FAIL)
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = {'context.txt', 'questions.json', 'summary_keys.json'}
PAPER_SEP = '=== PAPER: '
MAX_OCCURRENCES = 3
POSITION_TOLERANCE = 0.05


def pattern(answer):
    """Whitespace-flexible, case-insensitive match that is not part of a longer word or number."""
    body = r'\s+'.join(re.escape(part) for part in answer.split())
    return re.compile(rf'(?<![\w.]){body}(?!\w|\.\d)', re.IGNORECASE)


def summary_scope_text(context, scope):
    if scope == 'entire context':
        return context
    first = context.index(PAPER_SEP)
    second = context.find(PAPER_SEP, first + 1)
    return context[first:second if second != -1 else len(context)]


def check_task(task_dir):
    fails = []
    context = (task_dir / 'context.txt').read_text()
    questions = json.loads((task_dir / 'questions.json').read_text())
    summary = json.loads((task_dir / 'summary_keys.json').read_text())
    n_papers = context.count(PAPER_SEP)
    print(f'== {task_dir.name}: {len(context)} chars, ~{len(context) // 4} tokens (1 token ~ 4 chars), '
          f'{max(n_papers, 1)} paper(s), {len(questions)} questions')

    extra = {p.name for p in task_dir.iterdir()} ^ FILES
    if extra:
        fails.append(f'folder files differ from {sorted(FILES)}: {sorted(extra)}')
    if not 8 <= len(questions) <= 10:
        fails.append(f'{len(questions)} questions, expected 8-10')

    for q in questions:
        if set(q) != {'id', 'question', 'answer', 'aliases', 'position'}:
            fails.append(f'{q.get("id")}: fields {sorted(q)}')
        first = None
        for text in [q['answer'], *q['aliases']]:
            hits = [m.start() for m in pattern(text).finditer(context)]
            flag = ''
            if not hits:
                flag = '  MISSING'
                fails.append(f'{q["id"]}: {text!r} not in context')
            elif len(hits) > MAX_OCCURRENCES:
                flag = '  AMBIGUOUS-RISK'
                fails.append(f'{q["id"]}: {text!r} occurs {len(hits)} times')
            if text == q['answer'] and hits:
                first = hits[0] / len(context)
            print(f'  {q["id"]} {"answer" if text == q["answer"] else "alias "} {text!r}: {len(hits)}x{flag}')
        if first is not None:
            ok = abs(first - q['position']) <= POSITION_TOLERANCE
            print(f'  {q["id"]} position {q["position"]:.3f}, first occurrence {first:.3f}'
                  f'{"" if ok else "  POSITION-MISMATCH"}')
            if not ok:
                fails.append(f'{q["id"]}: position {q["position"]} vs first occurrence {first:.3f}')

    scope = summary['summary_scope']
    scope_text = summary_scope_text(context, scope).lower()
    if len(summary['key_facts']) != 5 or summary['max_words'] != 150:
        fails.append('summary_keys: expected max_words 150 and 5 key facts')
    for fact in summary['key_facts']:
        missing = [alt for alt in fact if alt != alt.lower() or alt not in scope_text]
        print(f'  key fact {fact}: {"MISSING " + str(missing) if missing else "ok"}')
        if missing:
            fails.append(f'key fact alternatives not lowercase or not in scope: {missing}')
    print(f'  summary scope: {scope} ({len(scope_text)} chars)')
    return fails


def main():
    tasks = sorted(p for p in HERE.iterdir() if p.is_dir() and (p / 'context.txt').exists())
    fails = {t.name: check_task(t) for t in tasks}
    print()
    for name, task_fails in fails.items():
        print(f'{name}: {"PASS" if not task_fails else "FAIL"}')
        for f in task_fails:
            print(f'  - {f}')
    sys.exit(1 if any(fails.values()) else 0)


if __name__ == '__main__':
    main()
