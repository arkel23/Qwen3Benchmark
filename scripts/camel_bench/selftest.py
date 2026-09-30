"""Negative-test every grader before any model is scored: references score 1, wrong or empty replies score less.

Run on camel in the camel_bench env: python selftest.py
"""
import json
from pathlib import Path

import grade

TASKS = Path(__file__).resolve().parent / "tasks"
failures = []


def expect(name, ok):
    print(("ok   " if ok else "FAIL ") + name)
    if not ok:
        failures.append(name)


for task in sorted((TASKS / "codegen").glob("[0-9][0-9]_*")):
    reference = grade.grade_codegen(task.name, f"```python\n{(task / 'reference.py').read_text()}```")[0]
    wrong = grade.grade_codegen(task.name, f"```python\n{(task / 'wrong.py').read_text()}```")[0]
    expect(f"codegen {task.name} reference={reference:.2f} wrong={wrong:.2f}", reference == 1 and wrong < 1)

for task in sorted((TASKS / "reading").glob("r[0-9]_*")):
    for question in json.loads((task / "questions.json").read_text()):
        right = grade.grade_reading(task.name, question["id"], f"**{question['answer']}**")[0]
        wrong = grade.grade_reading(task.name, question["id"], "The document does not say.")[0]
        expect(f"reading {task.name} {question['id']} right={right} wrong={wrong}", right == 1 and wrong == 0)
    keys = json.loads((task / "summary_keys.json").read_text())
    full = " ".join(fact[0] for fact in keys["key_facts"])
    expect(f"reading {task.name} summary full={grade.grade_reading(task.name, 'summary', full)[0]} "
           f"long={grade.grade_reading(task.name, 'summary', full + ' word' * 200)[0]}",
           grade.grade_reading(task.name, "summary", full)[0] == 1
           and grade.grade_reading(task.name, "summary", full + " word" * 200)[0] == 0)

for task in sorted((TASKS / "instruct").glob("i[0-9]_*")):
    reference = grade.grade_instruct(task.name, (task / "reference.txt").read_text())[0]
    source = grade.grade_instruct(task.name, (task / "source.txt").read_text())[0]
    empty = grade.grade_instruct(task.name, "OK.")[0]
    expect(f"instruct {task.name} reference={reference} source={source} empty={empty}",
           reference == 1 and source < 1 and empty == 0)

print(f"{len(failures)} failures")
raise SystemExit(1 if failures else 0)
