"""Grade one world-model run: restore the hidden tests, execute them, check the published numbers, measure style.

Run in the camel_bench env; appends one row to worldmodel.csv.
"""
import argparse
import ast
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HIDDEN = HERE / "tasks/worldmodel/hidden_tests"
TARGET = {"W1_render": "pong/render.py", "W2_classes": "pong/classes.py"}
TESTS = {"W1_render": "render", "W2_classes": "classes or consistent"}
EXPECTED_SUFFICIENCY = ["probes 8+8x128: 195 classes, 5 split on held-out probes",
                        "probes 32+32x256: 200 classes, 0 split on held-out probes",
                        "probes 64+64x256: 200 classes, 0 split on held-out probes",
                        "probes 128+128x512: 200 classes, 0 split on held-out probes"]
FIELDS = ["model_id", "task", "tier", "repeat", "wall_s", "file_written", "tests_passed", "tests_total",
          "numbers_match", "objective_score", "loc", "reference_loc", "comment_lines", "docstring_lines",
          "style_flags", "detail"]


def style(source, path):
    tree = ast.parse(source)
    docstrings = sum(len(ast.get_docstring(n).splitlines()) for n in ast.walk(tree)
                     if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef)) and ast.get_docstring(n))
    comments = sum(1 for line in source.splitlines() if line.strip().startswith("#"))
    flags = []
    if re.search(r"['\"]/(mnt|edahome|home)/", source):
        flags.append("absolute_path")
    if re.search(r"\b(TODO|FIXME|XXX)\b", source):
        flags.append("todo")
    if re.search(r"^\s*print\(", source, re.M) and "scripts/" not in str(path):
        flags.append("print_in_module")
    if re.search(r"#\s*(added|changed|fixed|previously|was )", source, re.I):
        flags.append("history_comment")
    if re.search(r"^\s*except\s+Exception", source, re.M):
        flags.append("broad_except")
    return len(source.strip().splitlines()), comments, docstrings, flags


def main():
    parser = argparse.ArgumentParser()
    for name in ["run", "model_id", "task", "tier", "repeat", "out"]:
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()

    run, task = Path(args.run), args.task
    graded = run / "graded_repo"
    shutil.rmtree(graded, ignore_errors=True)
    shutil.copytree(run / "repo", graded)
    shutil.copytree(HIDDEN / "tests", graded / "tests")
    if task == "W1_render":
        shutil.copy(HERE / "tasks/worldmodel/snapshot/pong/torch_render.py", graded / "pong/torch_render.py")

    written = (graded / TARGET[task]).exists()
    row = {"model_id": args.model_id, "task": task, "tier": args.tier, "repeat": args.repeat,
           "wall_s": (run / "wall_s").read_text().strip(), "file_written": int(written)}
    detail = []

    passed = total = 0
    if written:
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-k", TESTS[task],
                               "tests/test_engine.py"], cwd=graded, capture_output=True, text=True, timeout=1800)
        summary = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr[-300:]
        passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", summary)) else 0
        total = passed + sum(int(n) for n in re.findall(r"(\d+) (?:failed|error)", summary))
        detail.append(summary.strip())
    row.update(tests_passed=passed, tests_total=total)

    numbers = ""
    if written and task == "W2_classes":
        try:
            out = subprocess.run([sys.executable, "scripts/probe_sufficiency.py"], cwd=graded, capture_output=True,
                                 text=True, timeout=1800).stdout
            numbers = int(all(line in out for line in EXPECTED_SUFFICIENCY))
            detail.append("sufficiency " + ("matches FINDINGS" if numbers else f"differs: {out.strip()[:200]}"))
        except subprocess.TimeoutExpired:
            numbers, _ = 0, detail.append("sufficiency timeout")
    row["numbers_match"] = numbers

    objective = (passed / total if total else 0.0) if task == "W1_render" else \
                (0.5 * (passed / total if total else 0.0) + 0.5 * (numbers or 0))
    row["objective_score"] = round(objective, 3)

    if written:
        source = (graded / TARGET[task]).read_text()
        reference = (HIDDEN / f"reference_{task.split('_')[1]}.py").read_text()
        loc, comments, docstrings, flags = style(source, TARGET[task])
        row.update(loc=loc, reference_loc=len(reference.strip().splitlines()), comment_lines=comments,
                   docstring_lines=docstrings, style_flags=" ".join(flags))
    row["detail"] = " | ".join(detail)[:400]

    path = Path(args.out)
    path.touch()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
