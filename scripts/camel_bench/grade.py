"""Grade run_chat.py replies: hidden pytest tests for codegen, exact match for reading, scripted rule checks.

Run on camel in the camel_bench env: python grade.py --raw raw/M0.jsonl --out graded/M0.csv
"""
import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TASKS = Path(__file__).resolve().parent / "tasks"
FIELDS = ["model_id", "family", "task", "item", "repeat", "score", "detail", "prompt_tokens", "completion_tokens",
          "reasoning_chars", "prefill_tps", "decode_tps", "wall_s", "finish_reason", "error"]
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def extract_code(reply):
    blocks = re.findall(r"```(?:python|py)?[^\n]*\n(.*?)```", reply, flags=re.S)
    return max(blocks, key=len) if blocks else reply


def run_pytest(task_dir, code):
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(task_dir / "test_task.py", tmp)
        Path(tmp, "solution.py").write_text(code)
        try:
            proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_task.py"],
                                  cwd=tmp, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            return 0, 0, "timeout"
    summary = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr[-200:]
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", summary)) else 0
    failed = sum(int(n) for n in re.findall(r"(\d+) (?:failed|error)", summary))
    return passed, passed + failed, summary


_test_counts = {}


def grade_codegen(task, reply):
    task_dir = TASKS / "codegen" / task
    if task not in _test_counts:
        _test_counts[task] = run_pytest(task_dir, (task_dir / "reference.py").read_text())[0]
    passed, _, summary = run_pytest(task_dir, extract_code(reply))
    return passed / _test_counts[task], f"{passed}/{_test_counts[task]} {summary}"


def normalise(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = text.strip().strip("*`\"'").rstrip(".").strip()
    text = text.replace("−", "-").replace("×", "x")
    return re.sub(r"\s*=\s*", "=", re.sub(r"\s+", " ", text)).lower()


def grade_reading(task, item, reply):
    task_dir = TASKS / "reading" / task
    if item == "summary":
        keys = json.loads((task_dir / "summary_keys.json").read_text())
        words = len(reply.split())
        covered = sum(any(alt in reply.lower() for alt in fact) for fact in keys["key_facts"])
        coverage = covered / len(keys["key_facts"])
        within = words <= keys["max_words"]
        return coverage * within, f"words={words} facts={covered}/{len(keys['key_facts'])}"
    question = next(q for q in json.loads((task_dir / "questions.json").read_text()) if q["id"] == item)
    targets = [normalise(a) for a in [question["answer"], *question.get("aliases", [])]]
    answer = normalise(reply)
    exact = answer in targets
    contained = len(answer) <= max(map(len, targets)) + 40 and any(t in answer for t in targets)
    # A short answer may drop the unit or label it was printed with ("9" for "9 backbones").
    partial = any(answer in re.split(r"[\s=]+", t) and (not any(c.isdigit() for c in t) or any(c.isdigit() for c in answer))
                  for t in targets)
    score = float(exact or contained or partial)
    return score, f"exact={exact} contained={contained} partial={partial} reply={answer[:60]!r}"


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def grade_instruct(task, reply):
    task_dir = TASKS / "instruct" / task
    checks = json.loads((task_dir / "checks.json").read_text())
    source = (task_dir / "source.txt").read_text()
    text = re.sub(r"<think>.*?</think>", "", reply, flags=re.S).strip()
    results = {
        "max_sentence_words": max(len(s.split()) for s in sentences(text)) <= checks["max_sentence_words"],
        "forbidden": not any(f in text for f in checks["forbidden"]),
        "numbers_kept": sorted(NUMBER.findall(text)) == sorted(NUMBER.findall(source)),
        "required_kept": all(r in text for r in checks.get("required", [])),
        "length_ratio": checks["min_ratio"] <= len(text.split()) / len(source.split()) <= checks["max_ratio"],
    }
    detail = " ".join(f"{k}={int(v)}" for k, v in results.items())
    if not (results["length_ratio"] and results["required_kept"]):
        return 0.0, detail
    return sum(results.values()) / len(results), detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for record in map(json.loads, Path(args.raw).read_text().splitlines()):
            row = {k: record.get(k, "") for k in ["model_id", "family", "task", "item", "repeat", "wall_s",
                                                   "finish_reason", "reasoning_chars", "error"]}
            usage, timings = record.get("usage", {}), record.get("timings", {})
            row.update(prompt_tokens=usage.get("prompt_tokens", ""), completion_tokens=usage.get("completion_tokens", ""),
                       prefill_tps=round(timings.get("prompt_per_second", 0), 2),
                       decode_tps=round(timings.get("predicted_per_second", 0), 2))
            if "error" in record:
                row.update(score=0.0, detail="request failed")
            elif record["family"] == "codegen":
                row["score"], row["detail"] = grade_codegen(record["task"], record["content"])
            elif record["family"] == "reading":
                row["score"], row["detail"] = grade_reading(record["task"], record["item"], record["content"])
            else:
                row["score"], row["detail"] = grade_instruct(record["task"], record["content"])
            writer.writerow(row)
            print(row["model_id"], row["family"], row["task"], row["item"], row["repeat"], round(float(row["score"]), 3),
                  flush=True)


if __name__ == "__main__":
    main()
