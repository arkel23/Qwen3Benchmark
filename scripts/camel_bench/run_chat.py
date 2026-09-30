"""Send the single-turn screening tasks to a llama-server and record every reply as JSONL.

Run on camel with the system python3 (standard library only); grade with grade.py.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

TASKS = Path(__file__).resolve().parent / "tasks"
READING_INSTRUCTION = ("Answer the question using only the document above. Reply with the exact value "
                       "or name as it is written in the document, and nothing else.")


def chat(base_url, messages, seed, max_tokens):
    body = {"messages": messages, "seed": seed, "max_tokens": max_tokens, "cache_prompt": True,
            "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
    request = urllib.request.Request(f"{base_url}/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    start = time.time()
    with urllib.request.urlopen(request, timeout=7200) as response:
        reply = json.load(response)
    message = reply["choices"][0]["message"]
    return {"content": message.get("content") or "", "reasoning_chars": len(message.get("reasoning_content") or ""),
            "finish_reason": reply["choices"][0].get("finish_reason"), "usage": reply.get("usage", {}),
            "timings": reply.get("timings", {}), "wall_s": round(time.time() - start, 2)}


def codegen_requests():
    for task in sorted((TASKS / "codegen").glob("[0-9][0-9]_*")):
        yield "codegen", task.name, "", [{"role": "user", "content": (task / "prompt.md").read_text()}]


def instruct_requests():
    for task in sorted((TASKS / "instruct").glob("i[0-9]_*")):
        yield "instruct", task.name, "", [{"role": "user", "content": (task / "prompt.md").read_text()}]


def reading_requests():
    for task in sorted((TASKS / "reading").glob("r[0-9]_*")):
        context = (task / "context.txt").read_text()
        for question in json.loads((task / "questions.json").read_text()):
            content = f"{context}\n\n{READING_INSTRUCTION}\n\nQuestion: {question['question']}"
            yield "reading", task.name, question["id"], [{"role": "user", "content": content}]
        keys = json.loads((task / "summary_keys.json").read_text())
        scope = keys.get("summary_scope", "the document above")
        content = f"{context}\n\nSummarize {scope} in at most {keys['max_words']} words of plain prose."
        yield "reading", task.name, "summary", [{"role": "user", "content": content}]


FAMILIES = {"codegen": codegen_requests, "instruct": instruct_requests, "reading": reading_requests}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base_url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--families", nargs="+", default=["codegen", "instruct", "reading"], choices=FAMILIES)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model_repeats", type=json.loads, default={"M4": 1})
    parser.add_argument("--reading_repeats", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=16384)
    args = parser.parse_args()

    out = Path(args.out)
    done = set()
    if out.exists():
        done = {(r["family"], r["task"], r["item"], r["repeat"])
                for r in map(json.loads, out.read_text().splitlines()) if "error" not in r}
    with out.open("a") as f:
        for family in args.families:
            repeats = args.reading_repeats if family == "reading" else \
                args.model_repeats.get(args.model_id, args.repeats)
            for repeat in range(repeats):
                for family_name, task, item, messages in FAMILIES[family]():
                    key = (family_name, task, item, repeat)
                    if key in done:
                        continue
                    record = {"model_id": args.model_id, "family": family_name, "task": task, "item": item,
                              "repeat": repeat}
                    try:
                        record.update(chat(args.base_url, messages, seed=1000 + repeat, max_tokens=args.max_tokens))
                    except Exception as error:
                        record["error"] = repr(error)
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    print(f"{args.model_id} {family_name} {task} {item} r{repeat} "
                          f"{record.get('wall_s', 'ERR')} s {record.get('error', '')}", flush=True)


if __name__ == "__main__":
    main()
