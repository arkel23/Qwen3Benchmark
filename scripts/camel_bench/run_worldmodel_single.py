"""Single-turn world-model task: the repo's context goes in the prompt, the model returns the whole file.

The agentic form needs about an hour per run at the 16 tok/s these models reach in long tool-use turns,
so this form is the one that scales across models. Run on camel with the system python3.
"""
import argparse
import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE / "tasks/worldmodel/snapshot"
CONTEXT = {"W1_render": ["README.md", "pong/rules.py", "pong/sim.py", "pong/data.py"],
           "W2_classes": ["README.md", "FINDINGS.md", "pong/rules.py", "pong/sim.py",
                          "scripts/probe_sufficiency.py", "scripts/identifiability.py"]}
TARGET = {"W1_render": "pong/render.py", "W2_classes": "pong/classes.py"}


def build_prompt(task, tier):
    files = "\n\n".join(f"--- {name} ---\n{(SNAPSHOT / name).read_text()}" for name in CONTEXT[task])
    brief = (HERE / f"tasks/worldmodel/{task}/{tier}.md").read_text()
    return (f"You are writing one file for this repository. Its other files are below.\n\n{files}\n\n"
            f"{brief}\nReply with a single Python code block holding the complete file, and nothing else.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--task", required=True, choices=TARGET)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--repeat", default="0")
    parser.add_argument("--base_url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--out_dir", default="/edahome/pcslab/pcs05/edwin/model_bench/wm_single")
    parser.add_argument("--csv", default="/edahome/pcslab/pcs05/edwin/model_bench/worldmodel_single.csv")
    args = parser.parse_args()

    run = Path(args.out_dir) / f"{args.model_id}_{args.task}_{args.tier}_r{args.repeat}"
    subprocess.run(["rm", "-rf", str(run)], check=True)
    run.mkdir(parents=True)
    subprocess.run(["cp", "-r", str(SNAPSHOT), str(run / "repo")], check=True)
    (run / "repo" / TARGET[args.task]).unlink()
    if args.task == "W1_render":
        (run / "repo/pong/torch_render.py").unlink()

    body = {"messages": [{"role": "user", "content": build_prompt(args.task, args.tier)}],
            "max_tokens": 16384, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 1000}
    request = urllib.request.Request(f"{args.base_url}/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    start = time.time()
    reply = json.load(urllib.request.urlopen(request, timeout=7200))
    (run / "wall_s").write_text(str(round(time.time() - start)))
    content = reply["choices"][0]["message"]["content"] or ""
    (run / "reply.md").write_text(content)
    blocks = re.findall(r"```(?:python|py)?[^\n]*\n(.*?)```", content, flags=re.S)
    if blocks:
        (run / "repo" / TARGET[args.task]).write_text(max(blocks, key=len))

    print(f"{args.model_id} {args.task} {args.tier}: {reply['usage']['completion_tokens']} tokens, "
          f"{(run / 'wall_s').read_text()} s, file written: {(run / 'repo' / TARGET[args.task]).exists()}")
    subprocess.run(["/edahome/pcslab/pcs05/miniconda3/envs/camel_bench/bin/python", str(HERE / "grade_worldmodel.py"),
                    "--run", str(run), "--model_id", args.model_id, "--task", args.task, "--tier", args.tier,
                    "--repeat", args.repeat, "--out", args.csv])


if __name__ == "__main__":
    main()
