"""Speed of one model on camel: llama-bench at several context depths, then 1/2/4 concurrent chat users.

python3 serve_bench.py M0 [--n_cpu_moe 30]; appends rows to model_bench/speed.csv.
"""
import argparse
import csv
import json
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import serve

OUT = serve.EDWIN / "model_bench"
FIELDS = ["model_id", "test", "depth", "slots", "clients", "tok_per_s", "stddev", "per_client_decode_tps", "vram_mib"]
PROMPT = (Path(__file__).resolve().parent / "tasks/reading/r1_short/context.txt")


def vram_mib():
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                       text=True).split()[0])


def llama_bench(model_id, n_cpu_moe, depths):
    command = [str(serve.LLAMA_BIN / "llama-bench"), "-m", str(serve.gguf_path(model_id)), "-p", "512", "-n", "128",
               "-d", ",".join(map(str, depths)), "-r", "2", "-fa", "1", "-o", "json",
               *serve.model_args(model_id, n_cpu_moe)]
    command = [c if c != "--n-cpu-moe" else "-ncmoe" for c in command]
    runs = json.loads(subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL))
    for run in runs:
        test = f"pp{run['n_prompt']}" if run["n_prompt"] else f"tg{run['n_gen']}"
        yield {"model_id": model_id, "test": test, "depth": run.get("n_depth", 0), "slots": 1, "clients": 1,
               "tok_per_s": round(run["avg_ts"], 2), "stddev": round(run["stddev_ts"], 2)}


def one_chat(text, results):
    body = {"messages": [{"role": "user", "content": f"{text}\n\nSummarize the document above."}],
            "max_tokens": 256, "ignore_eos": True, "temperature": 0.6, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(f"http://127.0.0.1:{serve.PORT}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=3600) as response:
        results.append(json.load(response))


def concurrent_users(model_id, n_cpu_moe, slots=4):
    text = PROMPT.read_text()[:4000]
    serve.start(model_id, 16384 * slots, slots, n_cpu_moe, str(OUT / f"logs/server_{model_id}_np{slots}.log"))
    try:
        vram = vram_mib()
        for clients in (1, 2, 4):
            results = []
            threads = [threading.Thread(target=one_chat, args=(text, results)) for _ in range(clients)]
            start = time.time()
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            wall = time.time() - start
            generated = sum(r["usage"]["completion_tokens"] for r in results)
            per_client = [round(r["timings"]["predicted_per_second"], 2) for r in results]
            yield {"model_id": model_id, "test": "chat_aggregate", "depth": 0, "slots": slots, "clients": clients,
                   "tok_per_s": round(generated / wall, 2), "per_client_decode_tps": " ".join(map(str, per_client)),
                   "vram_mib": vram}
    finally:
        serve.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", choices=serve.MODELS)
    parser.add_argument("--n_cpu_moe", type=int, default=0)
    parser.add_argument("--depths", default="0,32768,100000")
    args = parser.parse_args()

    path = OUT / "speed.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        for row in [*llama_bench(args.model_id, args.n_cpu_moe, args.depths.split(",")),
                    *concurrent_users(args.model_id, args.n_cpu_moe)]:
            writer.writerow(row)
            f.flush()
            print(row, flush=True)


if __name__ == "__main__":
    main()
