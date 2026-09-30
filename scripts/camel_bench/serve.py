"""Start or stop a llama-server for one models.json entry; used by run_model.sh and serve_bench.py.

python3 serve.py start M0 --ctx 131072 [--slots 1] [--n_cpu_moe 30]; python3 serve.py stop
"""
import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = json.loads((HERE / "models.json").read_text())
EDWIN = Path("/edahome/pcslab/pcs05/edwin")
LLAMA_BIN = EDWIN / "projects/llama.cpp/build/bin"
PIDFILE = EDWIN / "model_bench/logs/bench_server.pid"
PORT = 8080


def gguf_path(model_id):
    model = MODELS[model_id]
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    matches = sorted((hub / f"models--{model['repo'].replace('/', '--')}" / "snapshots").glob(f"*/{model['file']}"))
    if not matches:
        raise FileNotFoundError(f"{model_id}: {model['file']} not downloaded")
    return matches[-1]


def model_args(model_id, n_cpu_moe):
    return [str(n_cpu_moe) if a == "NCPUMOE" else a for a in MODELS[model_id]["args"]]


def healthy():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5):
            return True
    except OSError:
        return False


def start(model_id, ctx, slots, n_cpu_moe, log):
    if healthy():
        raise RuntimeError(f"a server already answers on :{PORT}")
    command = [str(LLAMA_BIN / "llama-server"), "-m", str(gguf_path(model_id)), "--host", "127.0.0.1",
               "--port", str(PORT), "--ctx-size", str(ctx), "-np", str(slots), "--jinja",
               "--reasoning-effort", "low", "-n", "16384", *model_args(model_id, n_cpu_moe)]
    process = subprocess.Popen(command, stdout=open(log, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               start_new_session=True)
    PIDFILE.write_text(str(process.pid))
    start_time = time.time()
    while not healthy():
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited during load, see {log}")
        time.sleep(3)
    return time.time() - start_time


def stop():
    if PIDFILE.exists():
        try:
            os.kill(int(PIDFILE.read_text()), signal.SIGTERM)
        except ProcessLookupError:
            pass
        PIDFILE.unlink()
        while healthy():
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "stop"])
    parser.add_argument("model_id", nargs="?", choices=MODELS)
    parser.add_argument("--ctx", type=int, default=131072)
    parser.add_argument("--slots", type=int, default=1)
    parser.add_argument("--n_cpu_moe", type=int, default=0)
    parser.add_argument("--log", default=str(EDWIN / "model_bench/logs/bench_server.log"))
    args = parser.parse_args()
    if args.action == "stop":
        stop()
        return
    print(f"{args.model_id} loaded in {start(args.model_id, args.ctx, args.slots, args.n_cpu_moe, args.log):.0f} s")


if __name__ == "__main__":
    main()
