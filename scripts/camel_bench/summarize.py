"""Aggregate graded/*.csv and speed.csv into one row per model for results/camel_model_bench.csv.

python summarize.py --bench_dir <model_bench dir> --out results/camel_model_bench.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd

MODELS = json.loads((Path(__file__).resolve().parent / "models.json").read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    bench = Path(args.bench_dir)

    graded = pd.concat([pd.read_csv(p) for p in sorted((bench / "graded").glob("*.csv"))])
    graded["kind"] = graded["family"].where(graded["item"].ne("summary"), "summary")
    covered = graded.groupby("model_id").apply(lambda g: set(zip(g["task"], g["item"])), include_groups=False).map(len)
    if covered.nunique() != 1:
        print(f"partial coverage: {covered.to_dict()} (most items = {covered.max()})")

    scores = graded.pivot_table(index="model_id", columns="kind", values="score", aggfunc="mean")
    scores.columns = [f"{c}_score" for c in scores.columns]
    codegen = graded[graded["family"].eq("codegen")]
    scores["codegen_all_pass_rate"] = codegen.assign(ok=codegen["score"].eq(1)).groupby("model_id")["ok"].mean()
    scores["request_errors"] = graded.groupby("model_id")["error"].apply(lambda e: e.notna().sum())
    scores["median_completion_tokens"] = graded.groupby("model_id")["completion_tokens"].median()
    scores["median_decode_tps"] = graded.groupby("model_id")["decode_tps"].median()

    speed = pd.read_csv(bench / "speed.csv")
    for test, depth in [("pp512", 0), ("tg128", 0), ("pp512", 32768), ("tg128", 32768), ("pp512", 100000),
                        ("tg128", 100000)]:
        rows = speed[speed["test"].eq(test) & speed["depth"].eq(depth)].set_index("model_id")
        scores[f"{test}_d{depth}_tps"] = rows["tok_per_s"]
    chat = speed[speed["test"].eq("chat_aggregate")]
    for clients in (1, 2, 4):
        scores[f"chat_{clients}users_aggregate_tps"] = chat[chat["clients"].eq(clients)].set_index("model_id")["tok_per_s"]

    scores.insert(1, "items_scored", covered)
    scores.insert(0, "model", [MODELS[m]["name"] for m in scores.index])
    scores.round(4).to_csv(args.out)
    print(scores.round(3).to_string())


if __name__ == "__main__":
    main()
