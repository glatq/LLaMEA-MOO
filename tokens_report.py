#!/usr/bin/env python3
import sys, pickle, re
from pathlib import Path
from csv import writer

# USAGE:
#   python tokens_report.py [ROOT] [--summaries]
#   - ROOT defaults to "exp_es_search"
#   - --summaries writes tokens_by_run.csv and tokens_by_run_model.csv

# Parse args
args = [a for a in sys.argv[1:] if a.strip()]
root_arg = next((a for a in args if not a.startswith("--")), "exp_es_search")
root = Path(root_arg)

print("file,run,code_name,llm_model,prompt_tokens,response_tokens,total_tokens")

# Aggregators
by_run_model = {}  # (run, model) -> [pt, rt, total]

for hp in sorted(root.rglob("*_handler.pkl")):
    try:
        with open(hp, "rb") as f:
            obj = pickle.load(f)
    except Exception:
        continue  # skip unreadable pickles

    # Token counts
    pt = getattr(obj, "prompt_token_count", 0) or 0
    rt = getattr(obj, "response_token_count", 0) or 0
    try:
        pt = int(pt)
    except:
        pt = 0
    try:
        rt = int(rt)
    except:
        rt = 0

    # Model id
    llm_model = getattr(obj, "llm_model", None) or ""

    # Code (algorithm) name
    code_name = getattr(obj, "code_name", None) or ""

    run = hp.parent.name
    total = pt + rt

    # Per-file CSV line (stdout)
    print(f"{hp},{run},{code_name},{llm_model},{pt},{rt},{total}")

    # Aggregate per (run, model)
    key = (run, llm_model)
    brm = by_run_model.setdefault(key, [0, 0, 0])
    brm[0] += pt
    brm[1] += rt
    brm[2] += total

with open("tokens_by_run_model.csv", "w", newline="") as f:
    w = writer(f)
    w.writerow(
        [
            "run",
            "llm_model",
            "total_prompt_tokens",
            "total_response_tokens",
            "overall_total_tokens",
        ]
    )
    for (run, model), (pt, rt, tt) in sorted(by_run_model.items()):
        w.writerow([run, model, pt, rt, tt])
