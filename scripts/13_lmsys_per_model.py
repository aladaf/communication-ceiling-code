"""
LMSYS-Chat-1M per-individual-model failure analysis (Frontiers R1.11, 2nd half).

Downloads the 6 parquet shards via hf_hub_download (resumable; requires the
gated-dataset access already granted to the logged-in HF account), then applies
the IDENTICAL failure heuristics of scripts/03 (imported) and the tier mapping
of scripts/04 (imported), reporting per-model x turn-stratum counts.

Output: data/lmsys_per_model_results.json
"""

import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT = DATA_DIR / "lmsys_per_model_results.json"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s03 = load_module("s03", ROOT / "scripts" / "03_failure_by_model.py")
s04 = load_module("s04", ROOT / "scripts" / "04_failure_lmsys.py")
analyze_conversation = s03.analyze_conversation
TIERS = s04.MODEL_TIERS


def bucket_of(n):
    if n == 2:
        return "2"
    if n == 3:
        return "3"
    if n <= 5:
        return "4-5"
    if n <= 10:
        return "6-10"
    return "11+"


def main():
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = [f for f in api.list_repo_files("lmsys/lmsys-chat-1m", repo_type="dataset")
             if f.endswith(".parquet")]
    print(f"{len(files)} parquet shards")

    per_model = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    total = english = multi = 0
    t0 = time.time()

    for i, remote in enumerate(sorted(files), 1):
        print(f"[{i}/{len(files)}] downloading {remote} ...", flush=True)
        local = hf_hub_download("lmsys/lmsys-chat-1m", remote, repo_type="dataset")
        df = pd.read_parquet(local, columns=["model", "language", "conversation"])
        for row in df.itertuples(index=False):
            total += 1
            if row.language != "English":
                continue
            english += 1
            model = row.model or ""
            if model not in TIERS:
                continue
            conv = row.conversation
            if conv is None:
                continue
            if not isinstance(conv, list):
                try:
                    conv = list(conv)
                except Exception:
                    continue
            res = analyze_conversation(conv)
            if res is None:
                continue
            multi += 1
            c = per_model[model][bucket_of(res["n_user_turns"])]
            c[0] += 1
            c[1] += res["has_any_failure"]
        print(f"    total={total:,} english={english:,} multi={multi:,} "
              f"{time.time()-t0:.0f}s", flush=True)

    OUT.write_text(json.dumps({
        "metadata": {"dataset": "lmsys/lmsys-chat-1m", "total": total,
                     "english": english, "multi_turn": multi,
                     "elapsed_seconds": round(time.time() - t0, 1),
                     "heuristics": "identical to scripts/03",
                     "tiers": "identical to scripts/04"},
        "per_model_stratified": {
            m: {b: {"total": v[0], "fail": v[1]} for b, v in bs.items()}
            for m, bs in per_model.items()},
    }, indent=2))

    print("\n== per-model @6-10 turns (n >= 100) ==")
    rows = []
    for m, bs in per_model.items():
        c = bs.get("6-10")
        if c and c[0] >= 100:
            rows.append((TIERS[m], m, c[0], 100 * c[1] / c[0]))
    for tier, m, n, r in sorted(rows):
        print(f"  [{tier:<8}] {m:<28} n={n:>6,}  fail={r:5.1f}%")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
