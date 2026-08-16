"""
Revision rerun over WildChat-4.8M (Frontiers R1 items 7, 8, 11, 15, 16).

One pass over the local HF cache computes:
  - R1.11: failure rate per individual model version x turn-count stratum
  - R1.8 : user experience re-operationalized as conversations PRECEDING each
           analyzed conversation (timestamp-ordered), same bins as the paper
  - R1.15: median / IQR / p90 of first-prompt word counts per experience level
  - R1.16: MATTR (window = 100 tokens), length-invariant lexical diversity
  - R1.7 : per-conversation records aggregated into cluster-robust (by hashed
           IP) standard errors for the key contrasts, with iid comparison and
           TOST re-run under robust SEs

Failure heuristics are IDENTICAL to scripts/03 (imported logic copied verbatim
via module import) so results are directly comparable.

Output: data/rerun_wildchat_results.json
"""

import importlib.util
import json
import math
import statistics
import sys
import time
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_FILE = DATA_DIR / "rerun_wildchat_results.json"

# Import analyze_conversation + MODEL_TIERS verbatim from script 03
spec = importlib.util.spec_from_file_location(
    "s03", ROOT / "scripts" / "03_failure_by_model.py")
s03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s03)
analyze_conversation = s03.analyze_conversation
MODEL_TIERS = s03.MODEL_TIERS
CACHE_DIR = s03.CACHE_DIR

MARGIN = 0.02
Z90 = 1.6448536269514722
Z95 = 1.959963984540054

EXP_BINS = [("1", 1, 1), ("2-3", 2, 3), ("4-10", 4, 10),
            ("11-30", 11, 30), ("31-100", 31, 100), ("100+", 101, 10**9)]


def exp_bin(nth):
    for name, lo, hi in EXP_BINS:
        if lo <= nth <= hi:
            return name
    return None


def bucket_of(n_turns):
    if n_turns == 2:
        return "2"
    if n_turns == 3:
        return "3"
    if n_turns <= 5:
        return "4-5"
    if n_turns <= 10:
        return "6-10"
    return "11+"


def mattr(tokens, window=100):
    if not tokens:
        return None
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    total = 0.0
    count = 0
    # slide in steps of 20 for speed; MATTR is stable under modest striding
    for start in range(0, len(tokens) - window + 1, 20):
        w = tokens[start:start + window]
        total += len(set(w)) / window
        count += 1
    return total / count


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def cluster_robust_diff(recs1, recs2):
    """Cluster-robust inference for p1 - p2 with possibly shared clusters.

    recs: list of (user, fail01). Sandwich variance of the difference in means
    treating each user as a cluster, allowing the same user in both groups.
    """
    n1, n2 = len(recs1), len(recs2)
    p1 = sum(f for _, f in recs1) / n1
    p2 = sum(f for _, f in recs2) / n2
    by_c1 = defaultdict(lambda: [0, 0])
    for u, f in recs1:
        by_c1[u][0] += f
        by_c1[u][1] += 1
    by_c2 = defaultdict(lambda: [0, 0])
    for u, f in recs2:
        by_c2[u][0] += f
        by_c2[u][1] += 1
    var = 0.0
    for u in set(by_c1) | set(by_c2):
        s1, m1 = by_c1.get(u, (0, 0))
        s2, m2 = by_c2.get(u, (0, 0))
        uc = (s1 - m1 * p1) / n1 - (s2 - m2 * p2) / n2
        var += uc * uc
    se_rob = math.sqrt(var)
    se_iid = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p1 - p2
    z = diff / se_rob
    p_two = 2 * norm_sf(abs(z))
    p_tost = max(norm_sf((diff + MARGIN) / se_rob),
                 norm_sf(-(diff - MARGIN) / se_rob))
    return {
        "n1": n1, "n2": n2, "p1": p1, "p2": p2, "diff": diff,
        "se_iid": se_iid, "se_cluster_robust": se_rob,
        "design_effect": (se_rob / se_iid) ** 2 if se_iid > 0 else None,
        "z_robust": z, "p_two_sided_robust": p_two,
        "p_tost_robust": p_tost,
        "ci90_robust": [diff - Z90 * se_rob, diff + Z90 * se_rob],
        "ci95_robust": [diff - Z95 * se_rob, diff + Z95 * se_rob],
        "margin": MARGIN,
    }


def quantiles(values):
    if not values:
        return None
    vs = sorted(values)
    q = statistics.quantiles(vs, n=100, method="inclusive")
    return {"n": len(vs), "mean": statistics.fmean(vs),
            "median": q[49], "q1": q[24], "q3": q[74], "p90": q[89],
            "min": vs[0], "max": vs[-1]}


def main():
    import pandas as pd

    t0 = time.time()
    user_ts = defaultdict(list)   # hashed_ip -> [epoch,...] ALL English convs
    records = []                  # multi-turn records

    total = 0
    for file_idx in range(86):
        pf = CACHE_DIR / f"train-{file_idx:05d}-of-00086.parquet"
        if not pf.exists():
            continue
        df = pd.read_parquet(pf, columns=[
            "model", "timestamp", "conversation", "language", "hashed_ip"])
        eng = df[df["language"] == "English"]
        for row in eng.itertuples(index=False):
            model = row.model or ""
            if model not in MODEL_TIERS:
                continue
            conv = row.conversation
            if conv is None:
                continue
            if not isinstance(conv, list):
                try:
                    conv = list(conv)
                except Exception:
                    continue
            if len(conv) == 0:
                continue
            total += 1
            ts = row.timestamp
            epoch = ts.timestamp() if ts is not None else None
            user = row.hashed_ip or ""
            user_ts[user].append(epoch if epoch is not None else -1.0)

            res = analyze_conversation(conv)
            if res is None:
                continue
            first_user = next((m.get("content") for m in conv
                               if m.get("role") == "user"
                               and isinstance(m.get("content"), str)), "")
            tokens = first_user.lower().split()
            records.append((
                model, MODEL_TIERS[model], bucket_of(res["n_user_turns"]),
                1 if res["has_any_failure"] else 0, user, epoch,
                len(tokens), mattr(tokens),
            ))
        print(f"  [{file_idx+1}/86] total={total:,} multi={len(records):,} "
              f"{time.time()-t0:.0f}s", flush=True)

    # ---- prior-conversation experience (R1.8) ----
    for u in user_ts:
        user_ts[u].sort()
    lifetime = {u: len(v) for u, v in user_ts.items()}

    per_model = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    exp_prior = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    exp_life = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    words_by_prior = defaultdict(list)
    words_by_life = defaultdict(list)
    mattr_by_prior = defaultdict(list)
    mattr_by_life = defaultdict(list)
    key_recs = defaultdict(list)  # for cluster-robust contrasts

    for model, tier, bucket, fail, user, epoch, words, mv in records:
        pm = per_model[model][bucket]
        pm[0] += 1
        pm[1] += fail

        nth_prior = (bisect_left(user_ts[user], epoch) + 1
                     if epoch is not None else None)
        b_prior = exp_bin(nth_prior) if nth_prior else None
        b_life = exp_bin(lifetime[user])

        if b_prior:
            c = exp_prior[b_prior][bucket]
            c[0] += 1
            c[1] += fail
        if b_life:
            c = exp_life[b_life][bucket]
            c[0] += 1
            c[1] += fail

        if bucket == "6-10":
            if b_prior:
                words_by_prior[b_prior].append(words)
                if mv is not None:
                    mattr_by_prior[b_prior].append(mv)
            if b_life:
                words_by_life[b_life].append(words)
                if mv is not None:
                    mattr_by_life[b_life].append(mv)
            if tier in ("gpt-4", "gpt-4o"):
                key_recs[f"tier_{tier}"].append((user, fail))
            if b_life in ("1", "100+"):
                key_recs[f"life_{b_life}"].append((user, fail))
            if b_prior in ("1", "100+"):
                key_recs[f"prior_{b_prior}"].append((user, fail))

    cluster = {
        "gpt4_vs_gpt4o_6-10": cluster_robust_diff(
            key_recs["tier_gpt-4"], key_recs["tier_gpt-4o"]),
        "lifetime_1_vs_100+_6-10": cluster_robust_diff(
            key_recs["life_1"], key_recs["life_100+"]),
        "prior_1st_vs_100+_6-10": cluster_robust_diff(
            key_recs["prior_1"], key_recs["prior_100+"]),
    }

    results = {
        "metadata": {
            "dataset": "WildChat-4.8M (local HF cache)",
            "total_english_known_model": total,
            "multi_turn": len(records),
            "elapsed_seconds": round(time.time() - t0, 1),
            "mattr_window": 100,
            "notes": "heuristics identical to scripts/03; experience bins as in paper",
        },
        "per_model_stratified": {
            m: {b: {"total": v[0], "fail": v[1]} for b, v in bs.items()}
            for m, bs in per_model.items()},
        "experience_prior": {
            l: {b: {"total": v[0], "fail": v[1]} for b, v in bs.items()}
            for l, bs in exp_prior.items()},
        "experience_lifetime_check": {
            l: {b: {"total": v[0], "fail": v[1]} for b, v in bs.items()}
            for l, bs in exp_life.items()},
        "prompt_words_6-10": {
            "by_prior": {l: quantiles(v) for l, v in words_by_prior.items()},
            "by_lifetime": {l: quantiles(v) for l, v in words_by_life.items()}},
        "mattr_6-10": {
            "by_prior": {l: quantiles(v) for l, v in mattr_by_prior.items()},
            "by_lifetime": {l: quantiles(v) for l, v in mattr_by_life.items()}},
        "cluster_robust": cluster,
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    print("\n== cluster-robust key contrasts (6-10 turns) ==")
    for k, v in cluster.items():
        print(f"{k}: p1={v['p1']*100:.2f}% p2={v['p2']*100:.2f}% "
              f"diff={v['diff']*100:+.2f}pp se_rob={v['se_cluster_robust']*100:.3f}pp "
              f"DEFF={v['design_effect']:.2f} pTOSTrob={v['p_tost_robust']:.3g}")
    print("\n== prior-experience failure @6-10 ==")
    for name, _, _ in EXP_BINS:
        c = exp_prior.get(name, {}).get("6-10")
        if c:
            print(f"  {name:>6}: {100*c[1]/c[0]:.2f}%  (n={c[0]:,})")
    print("\n== prompt words @6-10 (by lifetime) ==")
    for name, _, _ in EXP_BINS:
        q = quantiles(words_by_life.get(name, []))
        if q:
            print(f"  {name:>6}: mean={q['mean']:.0f} median={q['median']:.0f} "
                  f"IQR=[{q['q1']:.0f},{q['q3']:.0f}] p90={q['p90']:.0f}")
    print("\n== MATTR-100 @6-10 (by lifetime) ==")
    for name, _, _ in EXP_BINS:
        q = quantiles(mattr_by_life.get(name, []))
        if q:
            print(f"  {name:>6}: mean={q['mean']:.3f} median={q['median']:.3f}")
    print(f"\nSaved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
