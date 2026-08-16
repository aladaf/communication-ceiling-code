"""
Capability Overhang — Communication Failure by Model Tier

Key question: Do more capable models produce fewer communication failures?
If NOT — if failure rates are similar across tiers — then the interface,
not the model, is the bottleneck. That IS capability overhang.

Uses the same failure detection heuristics from Epistemic Apertures:
1. Explicit failure markers ("no I mean", "not what I meant", etc.)
2. Refinement chains (3+ reformulations of same request)

CRITICAL: Only compare multi-turn conversations (2+ user turns),
since single-turn conversations can't show failure.

Usage:
    python scripts/03_failure_by_model.py
"""

import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
import math

CACHE_DIR = Path(
    "/home/aladaf/.cache/huggingface/hub/datasets--allenai--WildChat-4.8M"
    "/snapshots/c827c6df8fcf008219ffaffa4d1dd77491099367/data"
)
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
RESULTS_FILE = DATA_DIR / "failure_by_model_results.json"

# --- Model tiers ---
MODEL_TIERS = {
    "gpt-3.5-turbo-0301": "gpt-3.5",
    "gpt-3.5-turbo-0613": "gpt-3.5",
    "gpt-3.5-turbo-0125": "gpt-3.5",
    "gpt-4-0314": "gpt-4",
    "gpt-4-0613": "gpt-4",
    "gpt-4-1106-preview": "gpt-4",
    "gpt-4-0125-preview": "gpt-4",
    "gpt-4-turbo-2024-04-09": "gpt-4",
    "gpt-4o-2024-05-13": "gpt-4o",
    "gpt-4o-2024-08-06": "gpt-4o",
    "gpt-4o-2024-11-20": "gpt-4o",
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "gpt-4.1-mini-2025-04-14": "gpt-4.1-mini",
    "o1-mini-2024-09-12": "o1",
    "o1-preview-2024-09-12": "o1",
}

TIER_ORDER = ["gpt-3.5", "gpt-4", "gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "o1"]

# --- Failure detection (from Epistemic Apertures) ---
FAILURE_MARKERS = [
    r"that'?s not what i mean",
    r"no,?\s*i mean",
    r"what i('m| am) trying to say",
    r"let me (try|explain|rephrase|clarify|re-?word)",
    r"i('ll| will) try (again|to explain|to rephrase)",
    r"i can'?t (explain|describe|put into words|articulate)",
    r"(hard|difficult) to (explain|describe|put into words|articulate|express)",
    r"i wish i could show you",
    r"(sorry|apologies),?\s*(let me|i('ll| will)) (try again|rephrase|clarify)",
    r"that came out wrong",
    r"i('m| am) not (explaining|describing) (this|it) well",
    r"does that make sense",
    r"am i making (any )?sense",
    r"if (only )?you could see",
    r"i know what i want but",
    r"it'?s like\.\.\.?\s*(you know|um|uh)",
    r"no,?\s*not (that|like that|what i)",
    r"you misunderstood",
    r"that'?s not (right|correct|what i)",
    r"close,?\s*but (not|no)",
    r"almost,?\s*but",
    r"not (quite|exactly)",
    r"you'?re (on the right track|close) but",
]

failure_patterns = [re.compile(p, re.IGNORECASE) for p in FAILURE_MARKERS]


def analyze_conversation(conversation_msgs):
    """Detect communication failure in a conversation."""
    user_messages = []

    for msg in conversation_msgs:
        role = msg.get('role', '')
        content = msg.get('content', '')
        if not isinstance(content, str):
            continue
        if role == 'user':
            user_messages.append(content)

    n_user = len(user_messages)
    if n_user < 2:
        return None  # Need 2+ turns to detect failure

    result = {
        'n_user_turns': n_user,
        'failure_marker_count': 0,
        'has_failure_markers': False,
        'has_refinement_chains': False,
        'has_any_failure': False,
    }

    # 1. Explicit failure markers
    for msg in user_messages[1:]:
        for pattern in failure_patterns:
            if pattern.search(msg):
                result['failure_marker_count'] += 1
                break

    result['has_failure_markers'] = result['failure_marker_count'] > 0

    # 2. Refinement chains (3+ consecutive reformulations)
    if n_user >= 3:
        stop_words = {'the', 'a', 'an', 'is', 'it', 'to', 'and', 'or', 'but', 'in',
                      'on', 'for', 'of', 'i', 'you', 'my', 'your', 'me', 'do', 'can',
                      'this', 'that', 'with', 'be', 'have', 'are', 'was', 'were', 'will'}
        chain_length = 1
        has_chain = False

        for i in range(1, len(user_messages)):
            prev = user_messages[i-1].lower()
            curr = user_messages[i].lower()

            is_refinement = False
            if len(curr) < 300:
                if any(curr.startswith(s) for s in ['no', 'not', 'i mean', 'sorry', 'actually', 'wait', 'let me']):
                    is_refinement = True

                prev_words = set(prev.split()) - stop_words
                curr_words = set(curr.split()) - stop_words
                if prev_words and curr_words:
                    overlap = len(prev_words & curr_words) / max(len(prev_words), len(curr_words))
                    if overlap > 0.3:
                        is_refinement = True

            if is_refinement:
                chain_length += 1
            else:
                if chain_length >= 3:
                    has_chain = True
                chain_length = 1

        if chain_length >= 3:
            has_chain = True
        result['has_refinement_chains'] = has_chain

    result['has_any_failure'] = result['has_failure_markers'] or result['has_refinement_chains']
    return result


def main():
    import pandas as pd

    print("=" * 70)
    print("  Communication Failure Rate by Model Capability Tier")
    print("  (Multi-turn conversations only)")
    print("=" * 70)
    print()

    # Per-tier stats, stratified by turn count
    # tier_stats[tier][turn_bucket] = {total, with_failure, with_markers, with_chains}
    tier_stats = defaultdict(lambda: defaultdict(lambda: {
        "total": 0,
        "with_any_failure": 0,
        "with_failure_markers": 0,
        "with_refinement_chains": 0,
        "failure_marker_count_sum": 0,
    }))

    # Also per-individual-model stats
    model_stats = defaultdict(lambda: {"total_multi": 0, "with_any_failure": 0})

    total = 0
    multi_turn = 0
    t0 = time.time()

    for file_idx in range(86):
        parquet_file = CACHE_DIR / f"train-{file_idx:05d}-of-00086.parquet"
        if not parquet_file.exists():
            continue

        df = pd.read_parquet(parquet_file)
        eng = df[df['language'] == 'English']

        for _, row in eng.iterrows():
            model = row.get('model', '')
            tier = MODEL_TIERS.get(model)
            if not tier:
                continue

            conversation = row.get('conversation', [])
            if conversation is None:
                continue
            if not isinstance(conversation, list):
                try:
                    conversation = list(conversation)
                except:
                    continue
            if len(conversation) == 0:
                continue

            total += 1

            result = analyze_conversation(conversation)
            if result is None:
                continue  # single-turn, skip

            multi_turn += 1
            n_turns = result['n_user_turns']

            # Bucket turns: 2, 3, 4-5, 6-10, 11+
            if n_turns == 2:
                bucket = "2"
            elif n_turns == 3:
                bucket = "3"
            elif n_turns <= 5:
                bucket = "4-5"
            elif n_turns <= 10:
                bucket = "6-10"
            else:
                bucket = "11+"

            ts = tier_stats[tier][bucket]
            ts["total"] += 1
            if result['has_any_failure']:
                ts["with_any_failure"] += 1
            if result['has_failure_markers']:
                ts["with_failure_markers"] += 1
            if result['has_refinement_chains']:
                ts["with_refinement_chains"] += 1
            ts["failure_marker_count_sum"] += result['failure_marker_count']

            ms = model_stats[model]
            ms["total_multi"] += 1
            if result['has_any_failure']:
                ms["with_any_failure"] += 1

        elapsed = time.time() - t0
        print(f"  [{file_idx+1}/86] Total: {total:,} | Multi-turn: {multi_turn:,} | {elapsed:.0f}s")

    # --- RESULTS ---
    print()
    print("=" * 80)
    print("  RESULT 1: FAILURE RATE BY MODEL TIER (all multi-turn)")
    print("=" * 80)

    header = f"{'Tier':<14} {'MultiTurn':>10} {'AnyFail':>10} {'Fail%':>8} {'Markers':>10} {'Mark%':>8} {'Chains':>10} {'Chain%':>8}"
    print(header)
    print("-" * len(header))

    for tier in TIER_ORDER:
        if tier not in tier_stats:
            continue
        total_t = sum(b["total"] for b in tier_stats[tier].values())
        fail_t = sum(b["with_any_failure"] for b in tier_stats[tier].values())
        mark_t = sum(b["with_failure_markers"] for b in tier_stats[tier].values())
        chain_t = sum(b["with_refinement_chains"] for b in tier_stats[tier].values())
        if total_t == 0:
            continue
        print(f"{tier:<14} {total_t:>10,} {fail_t:>10,} {100*fail_t/total_t:>7.1f}% "
              f"{mark_t:>10,} {100*mark_t/total_t:>7.1f}% "
              f"{chain_t:>10,} {100*chain_t/total_t:>7.1f}%")

    print()
    print("=" * 80)
    print("  RESULT 2: FAILURE RATE BY TIER × TURN COUNT")
    print("  (controlled for conversation length)")
    print("=" * 80)

    turn_buckets = ["2", "3", "4-5", "6-10", "11+"]

    for bucket in turn_buckets:
        print(f"\n--- {bucket} user turns ---")
        print(f"{'Tier':<14} {'N':>8} {'Fail%':>8} {'Marker%':>8} {'Chain%':>8}")
        print("-" * 50)
        for tier in TIER_ORDER:
            if tier not in tier_stats or bucket not in tier_stats[tier]:
                continue
            s = tier_stats[tier][bucket]
            n = s["total"]
            if n < 20:
                continue
            fail_pct = 100 * s["with_any_failure"] / n
            mark_pct = 100 * s["with_failure_markers"] / n
            chain_pct = 100 * s["with_refinement_chains"] / n
            print(f"{tier:<14} {n:>8,} {fail_pct:>7.1f}% {mark_pct:>7.1f}% {chain_pct:>7.1f}%")

    print()
    print("=" * 80)
    print("  RESULT 3: FAILURE RATE BY INDIVIDUAL MODEL")
    print("=" * 80)
    print(f"{'Model':<35} {'MultiTurn':>10} {'Fail':>8} {'Fail%':>8}")
    print("-" * 65)
    for model in sorted(model_stats.keys(), key=lambda m: MODEL_TIERS.get(m, 'zzz')):
        ms = model_stats[model]
        n = ms["total_multi"]
        if n < 50:
            continue
        fail_pct = 100 * ms["with_any_failure"] / n
        tier = MODEL_TIERS.get(model, '?')
        print(f"{model:<35} {n:>10,} {ms['with_any_failure']:>8,} {fail_pct:>7.1f}%  [{tier}]")

    # Save
    results = {
        "metadata": {
            "dataset": "WildChat-4.8M",
            "filter": "English, multi-turn only (2+ user turns)",
            "total_conversations": total,
            "multi_turn_conversations": multi_turn,
            "elapsed_seconds": round(time.time() - t0, 1),
        },
        "tier_stats": {
            tier: {bucket: dict(stats) for bucket, stats in buckets.items()}
            for tier, buckets in tier_stats.items()
        },
        "model_stats": dict(model_stats),
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
