"""
Capability Overhang — LMSYS-Chat-1M Communication Failure Analysis

Same analysis as script 03 but on LMSYS-Chat-1M dataset.
This gives us different models (Vicuna, WizardLM, Claude, GPT-4, etc.)
to confirm that the failure rate plateau is not GPT-specific.

Usage:
    python scripts/04_failure_lmsys.py
"""

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
RESULTS_FILE = DATA_DIR / "lmsys_failure_results.json"

# --- Model tiers for LMSYS ---
# Based on known Arena ELO scores at the time of data collection (2023)
MODEL_TIERS = {
    # Tier 1 — WEAK (small/early open-source, ELO ~800-900)
    "chatglm-6b": "weak",
    "dolly-v2-12b": "weak",
    "fastchat-t5-3b": "weak",
    "stablelm-tuned-alpha-7b": "weak",
    "oasst-pythia-12b": "weak",
    "llama-13b": "weak",
    "mpt-7b-chat": "weak",

    # Tier 2 — MEDIUM (competent open-source, ELO ~900-1000)
    "koala-13b": "medium",
    "alpaca-13b": "medium",
    "vicuna-7b": "medium",
    "RWKV-4-Raven-14B": "medium",

    # Tier 3 — STRONG (best open-source + early commercial, ELO ~1000-1100)
    "vicuna-13b": "strong",
    "vicuna-33b": "strong",
    "guanaco-33b": "strong",
    "guanaco-65b": "strong",
    "mpt-30b-chat": "strong",
    "wizardlm-13b": "strong",
    "wizardlm-30b": "strong",
    "claude-instant-v1": "strong",
    "palm-2": "strong",

    # Tier 4 — FRONTIER (best at the time, ELO ~1100+)
    "gpt-3.5-turbo": "frontier",
    "gpt-4": "frontier",
    "claude-v1": "frontier",
    "claude-2": "frontier",
    "claude-2.0": "frontier",
}

TIER_ORDER = ["weak", "medium", "strong", "frontier"]

# --- Failure markers (same as Epistemic Apertures) ---
FAILURE_MARKERS = [
    r"that'?s not what i mean",
    r"no,?\s*i mean",
    r"what i('m| am) trying to say",
    r"let me (try|explain|rephrase|clarify|re-?word)",
    r"i('ll| will) try (again|to explain|to rephrase)",
    r"i can'?t (explain|describe|put into words|articulate)",
    r"(hard|difficult) to (explain|describe|put into words|articulate|express)",
    r"(sorry|apologies),?\s*(let me|i('ll| will)) (try again|rephrase|clarify)",
    r"that came out wrong",
    r"i('m| am) not (explaining|describing) (this|it) well",
    r"does that make sense",
    r"am i making (any )?sense",
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
        return None

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
    from datasets import load_dataset

    print("=" * 70)
    print("  LMSYS-Chat-1M — Communication Failure by Model Tier")
    print("  (Multi-turn conversations only)")
    print("=" * 70)
    print()
    print("Loading dataset (streaming)...")

    ds = load_dataset('lmsys/lmsys-chat-1m', split='train', streaming=True)

    tier_stats = defaultdict(lambda: defaultdict(lambda: {
        "total": 0,
        "with_any_failure": 0,
        "with_failure_markers": 0,
        "with_refinement_chains": 0,
    }))

    model_stats = defaultdict(lambda: {"total_multi": 0, "with_any_failure": 0, "total_all": 0})

    total = 0
    english = 0
    multi_turn = 0
    unknown_models = Counter()
    t0 = time.time()

    for example in ds:
        total += 1

        lang = example.get('language', '')
        if lang != 'English':
            continue
        english += 1

        model = example.get('model', '')
        tier = MODEL_TIERS.get(model)

        if not tier:
            unknown_models[model] += 1
            continue

        conversation = example.get('conversation', [])
        if not conversation or not isinstance(conversation, list):
            continue

        model_stats[model]["total_all"] += 1

        result = analyze_conversation(conversation)
        if result is None:
            continue

        multi_turn += 1
        n_turns = result['n_user_turns']

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

        ms = model_stats[model]
        ms["total_multi"] += 1
        if result['has_any_failure']:
            ms["with_any_failure"] += 1

        if total % 100000 == 0:
            elapsed = time.time() - t0
            print(f"  {total:,} processed | English: {english:,} | Multi-turn: {multi_turn:,} | {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Total: {total:,} | English: {english:,} | Multi-turn: {multi_turn:,}")

    # --- RESULTS ---
    print("\n" + "=" * 70)
    print("  RESULT 1: FAILURE RATE BY TIER (all multi-turn)")
    print("=" * 70)

    header = f"{'Tier':<12} {'MultiTurn':>10} {'AnyFail':>10} {'Fail%':>8} {'Mark%':>8} {'Chain%':>8}"
    print(header)
    print("-" * 60)

    for tier in TIER_ORDER:
        if tier not in tier_stats:
            continue
        total_t = sum(b["total"] for b in tier_stats[tier].values())
        fail_t = sum(b["with_any_failure"] for b in tier_stats[tier].values())
        mark_t = sum(b["with_failure_markers"] for b in tier_stats[tier].values())
        chain_t = sum(b["with_refinement_chains"] for b in tier_stats[tier].values())
        if total_t == 0:
            continue
        print(f"{tier:<12} {total_t:>10,} {fail_t:>10,} {100*fail_t/total_t:>7.1f}% "
              f"{100*mark_t/total_t:>7.1f}% {100*chain_t/total_t:>7.1f}%")

    print("\n" + "=" * 70)
    print("  RESULT 2: FAILURE RATE BY TIER × TURN COUNT")
    print("=" * 70)

    turn_buckets = ["2", "3", "4-5", "6-10", "11+"]
    for bucket in turn_buckets:
        print(f"\n--- {bucket} user turns ---")
        print(f"{'Tier':<12} {'N':>8} {'Fail%':>8} {'Mark%':>8} {'Chain%':>8}")
        print("-" * 48)
        for tier in TIER_ORDER:
            if tier not in tier_stats or bucket not in tier_stats[tier]:
                continue
            s = tier_stats[tier][bucket]
            n = s["total"]
            if n < 20:
                continue
            f_pct = 100 * s["with_any_failure"] / n
            m_pct = 100 * s["with_failure_markers"] / n
            c_pct = 100 * s["with_refinement_chains"] / n
            print(f"{tier:<12} {n:>8,} {f_pct:>7.1f}% {m_pct:>7.1f}% {c_pct:>7.1f}%")

    print("\n" + "=" * 70)
    print("  RESULT 3: PER-MODEL FAILURE RATE")
    print("=" * 70)
    print(f"{'Model':<30} {'All':>8} {'Multi':>8} {'Fail':>8} {'Fail%':>8} {'Tier':<10}")
    print("-" * 76)
    for model in sorted(model_stats.keys(), key=lambda m: TIER_ORDER.index(MODEL_TIERS.get(m, 'weak'))):
        ms = model_stats[model]
        tier = MODEL_TIERS.get(model, '?')
        n = ms["total_multi"]
        if n < 30:
            continue
        fail_pct = 100 * ms["with_any_failure"] / n
        print(f"{model:<30} {ms['total_all']:>8,} {n:>8,} {ms['with_any_failure']:>8,} {fail_pct:>7.1f}% {tier}")

    if unknown_models:
        print(f"\n--- Unknown models (top 10) ---")
        for m, c in unknown_models.most_common(10):
            print(f"  {m:<30} {c:>6,}")

    # Save
    results = {
        "metadata": {
            "dataset": "lmsys/lmsys-chat-1m",
            "total": total,
            "english": english,
            "multi_turn": multi_turn,
            "elapsed_seconds": round(elapsed, 1),
        },
        "tier_stats": {
            tier: {bucket: dict(stats) for bucket, stats in buckets.items()}
            for tier, buckets in tier_stats.items()
        },
        "model_stats": {m: dict(s) for m, s in model_stats.items()},
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
