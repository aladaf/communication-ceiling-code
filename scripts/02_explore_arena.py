"""
Capability Overhang — LMSYS Chatbot Arena Analysis

Core hypothesis: When users submit simple prompts, they cannot distinguish
between weak and strong models (→ more ties = capability overhang).
When prompts are complex, the stronger model wins more often.

This directly measures overhang: the gap between capability and perceived value.

Usage:
    python scripts/02_explore_arena.py
"""

import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
RESULTS_FILE = DATA_DIR / "arena_overhang_results.json"

# --- Model capability tiers (based on known Arena ELO rankings) ---
# Grouped into 4 tiers: weak, medium, strong, frontier
MODEL_TIERS = {
    # Tier 1 — WEAK (early/small open-source)
    "chatglm-6b": "weak",
    "dolly-v2-12b": "weak",
    "fastchat-t5-3b": "weak",
    "stablelm-tuned-alpha-7b": "weak",
    "oasst-pythia-12b": "weak",
    "llama-13b": "weak",
    "mpt-7b-chat": "weak",

    # Tier 2 — MEDIUM (competent open-source)
    "koala-13b": "medium",
    "alpaca-13b": "medium",
    "vicuna-7b": "medium",
    "vicuna-13b": "medium",
    "guanaco-33b": "medium",
    "mpt-30b-chat": "medium",
    "wizardlm-13b": "medium",
    "RWKV-4-Raven-14B": "medium",

    # Tier 3 — STRONG (top open-source & commercial)
    "vicuna-33b": "strong",
    "claude-instant-v1": "strong",
    "gpt-3.5-turbo": "strong",
    "palm-2": "strong",
    "claude-v1": "strong",
    "guanaco-65b": "strong",
    "wizardlm-30b": "strong",

    # Tier 4 — FRONTIER (best available at the time)
    "gpt-4": "frontier",
    "claude-2.0": "frontier",
    "claude-2": "frontier",
}

TIER_RANK = {"weak": 0, "medium": 1, "strong": 2, "frontier": 3}
TIER_ORDER = ["weak", "medium", "strong", "frontier"]

# --- Bloom's taxonomy classifier (same as WildChat, refined) ---
BLOOM_PATTERNS = {
    "L1_remember": [
        r"\b(what is|what are|who is|who are|when did|when was|where is|where are)\b",
        r"\b(define|list|name|state|recall|identify)\b",
        r"\b(tell me about|what does .+ mean)\b",
    ],
    "L2_understand": [
        r"\b(explain|summarize|paraphrase|interpret|classify|compare)\b",
        r"\b(in your own words|what is the difference|how does .+ work)\b",
        r"\b(translate|convert|rewrite)\b",
    ],
    "L3_apply": [
        r"\b(write|code|generate|make|build|implement|solve|calculate)\b",
        r"\b(give me|show me|produce|draft|compose)\b",
        r"\b(use .+ to|apply .+ to|how (do|can|would|should) i)\b",
    ],
    "L4_analyze": [
        r"\b(analyze|examine|investigate|break down|categorize|distinguish)\b",
        r"\b(why does|why is|why are|what causes|what factors)\b",
        r"\b(relationship between|pros and cons|advantages and disadvantages)\b",
    ],
    "L5_evaluate": [
        r"\b(evaluate|assess|judge|critique|review|justify|argue|defend)\b",
        r"\b(which is better|should i|is it worth|do you think|what do you recommend)\b",
        r"\b(rate|rank|prioritize)\b",
    ],
    "L6_create": [
        r"\b(devise|formulate|propose|hypothesize|construct)\b",
        r"\b(original|novel|innovative|creative|unique)\b",
        r"\b(what if|imagine|suppose|how might)\b",
    ],
}

bloom_compiled = {}
for level, patterns in BLOOM_PATTERNS.items():
    bloom_compiled[level] = [re.compile(p, re.IGNORECASE) for p in patterns]

BLOOM_ORDER = ["L1_remember", "L2_understand", "L3_apply", "L4_analyze", "L5_evaluate", "L6_create"]


def classify_bloom(text):
    """Classify by highest Bloom's level matched."""
    text = text[:2000]
    highest = None
    highest_rank = -1
    for level, patterns in bloom_compiled.items():
        for p in patterns:
            if p.search(text):
                rank = BLOOM_ORDER.index(level)
                if rank > highest_rank:
                    highest_rank = rank
                    highest = level
                break
    return highest if highest else "unclassified"


def word_count(text):
    return len(text.split())


def get_capability_gap(tier_a, tier_b):
    """
    Returns (stronger_model, gap_size).
    gap_size: 0=same, 1=adjacent, 2=two apart, 3=three apart
    """
    rank_a = TIER_RANK.get(tier_a, -1)
    rank_b = TIER_RANK.get(tier_b, -1)
    if rank_a < 0 or rank_b < 0:
        return None, None
    gap = abs(rank_a - rank_b)
    if rank_a > rank_b:
        stronger = "model_a"
    elif rank_b > rank_a:
        stronger = "model_b"
    else:
        stronger = "same"
    return stronger, gap


def main():
    from datasets import load_dataset

    print("=" * 70)
    print("  LMSYS Chatbot Arena — Capability Overhang Analysis")
    print("=" * 70)
    print()
    print("Loading dataset (streaming)...")

    ds = load_dataset('lmsys/chatbot_arena_conversations', split='train', streaming=True)

    # Accumulators
    total = 0
    english = 0
    classified = 0
    unknown_models = Counter()

    # Per capability-gap × bloom-level stats
    # gap_bloom_stats[gap][bloom] = {"total": N, "stronger_wins": N, "tie": N, "weaker_wins": N}
    gap_bloom_stats = defaultdict(lambda: defaultdict(lambda: {
        "total": 0, "stronger_wins": 0, "tie": 0, "weaker_wins": 0
    }))

    # Overall bloom distribution
    bloom_total = Counter()

    # Per-gap overall stats
    gap_stats = defaultdict(lambda: {
        "total": 0, "stronger_wins": 0, "tie": 0, "weaker_wins": 0,
        "prompt_words_sum": 0
    })

    # Model pair data
    model_a_counts = Counter()
    model_b_counts = Counter()

    # Prompt complexity vs discrimination
    prompt_length_bins = defaultdict(lambda: {"total": 0, "tie": 0})

    t0 = time.time()

    for example in ds:
        total += 1

        lang = example.get('language', '')
        if lang != 'English':
            continue
        english += 1

        model_a = example.get('model_a', '')
        model_b = example.get('model_b', '')
        winner = example.get('winner', '')
        conv_a = example.get('conversation_a', [])

        # Get tier
        tier_a = MODEL_TIERS.get(model_a)
        tier_b = MODEL_TIERS.get(model_b)

        if not tier_a:
            unknown_models[model_a] += 1
        if not tier_b:
            unknown_models[model_b] += 1

        if not tier_a or not tier_b:
            continue

        # Get user prompt (first user message from conversation_a)
        user_prompt = ""
        for msg in conv_a:
            if msg.get('role') == 'user':
                user_prompt = msg.get('content', '')
                break

        if not user_prompt:
            continue

        classified += 1
        model_a_counts[model_a] += 1
        model_b_counts[model_b] += 1

        # Classify prompt
        bloom = classify_bloom(user_prompt)
        bloom_total[bloom] += 1
        prompt_words = word_count(user_prompt)

        # Determine capability gap
        stronger, gap = get_capability_gap(tier_a, tier_b)

        # Determine outcome relative to stronger model
        if stronger == "same":
            outcome = "tie" if "tie" in winner else ("model_a" if winner == "model_a" else "model_b")
            # For same-tier battles, just track tie rate
            gap_bloom_stats[0][bloom]["total"] += 1
            gap_stats[0]["total"] += 1
            gap_stats[0]["prompt_words_sum"] += prompt_words
            if "tie" in winner:
                gap_bloom_stats[0][bloom]["tie"] += 1
                gap_stats[0]["tie"] += 1
        else:
            gap_bloom_stats[gap][bloom]["total"] += 1
            gap_stats[gap]["total"] += 1
            gap_stats[gap]["prompt_words_sum"] += prompt_words

            if "tie" in winner:
                gap_bloom_stats[gap][bloom]["tie"] += 1
                gap_stats[gap]["tie"] += 1
            elif winner == stronger:
                gap_bloom_stats[gap][bloom]["stronger_wins"] += 1
                gap_stats[gap]["stronger_wins"] += 1
            else:
                gap_bloom_stats[gap][bloom]["weaker_wins"] += 1
                gap_stats[gap]["weaker_wins"] += 1

        # Prompt length binning
        if prompt_words <= 10:
            pbin = "1-10"
        elif prompt_words <= 25:
            pbin = "11-25"
        elif prompt_words <= 50:
            pbin = "26-50"
        elif prompt_words <= 100:
            pbin = "51-100"
        elif prompt_words <= 200:
            pbin = "101-200"
        else:
            pbin = "200+"

        prompt_length_bins[pbin]["total"] += 1
        if "tie" in winner:
            prompt_length_bins[pbin]["tie"] += 1

        if total % 5000 == 0:
            print(f"  {total:,} processed, {classified:,} classified...")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Total: {total:,} | English: {english:,} | Classified: {classified:,}")

    # --- RESULTS ---

    print("\n" + "=" * 70)
    print("  RESULT 1: TIE RATE BY CAPABILITY GAP")
    print("  Do users distinguish models when the gap is larger?")
    print("=" * 70)
    print(f"{'Gap':>5} {'Total':>8} {'Stronger%':>10} {'Tie%':>8} {'Weaker%':>9} {'AvgWords':>10}")
    print("-" * 55)
    for gap in sorted(gap_stats.keys()):
        s = gap_stats[gap]
        n = s["total"]
        if n == 0:
            continue
        sw = 100 * s["stronger_wins"] / n if gap > 0 else 0
        tie = 100 * s["tie"] / n
        ww = 100 * s["weaker_wins"] / n if gap > 0 else 0
        avg_w = s["prompt_words_sum"] / n
        label = "same" if gap == 0 else f"+{gap}"
        print(f"{label:>5} {n:>8,} {sw:>9.1f}% {tie:>7.1f}% {ww:>8.1f}% {avg_w:>10.1f}")

    print("\n" + "=" * 70)
    print("  RESULT 2: TIE RATE BY BLOOM LEVEL × CAPABILITY GAP")
    print("  Is overhang concentrated in low-complexity prompts?")
    print("=" * 70)
    for gap in sorted(gap_bloom_stats.keys()):
        label = "SAME TIER" if gap == 0 else f"GAP +{gap}"
        print(f"\n--- {label} ---")
        print(f"{'Bloom':<16} {'Total':>8} {'Tie%':>8} {'Stronger%':>10} {'Weaker%':>9}")
        print("-" * 55)
        for bloom in BLOOM_ORDER + ["unclassified"]:
            s = gap_bloom_stats[gap][bloom]
            n = s["total"]
            if n < 5:
                continue
            tie = 100 * s["tie"] / n
            sw = 100 * s["stronger_wins"] / n
            ww = 100 * s["weaker_wins"] / n
            print(f"{bloom:<16} {n:>8,} {tie:>7.1f}% {sw:>9.1f}% {ww:>8.1f}%")

    print("\n" + "=" * 70)
    print("  RESULT 3: TIE RATE BY PROMPT LENGTH")
    print("  Do shorter/simpler prompts produce more ties?")
    print("=" * 70)
    bin_order = ["1-10", "11-25", "26-50", "51-100", "101-200", "200+"]
    print(f"{'Words':<12} {'Total':>8} {'Tie%':>8}")
    print("-" * 30)
    for b in bin_order:
        s = prompt_length_bins.get(b, {"total": 0, "tie": 0})
        n = s["total"]
        if n == 0:
            continue
        tie = 100 * s["tie"] / n
        print(f"{b:<12} {n:>8,} {tie:>7.1f}%")

    print("\n" + "=" * 70)
    print("  BLOOM LEVEL DISTRIBUTION (all classified)")
    print("=" * 70)
    for bloom in BLOOM_ORDER + ["unclassified"]:
        c = bloom_total.get(bloom, 0)
        pct = 100 * c / classified if classified > 0 else 0
        print(f"  {bloom:<16} {c:>6,} ({pct:.1f}%)")

    if unknown_models:
        print(f"\n--- Unknown models (top 20, not in tier map) ---")
        for m, c in unknown_models.most_common(20):
            print(f"  {m:<35} {c:>5,}")

    # Save results
    results = {
        "metadata": {
            "dataset": "lmsys/chatbot_arena_conversations",
            "total": total,
            "english": english,
            "classified": classified,
            "elapsed_seconds": round(elapsed, 1),
        },
        "gap_stats": {str(k): v for k, v in gap_stats.items()},
        "gap_bloom_stats": {
            str(gap): {bloom: dict(stats) for bloom, stats in blooms.items()}
            for gap, blooms in gap_bloom_stats.items()
        },
        "bloom_distribution": dict(bloom_total),
        "prompt_length_bins": dict(prompt_length_bins),
        "unknown_models": dict(unknown_models.most_common(30)),
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
