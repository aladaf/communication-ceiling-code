"""
Capability Overhang Exploration — WildChat-4.8M
Three studies in one pass:

Study 1 — Cross-Model Flattening:
    Does user behavior change when the model is more capable?
    Compare prompt length, vocabulary richness, turn count across model tiers.

Study 2 — Cognitive Demand Distribution:
    What level of cognitive demand do users place on each model tier?
    Classify prompts using lexical proxies for Bloom's taxonomy levels.

Study 3 — Prompt/Response Asymmetry:
    How much does the user "invest" vs. what the model "produces"?
    Measure character/word ratios between user prompts and model responses.

Usage:
    python scripts/01_explore_overhang.py          # run full analysis
    python scripts/01_explore_overhang.py --quick   # run on 5 files only (test)
"""

import json
import re
import sys
import time
import math
from collections import Counter, defaultdict
from pathlib import Path

# --- Paths ---
CACHE_DIR = Path(
    "/home/aladaf/.cache/huggingface/hub/datasets--allenai--WildChat-4.8M"
    "/snapshots/c827c6df8fcf008219ffaffa4d1dd77491099367/data"
)
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
RESULTS_FILE = DATA_DIR / "overhang_exploration_results.json"

# --- Model tiers (by capability level) ---
MODEL_TIERS = {
    # Tier 1: GPT-3.5
    "gpt-3.5-turbo-0301": "gpt-3.5",
    "gpt-3.5-turbo-0613": "gpt-3.5",
    "gpt-3.5-turbo-0125": "gpt-3.5",
    # Tier 2: GPT-4 (original)
    "gpt-4-0314": "gpt-4",
    "gpt-4-0613": "gpt-4",
    "gpt-4-1106-preview": "gpt-4",
    "gpt-4-0125-preview": "gpt-4",
    "gpt-4-turbo-2024-04-09": "gpt-4",
    # Tier 3: GPT-4o family (multimodal, faster, but mini = cheaper/smaller)
    "gpt-4o-2024-05-13": "gpt-4o",
    "gpt-4o-2024-08-06": "gpt-4o",
    "gpt-4o-2024-11-20": "gpt-4o",
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "gpt-4.1-mini-2025-04-14": "gpt-4.1-mini",
    # Tier 4: o1 (reasoning models)
    "o1-mini-2024-09-12": "o1",
    "o1-preview-2024-09-12": "o1",
}

# Ordered by capability for display
TIER_ORDER = ["gpt-3.5", "gpt-4", "gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "o1"]

# --- Cognitive demand lexical proxies (simplified Bloom's) ---
# We classify the FIRST user message by highest matching level.
# Levels: remember, understand, apply, analyze, evaluate, create

BLOOM_PATTERNS = {
    "L1_remember": [
        r"\b(what is|what are|who is|who are|when did|when was|where is|where are)\b",
        r"\b(define|list|name|state|recall|identify|describe)\b",
        r"\b(tell me about|what does .+ mean)\b",
    ],
    "L2_understand": [
        r"\b(explain|summarize|paraphrase|interpret|classify|compare)\b",
        r"\b(in your own words|what is the difference|how does .+ work)\b",
        r"\b(translate|convert|rewrite)\b",
    ],
    "L3_apply": [
        r"\b(write|code|generate|create|make|build|implement|solve|calculate)\b",
        r"\b(give me|show me|produce|draft|compose)\b",
        r"\b(use .+ to|apply .+ to|how (do|can|would|should) i)\b",
    ],
    "L3_apply_code": [
        r"\b(python|javascript|java|html|css|sql|code|function|script|program|api)\b",
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
        r"\b(design|invent|devise|formulate|propose|hypothesize|construct)\b",
        r"\b(original|novel|innovative|creative|unique)\b",
        r"\b(what if|imagine|suppose|how might)\b",
        r"\b(plan|strategy|framework|architecture|system)\b",
    ],
}

# Compile patterns
bloom_compiled = {}
for level, patterns in BLOOM_PATTERNS.items():
    bloom_compiled[level] = [re.compile(p, re.IGNORECASE) for p in patterns]

# Simplified level mapping for aggregation
BLOOM_SIMPLIFIED = {
    "L1_remember": "L1_remember",
    "L2_understand": "L2_understand",
    "L3_apply": "L3_apply",
    "L3_apply_code": "L3_apply",
    "L4_analyze": "L4_analyze",
    "L5_evaluate": "L5_evaluate",
    "L6_create": "L6_create",
}

BLOOM_ORDER = ["L1_remember", "L2_understand", "L3_apply", "L4_analyze", "L5_evaluate", "L6_create"]


def classify_bloom(text):
    """Classify text by highest Bloom's level matched."""
    text = text[:2000]  # limit for speed
    highest = None
    highest_rank = -1
    
    for level, patterns in bloom_compiled.items():
        for p in patterns:
            if p.search(text):
                simplified = BLOOM_SIMPLIFIED[level]
                rank = BLOOM_ORDER.index(simplified) if simplified in BLOOM_ORDER else -1
                if rank > highest_rank:
                    highest_rank = rank
                    highest = simplified
                break
    
    return highest if highest else "unclassified"


def vocabulary_richness(text):
    """Type-Token Ratio (TTR) as a measure of vocabulary diversity."""
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if len(words) < 5:
        return None
    return len(set(words)) / len(words)


def word_count(text):
    """Count words in text."""
    return len(text.split())


def analyze_conversation_overhang(conversation_msgs, model):
    """Analyze a single conversation for all three studies."""
    tier = MODEL_TIERS.get(model)
    if not tier:
        return None
    
    user_messages = []
    assistant_messages = []
    
    for msg in conversation_msgs:
        role = msg.get('role', '')
        content = msg.get('content', '')
        if not isinstance(content, str):
            continue
        if role == 'user':
            user_messages.append(content)
        elif role == 'assistant':
            assistant_messages.append(content)
    
    if not user_messages:
        return None
    
    first_user = user_messages[0]
    
    # --- Study 1: Cross-model flattening metrics ---
    user_total_words = sum(word_count(m) for m in user_messages)
    user_total_chars = sum(len(m) for m in user_messages)
    first_prompt_words = word_count(first_user)
    first_prompt_chars = len(first_user)
    ttr = vocabulary_richness(' '.join(user_messages))
    n_user_turns = len(user_messages)
    
    # --- Study 2: Cognitive demand ---
    bloom_level = classify_bloom(first_user)
    
    # --- Study 3: Prompt/Response asymmetry ---
    assistant_total_words = sum(word_count(m) for m in assistant_messages)
    assistant_total_chars = sum(len(m) for m in assistant_messages)
    
    if user_total_words > 0:
        word_ratio = assistant_total_words / user_total_words
    else:
        word_ratio = None
    
    return {
        "tier": tier,
        "n_user_turns": n_user_turns,
        "n_assistant_turns": len(assistant_messages),
        # Study 1
        "first_prompt_words": first_prompt_words,
        "first_prompt_chars": first_prompt_chars,
        "user_total_words": user_total_words,
        "ttr": ttr,
        # Study 2
        "bloom_level": bloom_level,
        # Study 3
        "assistant_total_words": assistant_total_words,
        "word_ratio": word_ratio,
    }


def init_tier_stats():
    """Initialize stats dict for a tier."""
    return {
        "count": 0,
        # Study 1: distributions (we store sums for means + histograms)
        "first_prompt_words_sum": 0,
        "first_prompt_words_sq_sum": 0,  # for std dev
        "first_prompt_words_hist": Counter(),  # bucketed
        "user_total_words_sum": 0,
        "n_user_turns_sum": 0,
        "n_user_turns_hist": Counter(),
        "ttr_sum": 0.0,
        "ttr_count": 0,
        # Study 2: Bloom distribution
        "bloom_counts": Counter(),
        # Study 3: asymmetry
        "assistant_total_words_sum": 0,
        "word_ratio_sum": 0.0,
        "word_ratio_count": 0,
        "word_ratio_hist": Counter(),
    }


def bucket_value(value, buckets):
    """Place value into a bucket."""
    for b in buckets:
        if value <= b:
            return str(b)
    return str(buckets[-1]) + "+"


# Buckets for histograms
WORD_BUCKETS = [5, 10, 20, 50, 100, 200, 500, 1000]
TURN_BUCKETS = [1, 2, 3, 5, 10, 20]
RATIO_BUCKETS = [0.5, 1, 2, 5, 10, 20, 50, 100]


def update_tier_stats(stats, result):
    """Update tier statistics with one conversation result."""
    stats["count"] += 1
    
    # Study 1
    fpw = result["first_prompt_words"]
    stats["first_prompt_words_sum"] += fpw
    stats["first_prompt_words_sq_sum"] += fpw * fpw
    stats["first_prompt_words_hist"][bucket_value(fpw, WORD_BUCKETS)] += 1
    stats["user_total_words_sum"] += result["user_total_words"]
    stats["n_user_turns_sum"] += result["n_user_turns"]
    stats["n_user_turns_hist"][bucket_value(result["n_user_turns"], TURN_BUCKETS)] += 1
    
    if result["ttr"] is not None:
        stats["ttr_sum"] += result["ttr"]
        stats["ttr_count"] += 1
    
    # Study 2
    stats["bloom_counts"][result["bloom_level"]] += 1
    
    # Study 3
    stats["assistant_total_words_sum"] += result["assistant_total_words"]
    if result["word_ratio"] is not None:
        stats["word_ratio_sum"] += result["word_ratio"]
        stats["word_ratio_count"] += 1
        stats["word_ratio_hist"][bucket_value(result["word_ratio"], RATIO_BUCKETS)] += 1


def compute_summaries(all_stats):
    """Compute summary statistics from raw accumulators."""
    summaries = {}
    for tier in TIER_ORDER:
        s = all_stats[tier]
        n = s["count"]
        if n == 0:
            continue
        
        mean_fpw = s["first_prompt_words_sum"] / n
        var_fpw = (s["first_prompt_words_sq_sum"] / n) - (mean_fpw ** 2)
        std_fpw = math.sqrt(max(0, var_fpw))
        
        summaries[tier] = {
            "n": n,
            # Study 1
            "mean_first_prompt_words": round(mean_fpw, 1),
            "std_first_prompt_words": round(std_fpw, 1),
            "median_first_prompt_words_bucket": max(s["first_prompt_words_hist"], key=s["first_prompt_words_hist"].get) if s["first_prompt_words_hist"] else None,
            "mean_user_total_words": round(s["user_total_words_sum"] / n, 1),
            "mean_n_user_turns": round(s["n_user_turns_sum"] / n, 2),
            "turn_distribution": dict(sorted(s["n_user_turns_hist"].items(), key=lambda x: int(x[0].replace('+','')))),
            "mean_ttr": round(s["ttr_sum"] / s["ttr_count"], 4) if s["ttr_count"] > 0 else None,
            # Study 2
            "bloom_distribution": {
                level: s["bloom_counts"].get(level, 0)
                for level in BLOOM_ORDER + ["unclassified"]
            },
            "bloom_pct": {
                level: round(100 * s["bloom_counts"].get(level, 0) / n, 1)
                for level in BLOOM_ORDER + ["unclassified"]
            },
            # Study 3
            "mean_assistant_total_words": round(s["assistant_total_words_sum"] / n, 1),
            "mean_word_ratio": round(s["word_ratio_sum"] / s["word_ratio_count"], 2) if s["word_ratio_count"] > 0 else None,
            "word_ratio_distribution": dict(sorted(
                s["word_ratio_hist"].items(),
                key=lambda x: float(x[0].replace('+',''))
            )) if s["word_ratio_hist"] else {},
        }
    
    return summaries


def print_results(summaries):
    """Print formatted results for all three studies."""
    
    print("\n" + "=" * 80)
    print("  STUDY 1: CROSS-MODEL FLATTENING")
    print("  Does user behavior change with model capability?")
    print("=" * 80)
    
    header = f"{'Tier':<14} {'N':>10} {'PromptW':>9} {'±StdDev':>9} {'TotalW':>9} {'Turns':>7} {'TTR':>7}"
    print(header)
    print("-" * len(header))
    
    for tier in TIER_ORDER:
        if tier not in summaries:
            continue
        s = summaries[tier]
        print(f"{tier:<14} {s['n']:>10,} {s['mean_first_prompt_words']:>9.1f} "
              f"{s['std_first_prompt_words']:>9.1f} {s['mean_user_total_words']:>9.1f} "
              f"{s['mean_n_user_turns']:>7.2f} {s['mean_ttr'] or 0:>7.4f}")
    
    print("\n" + "=" * 80)
    print("  STUDY 2: COGNITIVE DEMAND DISTRIBUTION")
    print("  What level of complexity do users demand from each model?")
    print("=" * 80)
    
    header = f"{'Tier':<14} {'N':>10} {'Remember':>9} {'Underst':>9} {'Apply':>9} {'Analyze':>9} {'Evaluate':>9} {'Create':>9} {'Unclass':>9}"
    print(header)
    print("-" * len(header))
    
    for tier in TIER_ORDER:
        if tier not in summaries:
            continue
        s = summaries[tier]
        bp = s['bloom_pct']
        print(f"{tier:<14} {s['n']:>10,} "
              f"{bp.get('L1_remember',0):>8.1f}% "
              f"{bp.get('L2_understand',0):>8.1f}% "
              f"{bp.get('L3_apply',0):>8.1f}% "
              f"{bp.get('L4_analyze',0):>8.1f}% "
              f"{bp.get('L5_evaluate',0):>8.1f}% "
              f"{bp.get('L6_create',0):>8.1f}% "
              f"{bp.get('unclassified',0):>8.1f}%")
    
    print("\n" + "=" * 80)
    print("  STUDY 3: PROMPT/RESPONSE ASYMMETRY")
    print("  How much does the model 'produce' relative to user 'investment'?")
    print("=" * 80)
    
    header = f"{'Tier':<14} {'N':>10} {'UserW':>10} {'AssistW':>10} {'Ratio':>8}"
    print(header)
    print("-" * len(header))
    
    for tier in TIER_ORDER:
        if tier not in summaries:
            continue
        s = summaries[tier]
        print(f"{tier:<14} {s['n']:>10,} {s['mean_user_total_words']:>10.1f} "
              f"{s['mean_assistant_total_words']:>10.1f} "
              f"{s['mean_word_ratio'] or 0:>8.2f}x")


def main():
    import pandas as pd
    
    quick = "--quick" in sys.argv
    total_files = 5 if quick else 86
    
    print(f"Capability Overhang Exploration — WildChat-4.8M")
    print(f"Mode: {'QUICK (5 files)' if quick else 'FULL (86 files)'}")
    print(f"Three studies: cross-model flattening, cognitive demand, prompt/response asymmetry")
    print()
    
    # Initialize per-tier stats
    all_stats = {tier: init_tier_stats() for tier in TIER_ORDER}
    total_processed = 0
    total_skipped = 0
    t0 = time.time()
    
    for file_idx in range(total_files):
        parquet_file = CACHE_DIR / f"train-{file_idx:05d}-of-00086.parquet"
        
        if not parquet_file.exists():
            print(f"  [{file_idx+1}/{total_files}] File not found, skipping.")
            continue
        
        df = pd.read_parquet(parquet_file)
        eng = df[df['language'] == 'English']
        
        file_count = 0
        for _, row in eng.iterrows():
            model = row.get('model', '')
            conversation = row.get('conversation', [])
            
            if conversation is None:
                continue
            if not isinstance(conversation, list):
                try:
                    conversation = list(conversation)
                except (TypeError, ValueError):
                    continue
            if len(conversation) == 0:
                continue
            
            result = analyze_conversation_overhang(conversation, model)
            if result is None:
                total_skipped += 1
                continue
            
            update_tier_stats(all_stats[result["tier"]], result)
            total_processed += 1
            file_count += 1
        
        elapsed = time.time() - t0
        rate = total_processed / elapsed if elapsed > 0 else 0
        print(f"  [{file_idx+1}/{total_files}] {file_count:,} convos | "
              f"Total: {total_processed:,} | {rate:,.0f}/s | {elapsed:.0f}s")
    
    # Compute summaries
    summaries = compute_summaries(all_stats)
    
    # Print results
    print_results(summaries)
    
    # Save results
    results = {
        "metadata": {
            "dataset": "WildChat-4.8M",
            "filter": "English only",
            "files_processed": total_files,
            "total_conversations": total_processed,
            "total_skipped": total_skipped,
            "elapsed_seconds": round(time.time() - t0, 1),
            "mode": "quick" if quick else "full",
        },
        "tier_order": TIER_ORDER,
        "model_to_tier": MODEL_TIERS,
        "summaries": summaries,
        # Raw counts for reproducibility
        "raw_stats": {
            tier: {
                "count": s["count"],
                "bloom_counts": dict(s["bloom_counts"]),
                "turn_hist": dict(s["n_user_turns_hist"]),
                "prompt_word_hist": dict(s["first_prompt_words_hist"]),
                "word_ratio_hist": dict(s["word_ratio_hist"]),
            }
            for tier, s in all_stats.items()
        },
    }
    
    DATA_DIR.mkdir(exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {RESULTS_FILE}")
    
    total_time = time.time() - t0
    print(f"✓ Total time: {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()
