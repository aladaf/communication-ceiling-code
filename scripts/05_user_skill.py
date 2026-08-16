"""
Capability Overhang — User Competence vs Interface Constraint

Key question: Does user experience/skill reduce communication failure?
If YES → the bottleneck is user competence, not the interface
If NO → the interface constrains EVEN skilled users

Strategy:
1. Group users by experience (# conversations per hashed_ip)
2. Compare failure rates: power users (50+) vs casual (1-2)
3. Also check prompt sophistication proxies (length, vocabulary)

Usage:
    python scripts/05_user_skill.py
"""

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(
    "/home/aladaf/.cache/huggingface/hub/datasets--allenai--WildChat-4.8M"
    "/snapshots/c827c6df8fcf008219ffaffa4d1dd77491099367/data"
)
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# --- Failure markers (same as previous scripts) ---
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
    """Detect communication failure + compute prompt sophistication."""
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

    # Failure detection
    failure_marker_count = 0
    for msg in user_messages[1:]:
        for pattern in failure_patterns:
            if pattern.search(msg):
                failure_marker_count += 1
                break

    has_markers = failure_marker_count > 0

    # Refinement chains
    has_chain = False
    if n_user >= 3:
        stop_words = {'the', 'a', 'an', 'is', 'it', 'to', 'and', 'or', 'but', 'in',
                      'on', 'for', 'of', 'i', 'you', 'my', 'your', 'me', 'do', 'can',
                      'this', 'that', 'with', 'be', 'have', 'are', 'was', 'were', 'will'}
        chain_length = 1
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

    has_any_failure = has_markers or has_chain

    # Prompt sophistication: first user message
    first_prompt = user_messages[0]
    prompt_words = len(first_prompt.split())
    unique_words = len(set(first_prompt.lower().split()))
    ttr = unique_words / max(prompt_words, 1)

    return {
        'n_user_turns': n_user,
        'has_any_failure': has_any_failure,
        'has_markers': has_markers,
        'has_chain': has_chain,
        'prompt_words': prompt_words,
        'prompt_ttr': ttr,
    }


def main():
    print("=" * 70)
    print("  Phase 1: Count conversations per user (hashed_ip)")
    print("=" * 70)

    # First pass: count conversations per user
    user_conv_count = Counter()
    t0 = time.time()

    # Try to detect available columns first
    first_file = CACHE_DIR / "train-00000-of-00086.parquet"
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(first_file)
    all_cols = [f.name for f in pf.schema_arrow]
    print(f"Available columns: {all_cols}")

    # Find the IP/user column
    ip_col = None
    for candidate in ['hashed_ip', 'ip', 'conversation_hash', 'user_id']:
        if candidate in all_cols:
            ip_col = candidate
            break

    if not ip_col:
        print("ERROR: No user identifier column found!")
        print(f"Available: {all_cols}")
        return

    print(f"Using '{ip_col}' as user identifier")
    print()

    for file_idx in range(86):
        f = CACHE_DIR / f"train-{file_idx:05d}-of-00086.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f, columns=[ip_col, 'language'])
        eng = df[df['language'] == 'English']
        for uid in eng[ip_col]:
            if uid:
                user_conv_count[uid] += 1

        if (file_idx + 1) % 10 == 0:
            print(f"  [{file_idx+1}/86] Users so far: {len(user_conv_count):,} | {time.time()-t0:.0f}s")

    print(f"\nTotal unique users: {len(user_conv_count):,}")

    # User experience distribution
    exp_dist = Counter()
    for uid, count in user_conv_count.items():
        if count == 1:
            exp_dist["1"] += 1
        elif count <= 3:
            exp_dist["2-3"] += 1
        elif count <= 10:
            exp_dist["4-10"] += 1
        elif count <= 30:
            exp_dist["11-30"] += 1
        elif count <= 100:
            exp_dist["31-100"] += 1
        else:
            exp_dist["100+"] += 1

    print("\n=== User experience distribution ===")
    for bucket in ["1", "2-3", "4-10", "11-30", "31-100", "100+"]:
        print(f"  {bucket:>8} conversations: {exp_dist.get(bucket, 0):>8,} users")

    # Phase 2: Analyze failures by user experience
    print("\n" + "=" * 70)
    print("  Phase 2: Failure rate by user experience")
    print("=" * 70)

    # exp_bucket → turn_bucket → {total, with_failure}
    exp_stats = defaultdict(lambda: defaultdict(lambda: {
        "total": 0, "with_failure": 0, "prompt_words_sum": 0, "prompt_ttr_sum": 0.0
    }))

    t1 = time.time()
    total_analyzed = 0

    for file_idx in range(86):
        f = CACHE_DIR / f"train-{file_idx:05d}-of-00086.parquet"
        if not f.exists():
            continue

        cols = [ip_col, 'language', 'conversation', 'model']
        df = pd.read_parquet(f, columns=cols)
        eng = df[df['language'] == 'English']

        for _, row in eng.iterrows():
            uid = row[ip_col]
            if not uid or uid not in user_conv_count:
                continue

            conv = row['conversation']
            if conv is None:
                continue
            if not isinstance(conv, list):
                try:
                    conv = list(conv)
                except:
                    continue
            if len(conv) == 0:
                continue

            result = analyze_conversation(conv)
            if result is None:
                continue

            total_analyzed += 1
            n_conv = user_conv_count[uid]

            # Experience bucket
            if n_conv == 1:
                exp_bucket = "1"
            elif n_conv <= 3:
                exp_bucket = "2-3"
            elif n_conv <= 10:
                exp_bucket = "4-10"
            elif n_conv <= 30:
                exp_bucket = "11-30"
            elif n_conv <= 100:
                exp_bucket = "31-100"
            else:
                exp_bucket = "100+"

            # Turn bucket
            n_turns = result['n_user_turns']
            if n_turns <= 3:
                turn_bucket = "2-3"
            elif n_turns <= 5:
                turn_bucket = "4-5"
            elif n_turns <= 10:
                turn_bucket = "6-10"
            else:
                turn_bucket = "11+"

            s = exp_stats[exp_bucket][turn_bucket]
            s["total"] += 1
            if result['has_any_failure']:
                s["with_failure"] += 1
            s["prompt_words_sum"] += result['prompt_words']
            s["prompt_ttr_sum"] += result['prompt_ttr']

        if (file_idx + 1) % 10 == 0:
            print(f"  [{file_idx+1}/86] Analyzed: {total_analyzed:,} | {time.time()-t1:.0f}s")

    # --- RESULTS ---
    print(f"\nTotal multi-turn analyzed: {total_analyzed:,}")

    exp_order = ["1", "2-3", "4-10", "11-30", "31-100", "100+"]
    turn_order = ["2-3", "4-5", "6-10", "11+"]

    print("\n" + "=" * 80)
    print("  RESULT 1: FAILURE RATE BY USER EXPERIENCE × TURN COUNT")
    print("=" * 80)

    for turn_bucket in turn_order:
        print(f"\n--- {turn_bucket} user turns ---")
        print(f"{'Experience':<14} {'N':>8} {'Fail%':>8} {'AvgWords':>10} {'AvgTTR':>8}")
        print("-" * 52)
        for exp in exp_order:
            s = exp_stats[exp][turn_bucket]
            n = s["total"]
            if n < 20:
                continue
            fail_pct = 100 * s["with_failure"] / n
            avg_words = s["prompt_words_sum"] / n
            avg_ttr = s["prompt_ttr_sum"] / n
            print(f"{exp:>10} conv {n:>8,} {fail_pct:>7.1f}% {avg_words:>10.1f} {avg_ttr:>7.3f}")

    print("\n" + "=" * 80)
    print("  RESULT 2: FAILURE RATE BY EXPERIENCE (all turns combined)")
    print("=" * 80)
    print(f"{'Experience':<14} {'MultiTurn':>10} {'Fail':>8} {'Fail%':>8} {'AvgWords':>10}")
    print("-" * 54)
    for exp in exp_order:
        total_n = sum(exp_stats[exp][tb]["total"] for tb in turn_order)
        total_fail = sum(exp_stats[exp][tb]["with_failure"] for tb in turn_order)
        total_words = sum(exp_stats[exp][tb]["prompt_words_sum"] for tb in turn_order)
        if total_n < 20:
            continue
        fail_pct = 100 * total_fail / total_n
        avg_words = total_words / total_n
        print(f"{exp:>10} conv {total_n:>10,} {total_fail:>8,} {fail_pct:>7.1f}% {avg_words:>10.1f}")

    # Save
    results = {
        "metadata": {
            "dataset": "WildChat-4.8M",
            "user_id_column": ip_col,
            "total_unique_users": len(user_conv_count),
            "total_multi_turn": total_analyzed,
        },
        "user_experience_distribution": dict(exp_dist),
        "exp_stats": {
            exp: {tb: dict(s) for tb, s in buckets.items()}
            for exp, buckets in exp_stats.items()
        },
    }
    with open(DATA_DIR / "user_skill_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved")


if __name__ == "__main__":
    main()
