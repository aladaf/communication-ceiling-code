# Reproducing the Results

This document gives the exact sequence to reproduce every table, figure, and statistical test in the paper. There are two levels:

- **Level 1 — from aggregate results (minutes, no raw data).** The repository ships the aggregate counts (`data/*.json`) produced by the corpus-processing scripts. All statistics and figures in the paper can be reproduced from these alone.
- **Level 2 — from raw corpora (hours, ~20 GB download).** Re-run the corpus processing from scratch and verify the aggregate counts themselves.

## Prerequisites

```bash
python -m pip install -r requirements.txt   # Python 3.11+
```

For Level 2 only:

1. Create a (free) Hugging Face account and **accept the terms** on each gated dataset page:
   - https://huggingface.co/datasets/allenai/WildChat-4.8M
   - https://huggingface.co/datasets/lmsys/lmsys-chat-1m
   - https://huggingface.co/datasets/lmsys/chatbot_arena_conversations
2. Authenticate locally: `huggingface-cli login`
3. Download WildChat (~15 GB; the other datasets stream/download on demand):

```bash
huggingface-cli download allenai/WildChat-4.8M --repo-type dataset
```

Scripts locate the WildChat cache automatically; to use a custom location, set `WILDCHAT_DIR=/path/to/parquet/dir`.

## Level 1 — statistics and figures from shipped aggregates

Run from the repository root. Each finishes in seconds.

| Step | Command | Reproduces |
| --- | --- | --- |
| 1 | `python scripts/06_stat_tests.py` | Wald 95% CIs, chi-squared tests, Cramér's V quoted in Studies 1-3 |
| 2 | `python scripts/09_equivalence_tests.py` | TOST equivalence tests (±2 pp margin): GPT-4 ≡ GPT-4o; novice ≡ 100+ users |
| 3 | `python figures/figure1_ceiling.py` | Figure 1 (panels a-c, with 95% CI error bars) |
| 4 | `python figures/figure2_paradox.py` | Figure 2 (panels a-b) |

## Level 2 — full pipeline from raw corpora

Run in this order (later scripts read nothing from earlier ones except where noted; all write their aggregate output into `data/`, overwriting the shipped copies so you can diff):

| Step | Command | Runtime* | Produces | Paper artifact |
| --- | --- | --- | --- | --- |
| 1 | `python scripts/03_failure_by_model.py` | ~7 min | `data/failure_by_model_results.json` | Table 1; per-model counts behind Supplementary Table S1 |
| 2 | `python scripts/04_failure_lmsys.py` | ~5 min (streams) | `data/lmsys_failure_results.json` | Tables 2-3 |
| 3 | `python scripts/05_user_skill.py` | ~7 min | `data/user_skill_results.json` | Tables 4-5 (means, raw TTR) |
| 4 | `python scripts/01_explore_overhang.py` | ~7 min | `data/overhang_exploration_results.json` | §4.3 (prompt length / turn counts by tier, o1 single-turn) |
| 5 | `python scripts/02_explore_arena.py` | ~5 min (streams) | `data/arena_overhang_results.json` | background analysis (not reported in the paper) |
| 6 | `python scripts/10_rerun_wildchat_revision.py` | ~5 min | `data/rerun_wildchat_results.json` | cluster-robust SEs (§3.5, §4.2, §6.2), temporal-experience re-operationalization (§6.2), prompt-length medians/IQR and MATTR-100 (Table 5), Supplementary Table S1 |
| 7 | `python scripts/13_lmsys_per_model.py` | ~3 min | `data/lmsys_per_model_results.json` | Supplementary Table S2 |
| 8 | Level 1 steps 1-4 | seconds | — | all reported statistics and figures |

\* Runtimes measured on a consumer laptop (WSL2) with the WildChat parquet cache on local disk.

The failure/repair heuristics (19 regex patterns; refinement-chain rules and thresholds) are defined once, in `scripts/03_failure_by_model.py`, and imported by scripts 10, 11, and 13 — guaranteeing that every analysis uses identical rules. LMSYS tier assignments are defined once in `scripts/04_failure_lmsys.py`.

## Human validation study (§3.2.4)

`scripts/11_build_validation_sample.py` builds the blinded, stratified 200-conversation annotation sample and the annotator workbooks (fixed seed 42 — the sample itself is exactly reproducible from the WildChat cache). `scripts/12_analyze_validation.py` computes inter-rater agreement (Cohen's κ), heuristic-vs-human precision/recall, and repair-outcome proportions from the filled workbooks.

Two inputs of that study are **deliberately not redistributed** here: (a) the blinding key (`validation_sample_key.json`), because a portion of the sample is held as a blind reserve for follow-up annotation; and (b) the filled annotator workbooks, which contain verbatim conversation texts from a gated corpus and are therefore shared only under the corpus' own access terms. Both are available from the corresponding author on reasonable request; the annotation codebook is embedded in `scripts/11_build_validation_sample.py`.

## Expected checks

After Level 2, the following headline numbers should match the paper exactly:

- WildChat: 1,679,371 English conversations; 246,642 multi-turn; GPT-3.5 25.0% vs GPT-4 15.3% vs GPT-4o 14.4% at 6-10 turns (Table 1).
- LMSYS: 777,453 English; 220,235 multi-turn; frontier 15.6% at 6-10 turns (Table 2).
- TOST: GPT-4 vs GPT-4o p = .024 (90% CI [−0.12, +1.81] pp); novice vs 100+ p = .007 (90% CI [−0.75, +1.46] pp).
