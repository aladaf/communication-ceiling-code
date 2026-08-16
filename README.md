# The Communication Ceiling — Analysis Code

Analysis code and aggregate results for:

> **The Communication Ceiling: Neither Capability Nor Experience Reduces Failure in Human–AI Chat.** Under review at *Frontiers in Artificial Intelligence* (Language and Computation).

Three large-scale observational studies over ~2.5 million real human–AI conversations (WildChat-4.8M and LMSYS-Chat-1M) showing that the incidence of observable communicative repair plateaus beyond GPT-4-level capability and is invariant to user experience.

## Data

No raw conversation data is redistributed here. Both corpora are publicly available on Hugging Face (gated; accept the terms on each dataset page):

- WildChat-4.8M (Studies 1 and 3): `allenai/WildChat-4.8M`
- LMSYS-Chat-1M (Study 2): `lmsys/lmsys-chat-1m`

`data/` contains the **aggregate** results (counts per model/tier × conversation-length stratum, etc.) produced by the scripts — sufficient to reproduce every table, figure, and statistical test in the paper without re-processing the corpora.

## Scripts

| Script | Purpose |
| --- | --- |
| `01_explore_overhang.py` | Cross-model behavior (prompt length, turn counts) on WildChat |
| `02_explore_arena.py` | Chatbot Arena tie rates × capability gap |
| `03_failure_by_model.py` | **Core heuristics** (19 repair-marker regexes + refinement-chain rules) and failure rates by model tier (WildChat). All patterns and thresholds are defined here. |
| `04_failure_lmsys.py` | Same analysis on LMSYS-Chat-1M (tier mapping defined here) |
| `05_user_skill.py` | Failure rates × user experience (hashed-IP histories) |
| `06_stat_tests.py` | Wald 95% CIs, chi-squared tests, Cramér's V |
| `09_equivalence_tests.py` | TOST equivalence tests (±2pp margin) from aggregate counts |
| `10_rerun_wildchat_revision.py` | Revision reruns: cluster-robust SEs, temporal (prior-conversation) experience, per-model rates, prompt-length medians, MATTR-100 |
| `11_build_validation_sample.py` | Builds the blinded 200-conversation human-validation sample and annotator workbooks |
| `12_analyze_validation.py` | Inter-rater agreement (Cohen's κ), heuristic-vs-human precision/recall, repair-outcome analysis |
| `13_lmsys_per_model.py` | Per-individual-model rates on LMSYS (Supplementary Table S2) |
| `figures/figure1_ceiling.py`, `figures/figure2_paradox.py` | Paper figures (95% CI error bars), generated from `data/*.json` |

## Reproducing

See **[REPRODUCING.md](REPRODUCING.md)** for the exact end-to-end sequence (prerequisites, gated-data access, run order, expected runtimes, and the mapping from each script to each table and figure in the paper).

## Running

```bash
pip install -r requirements.txt
python scripts/03_failure_by_model.py   # expects the WildChat parquet cache locally
```

Scripts that process raw corpora expect a local Hugging Face cache (see the `CACHE_DIR` constant at the top of each script; adjust the path for your environment). Scripts `06` and `09` and the figure scripts run directly from the aggregate JSONs in `data/`.

## Failure/repair operationalization (summary)

A multi-turn conversation is flagged when either:

1. **Explicit repair markers** — any user message after the first matches one of 19 case-insensitive regex patterns (e.g., `that's not what i mean`, `let me rephrase`, `you misunderstood`; full list in `03_failure_by_model.py`), or
2. **Refinement chains** — three or more consecutive user messages, each either shorter than 300 characters starting with a correction-initiating prefix, or sharing >30% content-word overlap with the previous message.

Analyses stratify by conversation length (2, 3, 4–5, 6–10, 11+ user turns).

## License

MIT — see `LICENSE`.
