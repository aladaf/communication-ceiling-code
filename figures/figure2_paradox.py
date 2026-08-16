#!/usr/bin/env python3
"""
Figure 2: The User Experience Paradox
Two-panel figure showing that user experience transforms prompting behavior
(17x longer prompts) but not communication outcomes (equivalent failure rates).

Panel (a): Failure rates by experience level (95% CI error bars)
Panel (b): Prompt length and raw TTR by experience level

Revision notes (Frontiers R1): rates, CIs, prompt means and TTR are computed
from the saved analysis counts (data/user_skill_results.json); interpretive
overlays ("Ceiling zone", "explodes" banner) removed; Ns shown.
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# ── Nature-style configuration ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

COLORS = {
    '11+ turns':  '#E63946',
    '6-10 turns': '#F4A261',
    '4-5 turns':  '#457B9D',
    '2-3 turns':  '#A8DADC',
}
DARK_BLUE = '#1D3557'
Z95 = 1.959963984540054

# ── Load counts ─────────────────────────────────────────────────────────────
skill = json.loads((DATA / "user_skill_results.json").read_text())
exp = skill["exp_stats"]

experience_levels = ['1', '2-3', '4-10', '11-30', '31-100', '100+']
experience_labels_full = [
    '1\nconv.', '2-3\nconv.', '4-10\nconv.',
    '11-30\nconv.', '31-100\nconv.', '100+\nconv.'
]
STRATA = {'2-3 turns': '2-3', '4-5 turns': '4-5',
          '6-10 turns': '6-10', '11+ turns': '11+'}


def rate_ci(level, stratum_key):
    c = exp[level][stratum_key]
    n, x = c['total'], c['with_failure']
    p = x / n
    half = Z95 * math.sqrt(p * (1 - p) / n)
    return 100 * p, 100 * half


# Prompt characteristics at the 6-10 turn stratum (as in Table 5)
prompt_words = [exp[l]['6-10']['prompt_words_sum'] / exp[l]['6-10']['total']
                for l in experience_levels]
ttr_values = [exp[l]['6-10']['prompt_ttr_sum'] / exp[l]['6-10']['total']
              for l in experience_levels]
n_multiturn = sum(exp[l][s]['total'] for l in experience_levels
                  for s in STRATA.values())

# ── Figure ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                         gridspec_kw={'wspace': 0.35})

x = np.arange(len(experience_levels))
markers = ['o', 'D', 's', '^']
turn_categories = ['11+ turns', '6-10 turns', '4-5 turns', '2-3 turns']

# ── Panel (a): Failure rates by experience ─────────────────────────────────
ax = axes[0]

for i, turn_cat in enumerate(turn_categories):
    pts = [rate_ci(l, STRATA[turn_cat]) for l in experience_levels]
    vals = [p for p, _ in pts]
    errs = [e for _, e in pts]
    ax.errorbar(x, vals, yerr=errs, marker=markers[i], markersize=7,
                linewidth=2.2, color=COLORS[turn_cat], label=turn_cat,
                zorder=3, capsize=2.5, elinewidth=0.9, capthick=0.9)

# Factual novice vs power-user difference at 6-10 turns
p1, _ = rate_ci('1', '6-10')
p100, _ = rate_ci('100+', '6-10')
ax.annotate('', xy=(5, p100), xytext=(0, p1),
            arrowprops=dict(arrowstyle='-', color=DARK_BLUE, lw=0.8,
                            linestyle=':'))
ax.text(2.5, (p1 + p100) / 2 - 2.6,
        f'1 vs 100+ conv.: Δ = {p1 - p100:+.1f} pp\n(equivalent within ±2 pp, TOST p = .007)',
        fontsize=6.5, color=DARK_BLUE, ha='center')

ax.set_xticks(x)
ax.set_xticklabels(experience_labels_full, fontsize=7)
ax.set_ylabel('Communication failure rate (%)', fontsize=10)
ax.set_xlabel('User experience level', fontsize=9)
ax.set_ylim(-1, 38)
ax.set_title('a', loc='left', fontsize=14, fontweight='bold', x=-0.08)
ax.text(0.5, 1.0, f'Failure rates by experience (N = {n_multiturn:,})',
        transform=ax.transAxes, ha='center', va='bottom', fontsize=9,
        fontstyle='italic', color='gray')
ax.legend(loc='upper left', framealpha=0.9, edgecolor='#DDD',
          borderpad=0.5, labelspacing=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
ax.grid(axis='y', alpha=0.2, linewidth=0.5)

# ── Panel (b): Prompt length and TTR ───────────────────────────────────────
ax = axes[1]

n_bars = len(prompt_words)
cmap = plt.colormaps['YlGnBu']
bar_colors = [cmap(0.25 + 0.6 * i / (n_bars - 1)) for i in range(n_bars)]

bars = ax.bar(x, prompt_words, width=0.6, color=bar_colors,
              edgecolor='white', linewidth=0.8, zorder=3)

for bar, val in zip(bars, prompt_words):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
            f'{val:,.0f}', ha='center', va='bottom', fontsize=8,
            fontweight='bold', color=DARK_BLUE)

ratio = prompt_words[-1] / prompt_words[0]
ax.annotate(f'×{ratio:.0f}', xy=(5, prompt_words[-1]), xytext=(4, 2500),
            fontsize=14, fontweight='bold', color='#E63946',
            arrowprops=dict(arrowstyle='->', color='#E63946', lw=2),
            ha='center')

ax2 = ax.twinx()
ax2.plot(x, ttr_values, marker='o', markersize=5, linewidth=1.8,
         color='#E63946', linestyle='--', zorder=4, label='Raw TTR')
ax2.set_ylabel('Raw type-token ratio (TTR)', fontsize=9, color='#E63946')
ax2.tick_params(axis='y', colors='#E63946')
ax2.set_ylim(0.65, 0.95)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_color('#E63946')

ax.set_xticks(x)
ax.set_xticklabels(experience_labels_full, fontsize=7)
ax.set_ylabel('Mean first-prompt length (words)', fontsize=10)
ax.set_xlabel('User experience level', fontsize=9)
ax.set_ylim(0, 3300)
ax.set_title('b', loc='left', fontsize=14, fontweight='bold', x=-0.08)
ax.text(0.5, 1.0, 'Prompt length and raw TTR (6-10 turn conversations)',
        transform=ax.transAxes, ha='center', va='bottom', fontsize=9,
        fontstyle='italic', color='gray')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, linewidth=0.5)

# ── Save ────────────────────────────────────────────────────────────────────
output_dir = Path(__file__).parent
fig.savefig(output_dir / 'figure2_paradox.png', dpi=300, facecolor='white')
fig.savefig(output_dir / 'figure2_paradox.pdf', facecolor='white')
print(f"Figure 2 saved to {output_dir / 'figure2_paradox.png'}")
