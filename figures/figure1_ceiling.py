#!/usr/bin/env python3
"""
Figure 1: The Communication Ceiling
Three-panel figure showing that communication failure plateaus beyond
GPT-4-level capability across two independent datasets.

Panel (a): WildChat — failure rate by model tier (95% CI error bars)
Panel (b): LMSYS — failure rate by model tier (cross-architecture replication)
Panel (c): Cross-dataset comparison at 6-10 and 11+ turns

Revision notes (Frontiers R1): rates and Wald 95% CIs are computed from the
saved analysis counts (data/*.json) rather than hardcoded; interpretive
overlays ("Ceiling zone" label and shading) removed; Ns shown per panel.
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
    '3 turns':    '#A8DADC',
    '2 turns':    '#B0B0B0',
}
DARK_BLUE = '#1D3557'
WILDCAT_COLOR = '#457B9D'
LMSYS_COLOR = '#E63946'

Z95 = 1.959963984540054

# ── Load counts and compute rates with Wald 95% CIs ─────────────────────────
wild = json.loads((DATA / "failure_by_model_results.json").read_text())
lmsys = json.loads((DATA / "lmsys_failure_results.json").read_text())

STRATA = {'2 turns': '2', '3 turns': '3', '4-5 turns': '4-5',
          '6-10 turns': '6-10', '11+ turns': '11+'}


def rate_ci(stats, tier, stratum_key):
    c = stats[tier][stratum_key]
    n, x = c['total'], c['with_any_failure']
    p = x / n
    half = Z95 * math.sqrt(p * (1 - p) / n)
    return 100 * p, 100 * half


wildchat_tiers = ['gpt-3.5', 'gpt-4', 'gpt-4o-mini', 'gpt-4.1-mini', 'gpt-4o']
wildchat_labels = ['GPT-3.5', 'GPT-4', 'GPT-4o\nmini', 'GPT-4.1\nmini', 'GPT-4o']
lmsys_tiers = ['weak', 'medium', 'strong', 'frontier']
lmsys_labels = ['Weak\n(6-14B)', 'Medium\n(13-14B)', 'Strong\n(13-65B)',
                'Frontier\n(GPT-4, Claude)']

wildchat_n = sum(wild['tier_stats'][t][s]['total']
                 for t in wildchat_tiers for s in STRATA.values())
lmsys_n = sum(lmsys['tier_stats'][t][s]['total']
              for t in lmsys_tiers for s in STRATA.values())

# ── Figure ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5),
                         gridspec_kw={'width_ratios': [5, 4, 3], 'wspace': 0.35})

turn_categories = ['11+ turns', '6-10 turns', '4-5 turns', '3 turns', '2 turns']
markers = ['o', 's', 'D', '^', 'v']


def panel_lines(ax, stats, tiers, labels):
    x = np.arange(len(tiers))
    for i, turn_cat in enumerate(turn_categories):
        vals, errs = zip(*[rate_ci(stats, t, STRATA[turn_cat]) for t in tiers])
        ax.errorbar(x, vals, yerr=errs, marker=markers[i], markersize=6,
                    linewidth=2, color=COLORS[turn_cat], label=turn_cat,
                    zorder=3, capsize=2.5, elinewidth=0.9, capthick=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(-1, 42)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.grid(axis='y', alpha=0.2, linewidth=0.5)


# ── Panel (a): WildChat ────────────────────────────────────────────────────
ax = axes[0]
panel_lines(ax, wild['tier_stats'], wildchat_tiers, wildchat_labels)
ax.tick_params(axis='x', labelsize=7.5)

# Factual drop annotation (GPT-3.5 -> GPT-4 at 6-10 turns)
p35, _ = rate_ci(wild['tier_stats'], 'gpt-3.5', '6-10')
p4, _ = rate_ci(wild['tier_stats'], 'gpt-4', '6-10')
ax.annotate('', xy=(1, p4), xytext=(0, p35),
            arrowprops=dict(arrowstyle='->', color=DARK_BLUE, lw=1.5))
ax.text(0.5, (p35 + p4) / 2 + 1.5, f'−{p35 - p4:.1f} pp', fontsize=8,
        color=DARK_BLUE, ha='center', fontweight='bold', rotation=-50)

ax.set_ylabel('Communication failure rate (%)', fontsize=10)
ax.set_xlabel('Model capability tier', fontsize=9)
ax.set_title('a', loc='left', fontsize=14, fontweight='bold', x=-0.08)
ax.text(0.5, 1.0, f'WildChat (N = {wildchat_n:,} multi-turn conversations)',
        transform=ax.transAxes, ha='center', va='bottom', fontsize=9,
        fontstyle='italic', color='gray')

# ── Panel (b): LMSYS ──────────────────────────────────────────────────────
ax = axes[1]
panel_lines(ax, lmsys['tier_stats'], lmsys_tiers, lmsys_labels)
ax.tick_params(axis='x', labelsize=6.5)
ax.set_xlabel('Model capability tier', fontsize=9)
ax.set_title('b', loc='left', fontsize=14, fontweight='bold', x=-0.08)
ax.text(0.5, 1.0, f'LMSYS-Chat-1M (N = {lmsys_n:,} multi-turn conversations)',
        transform=ax.transAxes, ha='center', va='bottom', fontsize=9,
        fontstyle='italic', color='gray')
ax.legend(loc='upper right', framealpha=0.9, edgecolor='#DDD',
          borderpad=0.5, labelspacing=0.3)

# ── Panel (c): Cross-dataset comparison ────────────────────────────────────
ax = axes[2]

categories = ['6-10\nturns', '11+\nturns']
wc_pts = [rate_ci(wild['tier_stats'], 'gpt-4', s) for s in ('6-10', '11+')]
lm_pts = [rate_ci(lmsys['tier_stats'], 'frontier', s) for s in ('6-10', '11+')]
wc_ns = [wild['tier_stats']['gpt-4'][s]['total'] for s in ('6-10', '11+')]
lm_ns = [lmsys['tier_stats']['frontier'][s]['total'] for s in ('6-10', '11+')]
x_conv = np.arange(len(categories))
width = 0.3

bars_wc = ax.bar(x_conv - width / 2, [p for p, _ in wc_pts], width,
                 yerr=[e for _, e in wc_pts], capsize=3,
                 color=WILDCAT_COLOR, label=f'WildChat GPT-4 (n = {wc_ns[0]:,}; {wc_ns[1]:,})',
                 edgecolor='white', linewidth=0.5, zorder=3,
                 error_kw=dict(elinewidth=0.9, capthick=0.9))
bars_lm = ax.bar(x_conv + width / 2, [p for p, _ in lm_pts], width,
                 yerr=[e for _, e in lm_pts], capsize=3,
                 color=LMSYS_COLOR, label=f'LMSYS frontier (n = {lm_ns[0]:,}; {lm_ns[1]:,})',
                 edgecolor='white', linewidth=0.5, zorder=3,
                 error_kw=dict(elinewidth=0.9, capthick=0.9))

for bar, (val, err), color in [(bars_wc[0], wc_pts[0], WILDCAT_COLOR),
                               (bars_wc[1], wc_pts[1], WILDCAT_COLOR),
                               (bars_lm[0], lm_pts[0], LMSYS_COLOR),
                               (bars_lm[1], lm_pts[1], LMSYS_COLOR)]:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 0.8,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=7.5,
            fontweight='bold', color=color)

# Factual difference annotation for 6-10 turns
diff_610 = abs(wc_pts[0][0] - lm_pts[0][0])
ax.annotate(f'Δ = {diff_610:.1f} pp', xy=(x_conv[0], max(wc_pts[0][0], lm_pts[0][0]) + 6),
            fontsize=8, ha='center', color=DARK_BLUE, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9E6',
                      edgecolor='#E9C46A', linewidth=0.8))

ax.set_xticks(x_conv)
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylabel('Failure rate (%)', fontsize=10)
ax.set_xlabel('Conversation length', fontsize=9)
ax.set_ylim(0, 38)
ax.set_title('c', loc='left', fontsize=14, fontweight='bold', x=-0.12)
ax.text(0.5, 1.0, 'Cross-dataset comparison', transform=ax.transAxes,
        ha='center', va='bottom', fontsize=9, fontstyle='italic', color='gray')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper left', framealpha=0.9, edgecolor='#DDD',
          fontsize=6.5, borderpad=0.5)
ax.grid(axis='y', alpha=0.2, linewidth=0.5)

# ── Save ────────────────────────────────────────────────────────────────────
output_dir = Path(__file__).parent
fig.savefig(output_dir / 'figure1_ceiling.png', dpi=300, facecolor='white')
fig.savefig(output_dir / 'figure1_ceiling.pdf', facecolor='white')
print(f"Figure 1 saved to {output_dir / 'figure1_ceiling.png'}")
