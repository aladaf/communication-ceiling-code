"""
Equivalence testing (TOST) for the communication-ceiling claims.

Responds to Reviewer 1 (Frontiers in AI, Major Revision):
  - #5: "A nonsignificant GPT-4 versus GPT-4o result does not prove a plateau."
  - #6: "Use equivalence testing or segmented regression to support the ceiling claim."

Reads the aggregate counts already saved by scripts 03/04/05 (no raw-data re-run
needed) and applies two one-sided z-tests (TOST) for the difference between two
proportions, with an equivalence margin of +/- 2 percentage points at the 6-10
turn stratum. TOST at alpha = .05 is equivalent to the 90% CI of the difference
lying inside the margin, which is also reported.

Outputs: data/equivalence_results.json and a console table.
"""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

MARGIN = 0.02  # +/- 2 percentage points
ALPHA = 0.05
STRATUM = "6-10"


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def tost_two_proportions(x1, n1, x2, n2, margin=MARGIN):
    """TOST for H0: |p1 - p2| >= margin vs H1: |p1 - p2| < margin.

    Uses the unpooled z statistic (appropriate for equivalence testing).
    Returns dict with the two one-sided p-values, the TOST p (max of the two),
    and the 90% CI of the difference.
    """
    p1, p2 = x1 / n1, x2 / n2
    diff = p1 - p2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z_lower = (diff + margin) / se   # H0: diff <= -margin
    z_upper = (diff - margin) / se   # H0: diff >= +margin
    p_lower = norm_sf(z_lower)        # P(reject diff <= -margin)
    p_upper = norm_sf(-z_upper)
    p_tost = max(p_lower, p_upper)
    z90 = 1.6448536269514722
    ci90 = (diff - z90 * se, diff + z90 * se)
    return {
        "p1": p1, "p2": p2, "diff": diff, "se": se,
        "n1": n1, "n2": n2, "x1": x1, "x2": x2,
        "p_lower": p_lower, "p_upper": p_upper, "p_tost": p_tost,
        "ci90": ci90,
        "equivalent": p_tost < ALPHA,
        "margin": margin,
    }


def fmt(res, label):
    lo, hi = res["ci90"]
    verdict = "EQUIVALENT" if res["equivalent"] else "not shown equivalent"
    return (
        f"{label:55s} {res['p1']*100:6.2f}% vs {res['p2']*100:6.2f}%  "
        f"diff={res['diff']*100:+5.2f}pp  90% CI [{lo*100:+5.2f}, {hi*100:+5.2f}]  "
        f"p_TOST={res['p_tost']:.2e}  -> {verdict}"
    )


def main():
    wild = json.loads((DATA / "failure_by_model_results.json").read_text())
    lmsys = json.loads((DATA / "lmsys_failure_results.json").read_text())
    skill = json.loads((DATA / "user_skill_results.json").read_text())

    results = {"stratum": STRATUM, "margin_pp": MARGIN * 100, "alpha": ALPHA,
               "comparisons": {}}
    lines = []

    def cell(tree, tier):
        c = tree["tier_stats"][tier][STRATUM]
        return c["with_any_failure"], c["total"]

    # --- Study 1 (WildChat): post-threshold tiers vs GPT-4 -----------------
    lines.append("== Study 1 (WildChat, 6-10 turns): equivalence to GPT-4 ==")
    x4, n4 = cell(wild, "gpt-4")
    for tier in ["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini"]:
        if tier in wild["tier_stats"]:
            xt, nt = cell(wild, tier)
            res = tost_two_proportions(x4, n4, xt, nt)
            results["comparisons"][f"wildchat_gpt-4_vs_{tier}"] = res
            lines.append(fmt(res, f"GPT-4 vs {tier}"))

    # Contrast: the genuine pre-threshold gap should NOT be equivalent.
    x35, n35 = cell(wild, "gpt-3.5")
    res = tost_two_proportions(x35, n35, x4, n4)
    results["comparisons"]["wildchat_gpt-3.5_vs_gpt-4"] = res
    lines.append(fmt(res, "GPT-3.5 vs GPT-4 (negative control)"))

    # --- Study 2 (LMSYS): adjacent-tier comparisons ------------------------
    lines.append("")
    lines.append("== Study 2 (LMSYS, 6-10 turns): adjacent tiers ==")
    order = ["weak", "medium", "strong", "frontier"]
    for a, b in zip(order, order[1:]):
        xa, na = cell(lmsys, a)
        xb, nb = cell(lmsys, b)
        res = tost_two_proportions(xa, na, xb, nb)
        results["comparisons"][f"lmsys_{a}_vs_{b}"] = res
        lines.append(fmt(res, f"{a} vs {b}"))

    # Cross-dataset convergence at the ceiling (WildChat GPT-4 vs LMSYS frontier)
    xf, nf = cell(lmsys, "frontier")
    res = tost_two_proportions(x4, n4, xf, nf)
    results["comparisons"]["wildchat_gpt-4_vs_lmsys_frontier"] = res
    lines.append(fmt(res, "WildChat GPT-4 vs LMSYS frontier (cross-dataset)"))

    # --- Study 3 (user experience): novice vs power users ------------------
    lines.append("")
    lines.append("== Study 3 (WildChat, 6-10 turns): experience levels ==")
    exp = skill["exp_stats"]

    def ecell(level):
        c = exp[level][STRATUM]
        return c["with_failure"], c["total"]

    x1c, n1c = ecell("1")
    for level in ["2-3", "4-10", "11-30", "31-100", "100+"]:
        xl, nl = ecell(level)
        res = tost_two_proportions(x1c, n1c, xl, nl)
        results["comparisons"][f"experience_1_vs_{level}"] = res
        lines.append(fmt(res, f"1 conversation vs {level}"))

    out = DATA / "equivalence_results.json"
    out.write_text(json.dumps(results, indent=2))
    print("\n".join(lines))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
