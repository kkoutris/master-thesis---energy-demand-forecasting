"""
Experiment 5 — Statistical-test depth (Note #10).

Post-processes existing CSV outputs to add the statistical-depth layer the
thesis currently lacks. Three additions, none of which re-run forecasts or
fits:

  (a) H1 effect sizes — rank-biserial correlation and paired Cohen's d
      alongside the existing Wilcoxon W/p in notebook 04 cell 8.

  (b) H1 prospective power curves — Monte Carlo simulation (10 000 reps)
      of Wilcoxon power across δ ∈ {0.05 … 0.50} CV units. Reports the
      minimum detectable effect (MDE) at 80 % power per pair, so the
      H1 null can be defended as "underpowered to detect effect ≥ X"
      rather than just "not significant."

  (c) DM tests with Holm-Bonferroni + ΔMAPE — augments
      data/sarimax_dm_tests.csv (42 tests) with multiple-testing
      correction (within-pair across horizons, and jointly across all 42)
      and joins MAPE-difference (pp) as the practical effect size.

NOTE (revision): the DM and power-curve outputs of this script are SUPERSEDED
and are no longer read by any notebook or cited in the draft:
  - The DM augmentation used the earlier, anti-conservative variance estimator;
    it is replaced by experiments/dm_correction.py (Newey-West + HLN), whose
    data/sarimax_dm_corrected.csv is what the notebooks and draft use.
  - The MDE/power curve is replaced by experiments/h1_power_mde_refinement.py
    (data/h1_mde_refined.csv, plots/h1_power_curve_refined.png).
This script is retained for the effect-size provenance (h1_effect_sizes.csv);
re-running it regenerates the superseded files, which should be discarded.

Outputs:
  - data/h1_effect_sizes.csv       Wilcoxon + r_rb + d + Holm per H1 pair   [current]
  - data/h1_power_curve.csv        Power × δ per pair (long form)           [superseded]
  - data/sarimax_dm_augmented.csv  DM + Holm-within / Holm-joint + ΔMAPE pp [superseded]
  - plots/h1_power_curve.png       Power curves, 80 % threshold line        [superseded]
  - plots/dm_holm_grid.png         Sig/not-sig grid (raw vs Holm)           [superseded]
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N_SIMS = 10_000
ALPHA = 0.05
EFFECT_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
HORIZONS = [1, 3, 7, 30, 90, 180]

# H1 pairs and the CV columns each uses
H1_PAIRS = [
    ("OLS-All", "Ridge",  "ols_all_cv", "ridge_cv"),
    ("OLS-All", "Filter", "ols_all_cv", "filter_cv"),
    ("Filter",  "Ridge",  "filter_cv",  "ridge_cv"),
    ("Filter",  "Lasso",  "filter_cv",  "lasso_cv"),
    ("Ridge",   "Lasso",  "ridge_cv",   "lasso_cv"),
]


def cohens_d_paired(diff: np.ndarray) -> float:
    """Cohen's d for paired differences: mean(diff) / sd(diff)."""
    sd = diff.std(ddof=1)
    if sd == 0:
        return np.nan
    return float(diff.mean() / sd)


def rank_biserial(W: float, n: int) -> float:
    """Wilcoxon rank-biserial correlation: r_rb = 1 − 2W / (n(n+1)/2).

    With SciPy convention W = min(W+, W−), this is the absolute magnitude
    of the rank-biserial correlation. Sign is restored from the paired
    median-difference direction by the caller.
    """
    T = n * (n + 1) / 2
    return float(1.0 - 2.0 * W / T)


def magnitude_label(d: float) -> str:
    a = abs(d)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# (a) H1 effect sizes
# ---------------------------------------------------------------------------
def compute_h1_effect_sizes(cv_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Returns the effect-size table and a dict {label: paired_diffs} for (b)."""
    rows = []
    paired_diffs = {}
    p_raw = []

    for a_name, b_name, col_a, col_b in H1_PAIRS:
        sub = cv_df[[col_a, col_b]].dropna()
        a = sub[col_a].values
        b = sub[col_b].values
        diff = a - b
        n = len(diff)

        W, p = wilcoxon(a, b, alternative="two-sided")
        r_rb = rank_biserial(W, n)
        # Direction: match the sign of median(diff) so r_rb reads as
        # "positive r_rb means A has the larger CV (less stable)."
        r_rb_signed = r_rb * (1.0 if np.median(diff) >= 0 else -1.0)
        d = cohens_d_paired(diff)

        label = f"{a_name} vs {b_name}"
        rows.append({
            "comparison": label,
            "n": n,
            "W": float(W),
            "p_raw": float(p),
            "median_diff": float(np.median(diff)),
            "mean_diff": float(diff.mean()),
            "sd_diff": float(diff.std(ddof=1)),
            "cohens_d": d,
            "r_rank_biserial": r_rb_signed,
            "effect_magnitude_label": magnitude_label(d),
        })
        paired_diffs[label] = diff
        p_raw.append(p)

    # Holm-Bonferroni across the 5 pairs.
    _, p_holm, _, _ = multipletests(p_raw, alpha=ALPHA, method="holm")
    out = pd.DataFrame(rows)
    out["p_holm_across_pairs"] = p_holm
    out["significant_holm_5pct"] = out["p_holm_across_pairs"] < ALPHA
    return out, paired_diffs


# ---------------------------------------------------------------------------
# (b) H1 prospective power curves
# ---------------------------------------------------------------------------
def simulate_power(obs_diffs: np.ndarray, delta: float,
                    n_sims: int = N_SIMS, alpha: float = ALPHA,
                    seed: int = SEED) -> float:
    """Bootstrap power at true effect δ.

    Center the observed paired differences at zero (to use as noise), then
    resample with replacement and shift by δ. Wilcoxon two-sided at α.
    """
    rng = np.random.default_rng(seed)
    n = len(obs_diffs)
    centered = obs_diffs - np.median(obs_diffs)
    rej = 0
    valid = 0
    for _ in range(n_sims):
        sample = rng.choice(centered, size=n, replace=True) + delta
        # All-zero samples raise; non-zero pairs are kept by default.
        if np.all(sample == 0):
            continue
        try:
            _, p = wilcoxon(sample, alternative="two-sided")
        except ValueError:
            continue
        valid += 1
        if p < alpha:
            rej += 1
    return rej / valid if valid > 0 else np.nan


def mde_at_power(effect_grid: list, powers: list, target: float = 0.80) -> float:
    """Linear-interpolate the δ that hits `target` power. NaN if never reached."""
    powers = np.asarray(powers)
    effect_grid = np.asarray(effect_grid)
    above = powers >= target
    if not above.any():
        return np.nan
    if powers[0] >= target:
        return float(effect_grid[0])
    # First index where power crosses the target
    idx = np.argmax(above)
    p0, p1 = powers[idx - 1], powers[idx]
    e0, e1 = effect_grid[idx - 1], effect_grid[idx]
    if p1 == p0:
        return float(e1)
    return float(e0 + (target - p0) * (e1 - e0) / (p1 - p0))


def compute_h1_power_curves(paired_diffs: dict) -> tuple[pd.DataFrame, dict]:
    rows = []
    mde = {}
    for label, diff in paired_diffs.items():
        powers = []
        for delta in EFFECT_GRID:
            pwr = simulate_power(diff, delta)
            powers.append(pwr)
            rows.append({"pair": label, "effect_size": delta,
                          "power": pwr, "n_simulations": N_SIMS,
                          "n_features": len(diff)})
        mde[label] = mde_at_power(EFFECT_GRID, powers, target=0.80)
    return pd.DataFrame(rows), mde


# ---------------------------------------------------------------------------
# (c) DM tests with Holm + ΔMAPE
# ---------------------------------------------------------------------------
def parse_strategy(model_label: str) -> str:
    """Extract the strategy name from 'SARIMAX-N (Strategy ...)' labels."""
    import re
    m = re.search(r"\((.*?)\)$", model_label)
    inner = m.group(1) if m else model_label
    # 'Lasso (l1=1.0)' → 'Lasso'
    return re.sub(r"\s*\(.*\)$", "", inner)


def normalise_strategy(s: str) -> str:
    return "Lasso" if s.startswith("Lasso") else s


def augment_dm(dm_csv: Path, results_csv: Path) -> pd.DataFrame:
    dm = pd.read_csv(dm_csv)
    res = pd.read_csv(results_csv)
    res["strategy"] = res["model"].apply(parse_strategy).apply(normalise_strategy)
    mape_map = (res.set_index(["strategy", "horizon_days"])["mape_pct"].to_dict())

    dm["strategy_A"] = dm["strategy_A"].apply(normalise_strategy)
    dm["strategy_B"] = dm["strategy_B"].apply(normalise_strategy)
    dm["pair"] = dm["strategy_A"] + " vs " + dm["strategy_B"]

    dm["mape_A"] = dm.apply(
        lambda r: mape_map.get((r["strategy_A"], int(r["horizon_days"])), np.nan), axis=1)
    dm["mape_B"] = dm.apply(
        lambda r: mape_map.get((r["strategy_B"], int(r["horizon_days"])), np.nan), axis=1)
    # Practical effect size: signed pp difference. Positive → B better than A.
    dm["delta_mape_pp"] = dm["mape_A"] - dm["mape_B"]

    # Holm within each strategy pair across the 6 horizons
    p_within = np.full(len(dm), np.nan)
    for pair, sub in dm.groupby("pair"):
        idx = sub.index.values
        _, p_adj, _, _ = multipletests(sub["p_two_sided"].values,
                                        alpha=ALPHA, method="holm")
        p_within[idx] = p_adj
    dm["p_holm_within_pair"] = p_within
    dm["significant_holm_within_pair"] = dm["p_holm_within_pair"] < ALPHA

    # Holm jointly across all 42 tests
    _, p_joint, _, _ = multipletests(dm["p_two_sided"].values,
                                      alpha=ALPHA, method="holm")
    dm["p_holm_joint"] = p_joint
    dm["significant_holm_joint"] = dm["p_holm_joint"] < ALPHA

    return dm[[
        "pair", "strategy_A", "strategy_B", "horizon_days",
        "dm_stat", "p_two_sided", "p_holm_within_pair", "p_holm_joint",
        "significant_5pct", "significant_holm_within_pair", "significant_holm_joint",
        "mape_A", "mape_B", "delta_mape_pp",
    ]]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_power_curves(power_df: pd.DataFrame, mde: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {
        "OLS-All vs Ridge":  "#888888",
        "OLS-All vs Filter": "#4477aa",
        "Filter vs Ridge":   "#117733",
        "Filter vs Lasso":   "#cc6677",
        "Ridge vs Lasso":    "#aa3377",
    }
    for pair, sub in power_df.groupby("pair"):
        sub = sub.sort_values("effect_size")
        m = mde.get(pair, np.nan)
        m_str = "n/a" if np.isnan(m) else f"{m:.2f}"
        ax.plot(sub["effect_size"], sub["power"], marker="o",
                color=colors.get(pair, "black"),
                label=f"{pair}  (MDE₈₀ = {m_str})")
    ax.axhline(0.80, color="red", lw=1, ls="--", alpha=0.6,
               label="80 % power")
    ax.axhline(ALPHA, color="grey", lw=0.5, ls=":", alpha=0.7,
               label="α = 0.05")
    ax.set_xlabel("True effect size δ (CV units, paired)")
    ax.set_ylabel("Power (P[reject H₀ | δ])")
    ax.set_title(
        "H1 prospective power curves — Wilcoxon signed-rank, 10 000 sims\n"
        "Bootstrap from observed paired CV differences, shifted to δ")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close()


def plot_dm_grid(dm: pd.DataFrame, out_path: Path):
    """Categorical heatmap of significance survival:
       0 = not sig under any test
       1 = sig raw only
       2 = sig raw + within-pair Holm
       3 = sig under all three (raw + within + joint)
    """
    pivot_rows = sorted(dm["pair"].unique())
    grid = np.zeros((len(pivot_rows), len(HORIZONS)), dtype=int)
    for i, pair in enumerate(pivot_rows):
        for j, h in enumerate(HORIZONS):
            sub = dm[(dm["pair"] == pair) & (dm["horizon_days"] == h)]
            if sub.empty:
                grid[i, j] = -1
                continue
            r = sub.iloc[0]
            if r["significant_holm_joint"]:
                grid[i, j] = 3
            elif r["significant_holm_within_pair"]:
                grid[i, j] = 2
            elif r["significant_5pct"]:
                grid[i, j] = 1
            else:
                grid[i, j] = 0

    cmap = ListedColormap(["#e8e8e8", "#fdd49e", "#fc8d59", "#b30000"])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=3)
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels([f"h={h}" for h in HORIZONS])
    ax.set_yticks(range(len(pivot_rows)))
    ax.set_yticklabels(pivot_rows, fontsize=9)
    # Annotate cells with the dm_stat
    for i, pair in enumerate(pivot_rows):
        for j, h in enumerate(HORIZONS):
            sub = dm[(dm["pair"] == pair) & (dm["horizon_days"] == h)]
            if sub.empty:
                continue
            stat = sub.iloc[0]["dm_stat"]
            ax.text(j, i, f"{stat:+.1f}", ha="center", va="center",
                    fontsize=7, color="black")
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color="#e8e8e8", label="0: n.s."),
        plt.Rectangle((0, 0), 1, 1, color="#fdd49e", label="1: raw only"),
        plt.Rectangle((0, 0), 1, 1, color="#fc8d59", label="2: + Holm-within"),
        plt.Rectangle((0, 0), 1, 1, color="#b30000", label="3: + Holm-joint"),
    ]
    ax.legend(handles=legend_patches, loc="lower left",
              bbox_to_anchor=(1.02, 0.0), fontsize=8)
    ax.set_title("DM significance grid — survival under multiple-testing correction\n"
                  "(cell values are dm_stat; positive = B better than A)")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("EXPERIMENT 5 — Statistical-test depth")
    print("=" * 70)

    cv_df = pd.read_csv(ROOT / "data" / "interpretability_coef_stability.csv",
                        index_col="feature")

    # (a) H1 effect sizes
    print("\n--- (a) H1 effect sizes (Wilcoxon + r_rb + Cohen's d + Holm) ---")
    eff_df, paired_diffs = compute_h1_effect_sizes(cv_df)
    eff_df.to_csv(ROOT / "data" / "h1_effect_sizes.csv", index=False)
    print(eff_df.round(4).to_string(index=False))

    # Sanity-check against the existing h1_tests.csv
    existing = pd.read_csv(ROOT / "data" / "interpretability_h1_tests.csv")
    existing = existing[existing["comparison"].isin(eff_df["comparison"])]
    merged = eff_df.merge(existing[["comparison", "wilcoxon_stat", "p_value"]],
                          on="comparison")
    merged["W_diff"] = (merged["W"] - merged["wilcoxon_stat"]).abs()
    merged["p_diff"] = (merged["p_raw"] - merged["p_value"]).abs()
    print("\nSanity check vs interpretability_h1_tests.csv:")
    print(merged[["comparison", "W", "wilcoxon_stat", "W_diff",
                   "p_raw", "p_value", "p_diff"]].round(4).to_string(index=False))
    if (merged["W_diff"] > 1e-6).any() or (merged["p_diff"] > 1e-4).any():
        print("  ⚠ Mismatch — investigate.")
    else:
        print("  ✓ W and p match existing CSV to within numerical tolerance.")

    # (b) H1 power curves
    print(f"\n--- (b) H1 prospective power curves ({N_SIMS} sims × "
          f"{len(EFFECT_GRID)} δ × {len(H1_PAIRS)} pairs) ---")
    t0 = time.time()
    power_df, mde = compute_h1_power_curves(paired_diffs)
    power_df.to_csv(ROOT / "data" / "h1_power_curve.csv", index=False)
    print(f"  Done in {time.time() - t0:.1f}s")
    pivot = power_df.pivot(index="pair", columns="effect_size",
                            values="power").round(3)
    pivot.columns = [f"δ={c}" for c in pivot.columns]
    pivot["MDE@80%"] = [mde[p] for p in pivot.index]
    print(pivot.to_string())

    # (c) DM tests augmented
    print("\n--- (c) DM tests + Holm + ΔMAPE ---")
    dm_aug = augment_dm(ROOT / "data" / "sarimax_dm_tests.csv",
                         ROOT / "data" / "sarimax_results.csv")
    dm_aug.to_csv(ROOT / "data" / "sarimax_dm_augmented.csv", index=False)

    print("\nSurvival of significance under correction:")
    print(f"  Raw α=0.05 only        : {dm_aug['significant_5pct'].sum():>2} of 42")
    print(f"  + Holm within-pair (×6): {dm_aug['significant_holm_within_pair'].sum():>2} of 42")
    print(f"  + Holm joint (×42)     : {dm_aug['significant_holm_joint'].sum():>2} of 42")

    print("\nDM tests significant at any level (raw, Holm-within, or Holm-joint):")
    survivors = dm_aug[dm_aug["significant_5pct"]
                       | dm_aug["significant_holm_within_pair"]
                       | dm_aug["significant_holm_joint"]]
    print(survivors[["pair", "horizon_days", "dm_stat",
                      "p_two_sided", "p_holm_within_pair", "p_holm_joint",
                      "delta_mape_pp"]].round(4).to_string(index=False))

    # Plots
    print("\nGenerating plots...")
    plot_power_curves(power_df, mde, ROOT / "plots" / "h1_power_curve.png")
    print("  Saved: plots/h1_power_curve.png")
    plot_dm_grid(dm_aug, ROOT / "plots" / "dm_holm_grid.png")
    print("  Saved: plots/dm_holm_grid.png")

    print("\nSaved data:")
    print("  data/h1_effect_sizes.csv")
    print("  data/h1_power_curve.csv")
    print("  data/sarimax_dm_augmented.csv")


if __name__ == "__main__":
    main()
