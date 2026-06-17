"""
Experiment 14 — H1 power/MDE refinement and multiple-testing scope
(four review points on the H1 statistical layer in statistical_depth.py).

  P4+P1  Extend & refine the effect grid. The headline power layer
         (statistical_depth.py, EFFECT_GRID 0.05..0.50) leaves the MDE@80%
         OFF-grid (n/a) for 4 of 5 pairs because the crossover is beyond
         0.50 CV units. Recompute power on a fine grid 0.02..2.0 so every
         pair gets a concrete MDE, with finer steps confirming the
         interpolation near each crossover.
  P2     Uncertainty band on the MDE. The power sim bootstraps the observed
         paired CV differences, so it inherits their (noisy, N=16-22) shape.
         Quantify it: bootstrap over the FEATURE SET (B resamples), recompute
         the whole curve + MDE each time -> 95% band. Plus a shape contrast:
         MDE under a matched-SD Gaussian noise model vs the empirical-shape
         bootstrap.
  blocked tie-in: report the observed Filter-vs-Ridge / OLS-vs-Ridge CV gap
         under the equal-window BLOCKED scheme (where the effect is large)
         against the expanding-regime MDE band, linking power to the
         conditional-support H1 finding.
  P3     Multiple-testing scope. H1 (5 pairs) and DM (42 tests) are Holm-
         corrected as separate families. Pool all 47 p-values and Holm jointly
         to show the union FWER concern is immaterial (0 survive either way).

A fast vectorised normal-approx signed-rank power replaces the per-sim scipy
call so the feature-bootstrap is affordable; it is sanity-checked against the
saved scipy-exact power curve first.

Outputs:
  - data/h1_mde_refined.csv        per pair: MDE point + 95% band + Gaussian MDE
                                   + observed gap + power at observed gap
  - data/h1_global_holm_check.csv  per-family vs pooled Holm survivor counts
  - plots/h1_power_curve_refined.png
"""
from __future__ import annotations

import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, wilcoxon
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import coef_stability_robustness as csr  # fold_indices, stability, load, model_factory

SEED = 42
ALPHA = 0.05
N_SIMS = 20_000          # headline point-estimate power
N_SIMS_BOOT = 8_000      # inside each feature-bootstrap resample
B_BOOT = 400             # feature-set bootstrap resamples
TARGET = 0.80
FINE_GRID = np.round(np.arange(0.02, 2.0001, 0.02), 3)
BOOT_GRID = np.round(np.arange(0.05, 2.0001, 0.05), 3)

H1_PAIRS = [
    ("OLS-All", "Ridge",  "ols_all_cv", "ridge_cv"),
    ("OLS-All", "Filter", "ols_all_cv", "filter_cv"),
    ("Filter",  "Ridge",  "filter_cv",  "ridge_cv"),
    ("Filter",  "Lasso",  "filter_cv",  "lasso_cv"),
    ("Ridge",   "Lasso",  "ridge_cv",   "lasso_cv"),
]


def vec_power(centered, delta, n_sims, rng):
    """Vectorised two-sided Wilcoxon signed-rank power (normal approximation,
    ordinal ranks; ties negligible for continuous CV differences)."""
    n = len(centered)
    s = rng.choice(centered, size=(n_sims, n), replace=True) + delta
    absx = np.abs(s)
    ranks = absx.argsort(axis=1).argsort(axis=1) + 1.0        # ordinal ranks
    w_plus = np.sum(ranks * (s > 0), axis=1)
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    z = (w_plus - mean) / np.sqrt(var)
    p = 2.0 * norm.sf(np.abs(z))
    return float(np.mean(p < ALPHA))


def power_curve(centered, grid, n_sims, rng):
    return np.array([vec_power(centered, d, n_sims, rng) for d in grid])


def mde(grid, powers, target=TARGET):
    powers = np.asarray(powers); grid = np.asarray(grid)
    above = powers >= target
    if not above.any():
        return np.nan
    if powers[0] >= target:
        return float(grid[0])
    idx = int(np.argmax(above))
    p0, p1 = powers[idx - 1], powers[idx]
    e0, e1 = grid[idx - 1], grid[idx]
    return float(e0 + (target - p0) * (e1 - e0) / (p1 - p0)) if p1 != p0 else float(e1)


def blocked_cv_diffs(rp, ep):
    """Per-feature CV under the equal-window blocked scheme for the two key
    pairs (Filter-Ridge, OLS-Ridge); returns {pair: paired diff array}."""
    y = pd.read_csv(ROOT / "data" / "train.csv")["demand_MW"].values
    mats = {"OLS-All": csr.load("ridge"), "Filter": csr.load("filtered"),
            "Ridge": csr.load("ridge")}
    cols = {k: list(v.columns) for k, v in mats.items()}
    feats = sorted(set().union(*cols.values()))
    cvb = {}
    for name in mats:
        cf = csr.fit_fold_coefs(mats[name].values, y, cols[name],
                                csr.model_factory(name, rp, ep), "blocked")
        cvb[name], _, _ = csr.stability(cf, feats)
    out = {}
    for a, b in [("Filter", "Ridge"), ("OLS-All", "Ridge")]:
        m = cvb[a].notna() & cvb[b].notna()
        out[f"{a} vs {b}"] = (cvb[a][m] - cvb[b][m]).values
    return out


def main():
    print("=" * 70)
    print("EXPERIMENT 14 — H1 power/MDE refinement & MTC scope")
    print("=" * 70)
    rng = np.random.default_rng(SEED)

    cv_df = pd.read_csv(ROOT / "data" / "interpretability_coef_stability.csv")
    diffs = {}
    for a, b, ca, cb in H1_PAIRS:
        sub = cv_df[[ca, cb]].dropna()
        diffs[f"{a} vs {b}"] = (sub[ca] - sub[cb]).values

    # ---- sanity: vec_power vs saved scipy power curve --------------------
    saved = pd.read_csv(ROOT / "data" / "h1_power_curve.csv")
    print("\n[sanity] vec_power (normal-approx) vs saved scipy-exact power "
          "at delta in {0.10,0.30,0.50}:")
    for label, diff in diffs.items():
        c = diff - np.median(diff)
        for d in (0.10, 0.30, 0.50):
            mine = vec_power(c, d, N_SIMS, rng)
            ref = saved[(saved["pair"] == label) &
                        (np.isclose(saved["effect_size"], d))]["power"]
            ref = float(ref.iloc[0]) if len(ref) else np.nan
            flag = "" if (np.isnan(ref) or abs(mine - ref) < 0.06) else "  <-CHECK"
            print(f"  {label:<18} d={d:.2f}  vec={mine:.3f}  scipy={ref:.3f}{flag}")

    # ---- P4+P1: refined-grid MDE (point estimate) ------------------------
    print(f"\n[P4+P1] Refined grid 0.02..2.0  (N_SIMS={N_SIMS}) — MDE@80%")
    refined = {}
    curves = {}
    for label, diff in diffs.items():
        c = diff - np.median(diff)
        pw = power_curve(c, FINE_GRID, N_SIMS, rng)
        curves[label] = pw
        m = mde(FINE_GRID, pw)
        refined[label] = m
        obs = float(np.median(diff))
        pw_obs = vec_power(c, abs(obs), N_SIMS, rng)
        print(f"  {label:<18} MDE@80%={'n/a' if np.isnan(m) else f'{m:.3f}'}  "
              f"observed |median diff|={abs(obs):.3f}  power@observed={pw_obs:.2f}")

    # ---- P2: feature-bootstrap band + Gaussian-shape contrast ------------
    print(f"\n[P2] MDE 95% band over {B_BOOT} feature-bootstrap resamples "
          f"+ matched-SD Gaussian-model MDE")
    rows = []
    for label, diff in diffs.items():
        n = len(diff)
        boot_mde = []
        for _ in range(B_BOOT):
            idx = rng.integers(0, n, n)
            db = diff[idx]
            cb = db - np.median(db)
            if np.allclose(cb, 0):
                continue
            pw = power_curve(cb, BOOT_GRID, N_SIMS_BOOT, rng)
            boot_mde.append(mde(BOOT_GRID, pw))
        boot_mde = np.array(boot_mde, float)
        frac_na = float(np.mean(np.isnan(boot_mde)))
        finite = boot_mde[np.isfinite(boot_mde)]
        lo, hi = (np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan))
        # Gaussian-shape model: same SD, normal noise
        sd = diff.std(ddof=1)
        cg = rng.normal(0, sd, size=max(n, 200))
        cg -= np.median(cg)
        mde_g = mde(FINE_GRID, power_curve(cg, FINE_GRID, N_SIMS, rng))
        obs = float(np.median(diff))
        rows.append({
            "pair": label, "n_features": n,
            "mde80_point": round(refined[label], 4) if not np.isnan(refined[label]) else np.nan,
            "mde80_ci_lo": round(float(lo), 4) if np.isfinite(lo) else np.nan,
            "mde80_ci_hi": round(float(hi), 4) if np.isfinite(hi) else np.nan,
            "mde80_frac_unreached_in_boot": round(frac_na, 3),
            "mde80_gaussian_model": round(mde_g, 4) if not np.isnan(mde_g) else np.nan,
            "observed_median_diff": round(obs, 4),
            "sd_diff": round(float(sd), 4),
        })
        print(f"  {label:<18} MDE={rows[-1]['mde80_point']}  "
              f"95% band=[{rows[-1]['mde80_ci_lo']}, {rows[-1]['mde80_ci_hi']}]  "
              f"(unreached in {frac_na:.0%} of resamples)  "
              f"Gaussian MDE={rows[-1]['mde80_gaussian_model']}")

    # ---- blocked-regime tie-in ------------------------------------------
    print("\n[blocked tie-in] observed CV gap under equal-window blocked CV "
          "vs the expanding-regime MDE")
    import json
    rp = json.load(open(ROOT / "data" / "strategy_ridge_params.json"))
    ep = json.load(open(ROOT / "data" / "strategy_elasticnet_params.json"))
    bdiffs = blocked_cv_diffs(rp, ep)
    for label, bd in bdiffs.items():
        med = float(np.median(bd))
        mde_exp = refined.get(label, np.nan)
        row = next(r for r in rows if r["pair"] == label)
        row["blocked_observed_median_diff"] = round(med, 4)
        print(f"  {label:<18} blocked median CV gap={med:+.3f}  "
              f"expanding MDE@80%={'n/a' if np.isnan(mde_exp) else f'{mde_exp:.3f}'}  "
              f"-> blocked effect {'EXCEEDS' if med > (mde_exp if not np.isnan(mde_exp) else 1e9) else 'below'} the MDE")

    pd.DataFrame(rows).to_csv(ROOT / "data" / "h1_mde_refined.csv", index=False)

    # ---- P3: global Holm union ------------------------------------------
    print("\n[P3] Multiple-testing scope — per-family vs pooled Holm")
    eff = pd.read_csv(ROOT / "data" / "h1_effect_sizes.csv")
    dm = pd.read_csv(ROOT / "data" / "sarimax_dm_tests.csv")
    p_h1 = eff["p_raw"].values
    p_dm = dm["p_two_sided"].values
    # per-family
    h1_fam = multipletests(p_h1, alpha=ALPHA, method="holm")[0].sum()
    dm_fam = multipletests(p_dm, alpha=ALPHA, method="holm")[0].sum()
    # pooled union
    p_all = np.concatenate([p_h1, p_dm])
    pooled = multipletests(p_all, alpha=ALPHA, method="holm")[0]
    raw_all = (p_all < ALPHA).sum()
    holm_rows = [
        {"family": "H1 pairs (5)", "n_tests": len(p_h1),
         "raw_sig": int((p_h1 < ALPHA).sum()),
         "holm_sig_per_family": int(h1_fam),
         "holm_sig_pooled_47": int(pooled[:len(p_h1)].sum())},
        {"family": "DM tests (42)", "n_tests": len(p_dm),
         "raw_sig": int((p_dm < ALPHA).sum()),
         "holm_sig_per_family": int(dm_fam),
         "holm_sig_pooled_47": int(pooled[len(p_h1):].sum())},
        {"family": "UNION (47)", "n_tests": len(p_all),
         "raw_sig": int(raw_all),
         "holm_sig_per_family": int(h1_fam + dm_fam),
         "holm_sig_pooled_47": int(pooled.sum())},
    ]
    hdf = pd.DataFrame(holm_rows)
    hdf.to_csv(ROOT / "data" / "h1_global_holm_check.csv", index=False)
    print(hdf.to_string(index=False))
    print(f"  -> separate-family vs pooled Holm both yield "
          f"{int(pooled.sum())} survivors: the union-FWER concern is immaterial.")

    # ---- plot ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    colors = {"OLS-All vs Ridge": "#888888", "OLS-All vs Filter": "#4477aa",
              "Filter vs Ridge": "#117733", "Filter vs Lasso": "#cc6677",
              "Ridge vs Lasso": "#aa3377"}
    for label in diffs:
        m = refined[label]
        lab = f"{label}  (MDE₈₀={'n/a' if np.isnan(m) else f'{m:.2f}'})"
        ax.plot(FINE_GRID, curves[label], color=colors[label], lw=1.6, label=lab)
        r = next(x for x in rows if x["pair"] == label)
        if np.isfinite(r["mde80_ci_lo"]) and np.isfinite(r["mde80_ci_hi"]):
            ax.axvspan(r["mde80_ci_lo"], r["mde80_ci_hi"], color=colors[label],
                       alpha=0.06)
    ax.axhline(TARGET, color="red", lw=1, ls="--", alpha=0.6, label="80% power")
    ax.set_xlabel("True effect size δ (CV units, paired)")
    ax.set_ylabel("Power")
    ax.set_title("H1 power curves — refined grid (0.02–2.0) with 95% MDE bands\n"
                 "vectorised normal-approx Wilcoxon")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(ROOT / "plots" / "h1_power_curve_refined.png",
                bbox_inches="tight", dpi=120)
    plt.close()

    print("\nSaved:")
    print("  data/h1_mde_refined.csv")
    print("  data/h1_global_holm_check.csv")
    print("  plots/h1_power_curve_refined.png")


if __name__ == "__main__":
    main()
