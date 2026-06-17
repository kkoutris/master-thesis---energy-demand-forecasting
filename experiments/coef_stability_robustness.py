"""
Experiment 13 — Coefficient-stability composite-test robustness (5 review
points on the H1 / "four-dimension" composite in 04_interpretability.ipynb).

Reuses the notebook-04 setup exactly: strategy matrices strategy_{filtered,
ridge,elasticnet}_train.csv (OLS-All shares the ridge 18-feature matrix),
hyperparameters from strategy_*_params.json, CV = std/|mean| over folds
(clipped at 5), sign consistency = fraction of folds matching the mean sign
(defined only when mean != 0).

Parts:
  P4 CV-scheme robustness — recompute the per-feature CV and the
     Filter-vs-Ridge / Ridge-vs-Lasso comparisons under three fold schemes:
       expanding  : TimeSeriesSplit(10)  [current notebook scheme — sanity]
       fixed_roll : 10 equal-length rolling windows (fixed train size)
       blocked    : 10 disjoint contiguous blocks (equal size, equal variance)
     Reports median CV per strategy + Wilcoxon p + bootstrap CI per scheme;
     answers whether unequal expanding windows drive or mask any conclusion.
  P3 CV sampling noise — leave-one-fold-out jackknife of each per-feature CV
     (expanding scheme) -> per-feature CV SE; and how often the sign of the
     Filter-Ridge per-feature CV gap is jackknife-stable.
  P2 Dimension non-orthogonality — Spearman correlation across features
     between the three per-feature dimensions (CV, sign consistency,
     abs-contrib std) per strategy; quantifies how much they are independent
     evidence vs the same fold-coefficient information re-expressed.
  P1 Composite scorecard — per strategy pair, the winner on each dimension
     (overall + 3 feature groups); shows whether the joint conclusion is
     unanimous or a papered-over win-some/lose-some.
  P5 Lasso feature loss — features dropped (mean beta == 0) from each pairwise
     comparison, and an alternative treatment counting all-fold-zeroed
     features as a perfectly stable selection decision (CV=0, sign cons=1);
     re-runs Filter-vs-Lasso / Ridge-vs-Lasso under it.

Outputs:
  - data/h1_cv_scheme_robustness.csv
  - data/h1_dimension_correlation.csv
  - data/h1_composite_scorecard.csv
  - data/h1_lasso_feature_loss.csv
"""
from __future__ import annotations

import json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
N_SPLITS = 10
SEED = 42
N_BOOT = 2000

FEAT_GROUPS = {
    "Behavioral": ["day_of_week", "is_holiday", "is_weekend", "month"],
    "Climate":    ["cloudiness", "global_rad", "humidity_pct", "nao",
                   "precip_mm", "pressure_hpa", "temp_c"],
    "Economic":   ["gdp_mln_eur", "population", "price_eur_kwh",
                   "solar_energy_gwh", "wind_energy_gwh", "wind_ms"],
}


def load(strategy):
    return pd.read_csv(ROOT / "data" / f"strategy_{strategy}_train.csv",
                       parse_dates=["date"]).drop(columns=["date"])


def fold_indices(n, scheme):
    """Return a list of training-index arrays (one per fold) for each scheme."""
    if scheme == "expanding":
        # exactly sklearn TimeSeriesSplit(n_splits=10) as used in notebook 04
        tscv = TimeSeriesSplit(n_splits=N_SPLITS)
        return [tr for tr, _ in tscv.split(np.arange(n))]
    if scheme == "blocked":
        bounds = np.linspace(0, n, N_SPLITS + 1).astype(int)
        return [np.arange(bounds[i], bounds[i + 1]) for i in range(N_SPLITS)]
    if scheme == "fixed_roll":
        # 10 equal windows of length W, last window ends at n
        W = n // 2                       # 2140 days — moderate, equal size
        starts = np.linspace(0, n - W, N_SPLITS).astype(int)
        return [np.arange(s, s + W) for s in starts]
    raise ValueError(scheme)


def model_factory(name, rp, ep):
    if name == "Ridge":
        return lambda: Ridge(alpha=rp["lambda"])
    if name == "Lasso":
        return lambda: ElasticNet(l1_ratio=ep["l1_ratio"], alpha=ep["alpha"],
                                  max_iter=5000)
    return lambda: LinearRegression()


def fit_fold_coefs(X, y, cols, mfact, scheme):
    n = len(y)
    out = []
    for tr in fold_indices(n, scheme):
        m = mfact().fit(X[tr], y[tr])
        out.append(dict(zip(cols, m.coef_)))
    return out


def stability(coef_list, feats, zero_as_stable=False):
    df = pd.DataFrame([{f: d.get(f, np.nan) for f in feats} for d in coef_list])
    mean_c, std_c = df.mean(), df.std()
    cv = (std_c / mean_c.abs()).clip(upper=5.0)
    sign = {}
    for f in feats:
        col = df[f].dropna()
        if len(col) == 0:
            continue
        if mean_c[f] != 0:
            sign[f] = float((np.sign(col) == np.sign(mean_c[f])).mean())
        elif zero_as_stable and (col == 0).all():
            cv[f] = 0.0          # selected-out in every fold = stable decision
            sign[f] = 1.0
    return cv, pd.Series(sign), std_c


def boot_ci_median_diff(a, b, rng):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    d = [np.median(a[idx] - b[idx])
         for idx in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    return float(np.median(a - b)), tuple(np.percentile(d, [2.5, 97.5]))


def main():
    print("=" * 70)
    print("EXPERIMENT 13 — Coefficient-stability composite robustness")
    print("=" * 70)
    rng = np.random.default_rng(SEED)

    with open(ROOT / "data" / "strategy_ridge_params.json") as f:
        rp = json.load(f)
    with open(ROOT / "data" / "strategy_elasticnet_params.json") as f:
        ep = json.load(f)

    y = pd.read_csv(ROOT / "data" / "train.csv")["demand_MW"].values
    mats = {"OLS-All": load("ridge"), "Filter": load("filtered"),
            "Ridge": load("ridge"), "Lasso": load("elasticnet")}
    cols = {k: list(v.columns) for k, v in mats.items()}
    Xv = {k: v.values for k, v in mats.items()}
    all_feats = sorted(set().union(*cols.values()))

    # ---------------- P4: CV-scheme robustness ----------------------------
    print("\n[P4] CV-scheme robustness (median CV + Filter-vs-Ridge / "
          "Ridge-vs-Lasso tests)")
    scheme_rows = []
    cv_by_scheme = {}
    for scheme in ("expanding", "blocked", "fixed_roll"):
        cvs, signs, stds, coefs = {}, {}, {}, {}
        for name in mats:
            cf = fit_fold_coefs(Xv[name], y, cols[name],
                                model_factory(name, rp, ep), scheme)
            coefs[name] = cf
            cvs[name], signs[name], stds[name] = stability(cf, all_feats)
        cv_by_scheme[scheme] = (cvs, signs, stds, coefs)

        def pair_test(a, b):
            m = cvs[a].notna() & cvs[b].notna()
            ca, cb = cvs[a][m].values, cvs[b][m].values
            if np.allclose(ca - cb, 0):
                return len(ca), np.nan, np.nan, (np.nan, np.nan)
            _, p = wilcoxon(ca, cb)
            obs, ci = boot_ci_median_diff(ca, cb, rng)
            return int(m.sum()), p, obs, ci

        for a, b in [("Filter", "Ridge"), ("Ridge", "Lasso"),
                     ("OLS-All", "Ridge")]:
            n_f, p, obs, ci = pair_test(a, b)
            scheme_rows.append({
                "scheme": scheme, "comparison": f"{a} vs {b}",
                "n_features": n_f, "wilcoxon_p": round(p, 4),
                "median_cv_diff": round(obs, 4),
                "boot_ci_lo": round(ci[0], 4), "boot_ci_hi": round(ci[1], 4),
                "ci_excludes_0": bool(not (ci[0] <= 0 <= ci[1]))
                                 if np.isfinite(ci[0]) else None,
            })
        med = {name: float(cvs[name].median()) for name in mats}
        print(f"  {scheme:<11} median CV — "
              + "  ".join(f"{k}:{v:.3f}" for k, v in med.items()))
    pd.DataFrame(scheme_rows).to_csv(
        ROOT / "data" / "h1_cv_scheme_robustness.csv", index=False)
    for r in scheme_rows:
        if r["comparison"] == "Filter vs Ridge":
            print(f"    Filter vs Ridge [{r['scheme']}]: p={r['wilcoxon_p']}, "
                  f"CI=[{r['boot_ci_lo']},{r['boot_ci_hi']}] "
                  f"excl0={r['ci_excludes_0']}")

    # Use the expanding scheme (notebook) for P1-P3, P5
    cvs, signs, stds, coefs = cv_by_scheme["expanding"]

    # Sanity: expanding per-group median CV must match notebook stratified CSV
    try:
        strat = pd.read_csv(ROOT / "data" / "interpretability_h1_stratified.csv"
                            ).set_index("group")
        worst = 0.0
        for g, feats in FEAT_GROUPS.items():
            v = [f for f in feats if f in cvs["Ridge"].index and cvs["Ridge"].notna()[f]]
            worst = max(worst, abs(float(cvs["Ridge"][v].median())
                                   - strat.loc[g, "ridge_median_cv"]))
        print(f"  [sanity] expanding Ridge per-group median CV vs notebook: "
              f"max |Δ| = {worst:.4f} ({'OK' if worst < 0.01 else 'CHECK'})")
    except Exception as e:
        print(f"  [sanity] skipped ({e})")

    # ---------------- P3: per-feature CV jackknife noise ------------------
    print("\n[P3] Per-feature CV jackknife (leave-one-fold-out) — "
          "Filter vs Ridge")
    def jackknife_cv(coef_list, feats):
        K = len(coef_list)
        per_feat = {f: [] for f in feats}
        for drop in range(K):
            sub = [c for j, c in enumerate(coef_list) if j != drop]
            cv_d, _, _ = stability(sub, feats)
            for f in feats:
                per_feat[f].append(cv_d.get(f, np.nan))
        se = {f: float(np.nanstd(v, ddof=1)) for f, v in per_feat.items()}
        return pd.Series(se)
    se_f = jackknife_cv(coefs["Filter"], all_feats)
    se_r = jackknife_cv(coefs["Ridge"], all_feats)
    m_fr = cvs["Filter"].notna() & cvs["Ridge"].notna()
    gap = (cvs["Filter"] - cvs["Ridge"])[m_fr]
    gap_se = np.sqrt(se_f[m_fr] ** 2 + se_r[m_fr] ** 2)
    robust = (gap.abs() > gap_se).sum()
    median_cv_se_f = float(np.nanmedian(se_f[m_fr]))
    print(f"  median per-feature CV jackknife SE — Filter:{median_cv_se_f:.3f} "
          f"Ridge:{float(np.nanmedian(se_r[m_fr])):.3f}")
    print(f"  per-feature Filter-Ridge CV gap exceeds its own jackknife SE in "
          f"{robust}/{int(m_fr.sum())} features "
          f"(median |gap|={float(gap.abs().median()):.3f}, "
          f"median gap SE={float(np.nanmedian(gap_se)):.3f})")

    # ---------------- P2: dimension non-orthogonality ---------------------
    print("\n[P2] Cross-feature Spearman between dimensions (per strategy)")
    abs_contrib = {name: stds[name] * mats["Ridge"].std().reindex(all_feats)
                   for name in mats}
    corr_rows = []
    for name in mats:
        common = (cvs[name].notna() & signs[name].reindex(all_feats).notna())
        cv_v = cvs[name][common]
        sg_v = signs[name].reindex(all_feats)[common]
        ac_v = abs_contrib[name][common]
        def sp(x, yv):
            if len(x) < 4 or np.std(x) == 0 or np.std(yv) == 0:
                return np.nan
            return float(spearmanr(x, yv).correlation)
        rho_cv_sign = sp(cv_v, sg_v)
        rho_cv_ac = sp(cv_v, ac_v)
        corr_rows.append({"strategy": name, "n_features": int(common.sum()),
                          "spearman_cv_vs_signcons": round(rho_cv_sign, 3),
                          "spearman_cv_vs_abscontrib": round(rho_cv_ac, 3)})
        print(f"  {name:<8} rho(CV, signcons)={rho_cv_sign:+.3f}  "
              f"rho(CV, abscontrib)={rho_cv_ac:+.3f}  (N={int(common.sum())})")
    pd.DataFrame(corr_rows).to_csv(
        ROOT / "data" / "h1_dimension_correlation.csv", index=False)

    # ---------------- P1: composite scorecard -----------------------------
    print("\n[P1] Composite scorecard — winner per dimension")
    score_rows = []
    for a, b in [("Filter", "Ridge"), ("Ridge", "Lasso"), ("OLS-All", "Ridge")]:
        m = cvs[a].notna() & cvs[b].notna()
        cv_win = b if cvs[b][m].median() < cvs[a][m].median() else a  # lower CV
        sm = signs[a].reindex(all_feats).notna() & signs[b].reindex(all_feats).notna()
        sc_win = (b if signs[b].reindex(all_feats)[sm].mean()
                  > signs[a].reindex(all_feats)[sm].mean() else a)     # higher sign-cons
        ac_win = b if abs_contrib[b][m].median() < abs_contrib[a][m].median() else a
        grp = {}
        for g, feats in FEAT_GROUPS.items():
            v = [f for f in feats if f in cvs[a].index and
                 cvs[a].notna()[f] and cvs[b].notna()[f]]
            if v:
                grp[g] = b if cvs[b][v].median() < cvs[a][v].median() else a
        winners = [cv_win, sc_win, ac_win] + list(grp.values())
        unanimous = len(set(winners)) == 1
        score_rows.append({"comparison": f"{a} vs {b}",
                           "cv_winner": cv_win, "signcons_winner": sc_win,
                           "abscontrib_winner": ac_win,
                           **{f"grp_{g}": w for g, w in grp.items()},
                           "unanimous": unanimous})
        print(f"  {a} vs {b}: CV->{cv_win}  sign->{sc_win}  absC->{ac_win}  "
              f"groups->{grp}  {'UNANIMOUS' if unanimous else 'MIXED'}")
    pd.DataFrame(score_rows).to_csv(
        ROOT / "data" / "h1_composite_scorecard.csv", index=False)

    # ---------------- P5: Lasso feature loss ------------------------------
    print("\n[P5] Lasso feature loss & alternative (zeroed = stable) treatment")
    # all-fold-zeroed features under Lasso
    lasso_df = pd.DataFrame([{f: d.get(f, np.nan) for f in all_feats}
                             for d in coefs["Lasso"]])
    zeroed_all = [f for f in cols["Lasso"] if (lasso_df[f] == 0).all()]
    dropped_signcons = [f for f in cols["Lasso"]
                        if lasso_df[f].mean() == 0]
    print(f"  Lasso features zeroed in ALL folds: {zeroed_all}")
    print(f"  Lasso features dropped from sign-consistency (mean==0): "
          f"{dropped_signcons}")

    loss_rows = []
    for a, b in [("Filter", "Lasso"), ("Ridge", "Lasso")]:
        n_default = int((cvs[a].notna() & cvs[b].notna()).sum())
        # alternative: zeroed-everywhere counted as CV=0, sign=1
        cv_alt, sign_alt, _ = stability(coefs["Lasso"], all_feats,
                                        zero_as_stable=True)
        m_alt = cvs[a].notna() & cv_alt.notna()
        ca, cb = cvs[a][m_alt].values, cv_alt[m_alt].values
        _, p_alt = wilcoxon(ca, cb) if not np.allclose(ca - cb, 0) else (np.nan, np.nan)
        obs_alt, ci_alt = boot_ci_median_diff(ca, cb, rng)
        loss_rows.append({
            "comparison": f"{a} vs Lasso",
            "n_features_default": n_default,
            "n_features_zero_as_stable": int(m_alt.sum()),
            "wilcoxon_p_zero_as_stable": round(p_alt, 4),
            "median_cv_diff_zero_as_stable": round(obs_alt, 4),
            "ci_lo": round(ci_alt[0], 4), "ci_hi": round(ci_alt[1], 4),
        })
        print(f"  {a} vs Lasso: N {n_default} -> {int(m_alt.sum())} "
              f"(zero-as-stable), Wilcoxon p={p_alt:.4f}, "
              f"median CV diff={obs_alt:+.4f}")
    pd.DataFrame(loss_rows).to_csv(
        ROOT / "data" / "h1_lasso_feature_loss.csv", index=False)

    print("\nSaved:")
    for f in ["h1_cv_scheme_robustness", "h1_dimension_correlation",
              "h1_composite_scorecard", "h1_lasso_feature_loss"]:
        print(f"  data/{f}.csv")


if __name__ == "__main__":
    main()
