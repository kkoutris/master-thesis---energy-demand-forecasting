"""
Experiment 16 — H3 SHAP rank-consistency robustness (4 review points).

Reuses the OOS-surrogate machinery from experiments/h2_surrogate_robustness.py
(build_splits, stage1_resid, oos_surrogate, explainer_for, XGB_BASELINE).
H3 is recomputed on the OUT-OF-SAMPLE surrogate (headline) with the CV-tuned
surrogate as a robustness arm, over two feature universes:
  ALL  — all 22 surrogate features (calendar + Fourier + continuous)
  CONT — the 14 continuous climate/economic predictors only

  P1  stricter threshold 0.85 + Kendall's tau-b (robust to rank ties) alongside
      Spearman rho, per pair.
  P2  Holm correction across the three pairwise p-values (rho and tau).
  P3  block-bootstrap 95% CI on rho and tau per pair (blocks 7 and 30), to
      propagate the per-feature mean|SHAP| sampling uncertainty.
  P4  Kendall's W coefficient of concordance across all strategies at once
      (3-way Filter/Ridge/Lasso and 4-way incl. OLS-All) with a permutation
      p-value — a principled omnibus replacing the arbitrary >=2/3 vote;
      plus the conservative min-pairwise-rho summary.

Outputs:
  - data/h3_rank_consistency_robustness.csv   per universe x surrogate x pair
  - data/h3_concordance.csv                   Kendall's W per universe x surrogate
"""
from __future__ import annotations

import json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau, rankdata
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import h2_surrogate_robustness as h2   # build_splits, stage1_resid, oos_surrogate, XGB_BASELINE, CONTINUOUS

SEED = 42
N_BOOT = 2000
N_PERM = 5000
ALPHA = 0.05
THRESH = 0.85
PAIRS = [("Filter", "Ridge"), ("Filter", "Lasso"), ("Ridge", "Lasso")]


def load_tuned_params():
    df = pd.read_csv(ROOT / "data" / "h2_robustness_xgb_tuning.csv")
    bp = df[df["strategy"] == "_best_params"].iloc[0]
    return dict(n_estimators=int(bp["n_estimators"]), max_depth=int(bp["max_depth"]),
                learning_rate=float(bp["learning_rate"]),
                min_child_weight=int(bp["min_child_weight"]),
                subsample=float(bp["subsample"]), colsample_bytree=float(bp["colsample_bytree"]),
                reg_lambda=float(bp["reg_lambda"]),
                random_state=SEED, verbosity=0, n_jobs=-1, tree_method="hist")


def kendalls_w(rank_matrix):
    """Kendall's W (tie-corrected) for an (n_items x m_raters) rank matrix."""
    n, m = rank_matrix.shape
    R = rank_matrix.sum(axis=1)
    S = float(np.sum((R - R.mean()) ** 2))
    # tie correction term per rater
    T = 0.0
    for j in range(m):
        _, counts = np.unique(rank_matrix[:, j], return_counts=True)
        T += np.sum(counts ** 3 - counts)
    denom = (m ** 2 * (n ** 3 - n) - m * T) / 12.0
    return S / denom if denom > 0 else np.nan


def w_perm_p(mean_abs_by_strategy, n_perm=N_PERM, rng=None):
    """Permutation p for Kendall's W: independently permute each strategy's
    ranking under H0 of no concordance."""
    rng = rng or np.random.default_rng(SEED)
    rank_mat = np.column_stack([rankdata(-v) for v in mean_abs_by_strategy])
    obs = kendalls_w(rank_mat)
    n, m = rank_mat.shape
    ge = 0
    for _ in range(n_perm):
        perm = np.column_stack([rng.permutation(rank_mat[:, j]) for j in range(m)])
        if kendalls_w(perm) >= obs:
            ge += 1
    return obs, (ge + 1) / (n_perm + 1)


def surrogate_shap(splits, params, rp, ep):
    """OOS SHAP (n_test x 22) per strategy on the full feature space."""
    X_tr, X_te = splits["X_tr"], splits["X_te"]
    y_tr, y_te = splits["y_tr"], splits["y_te"]
    specs = {
        "OLS-All": (X_tr, X_te, LinearRegression()),
        "Filter":  (X_tr[splits["filter_cols"]], X_te[splits["filter_cols"]], LinearRegression()),
        "Ridge":   (X_tr, X_te, Ridge(alpha=rp["lambda"])),
        "Lasso":   (X_tr[splits["lasso_cols"]], X_te[splits["lasso_cols"]],
                    ElasticNet(alpha=ep["alpha"], l1_ratio=1.0, max_iter=5000, random_state=SEED)),
    }
    sv = {}
    for name, (Xtr_s, Xte_s, model) in specs.items():
        rt, re = h2.stage1_resid(Xtr_s, Xte_s, y_tr, y_te, model)
        _, _, s, _, _ = h2.oos_surrogate(rt, re, X_tr, X_te, params)
        sv[name] = s
    return sv, list(X_te.columns)


def pair_stats(mean_abs):
    rows = []
    rho_ps, tau_ps = [], []
    for a, b in PAIRS:
        ra, rb = rankdata(-mean_abs[a]), rankdata(-mean_abs[b])
        rho, rp_ = spearmanr(ra, rb)
        tau, tp_ = kendalltau(mean_abs[a], mean_abs[b], variant="b")
        rows.append([f"{a} vs {b}", rho, rp_, tau, tp_])
        rho_ps.append(rp_); tau_ps.append(tp_)
    rho_holm = multipletests(rho_ps, alpha=ALPHA, method="holm")[1]
    tau_holm = multipletests(tau_ps, alpha=ALPHA, method="holm")[1]
    return rows, rho_holm, tau_holm


def block_ci(sv, cols, universe_idx, rng, block):
    """Block-bootstrap 95% CI on rho and tau per pair."""
    n = next(iter(sv.values())).shape[0]
    nb = int(np.ceil(n / block))
    acc = {f"{a} vs {b}": {"rho": [], "tau": []} for a, b in PAIRS}
    for _ in range(N_BOOT):
        starts = rng.integers(0, n, nb)
        rows = np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]
        ma = {k: np.abs(v[rows][:, universe_idx]).mean(0) for k, v in sv.items()}
        for a, b in PAIRS:
            acc[f"{a} vs {b}"]["rho"].append(spearmanr(rankdata(-ma[a]), rankdata(-ma[b]))[0])
            acc[f"{a} vs {b}"]["tau"].append(kendalltau(ma[a], ma[b], variant="b")[0])
    out = {}
    for p, d in acc.items():
        out[p] = dict(rho_lo=np.percentile(d["rho"], 2.5), rho_hi=np.percentile(d["rho"], 97.5),
                      tau_lo=np.percentile(d["tau"], 2.5), tau_hi=np.percentile(d["tau"], 97.5))
    return out


def main():
    print("=" * 70)
    print("EXPERIMENT 16 — H3 rank-consistency robustness")
    print("=" * 70)
    rng = np.random.default_rng(SEED)
    rp = json.load(open(ROOT / "data" / "strategy_ridge_params.json"))
    ep = json.load(open(ROOT / "data" / "strategy_elasticnet_params.json"))
    splits = h2.build_splits(include_fourier=True)
    tuned = load_tuned_params()

    universes = {"ALL22": list(splits["X_te"].columns), "CONT14": h2.CONTINUOUS}
    surrogates = {"oos_baseline": h2.XGB_BASELINE, "oos_tuned": tuned}

    sv_cache = {}
    for sname, params in surrogates.items():
        t0 = time.time()
        sv, cols = surrogate_shap(splits, params, rp, ep)
        sv_cache[sname] = (sv, cols)
        print(f"  surrogate {sname}: SHAP computed [{time.time()-t0:.1f}s]")

    rows_out, conc_rows = [], []
    for uname, ufeats in universes.items():
        for sname, (sv, cols) in sv_cache.items():
            uidx = [cols.index(f) for f in ufeats]
            mean_abs = {k: np.abs(v[:, uidx]).mean(0) for k, v in sv.items()}
            stats, rho_holm, tau_holm = pair_stats({k: mean_abs[k] for k in
                                                    ("OLS-All", "Filter", "Ridge", "Lasso")})
            # block CIs (block 30 headline, block 7 secondary) only for baseline
            ci30 = block_ci(sv, cols, uidx, np.random.default_rng(SEED), 30) if sname == "oos_baseline" else None
            ci7 = block_ci(sv, cols, uidx, np.random.default_rng(SEED), 7) if sname == "oos_baseline" else None
            for k, (pair, rho, rp_, tau, tp_) in enumerate(stats):
                row = dict(universe=uname, surrogate=sname, pair=pair,
                           n_features=len(ufeats),
                           spearman_rho=round(rho, 4), rho_p=float(rp_),
                           rho_holm_p=round(float(rho_holm[k]), 4),
                           kendall_tau=round(tau, 4), tau_p=float(tp_),
                           tau_holm_p=round(float(tau_holm[k]), 4),
                           meets_085_rho=bool(rho > THRESH), meets_085_tau=bool(tau > THRESH))
                if ci30:
                    row.update(rho_ci30_lo=round(ci30[pair]["rho_lo"], 3),
                               rho_ci30_hi=round(ci30[pair]["rho_hi"], 3),
                               tau_ci30_lo=round(ci30[pair]["tau_lo"], 3),
                               tau_ci30_hi=round(ci30[pair]["tau_hi"], 3),
                               rho_ci7_lo=round(ci7[pair]["rho_lo"], 3),
                               rho_ci7_hi=round(ci7[pair]["rho_hi"], 3))
                rows_out.append(row)
            # Kendall's W concordance (3-way and 4-way)
            for label, members in [("3way_FRL", ["Filter", "Ridge", "Lasso"]),
                                   ("4way_incl_OLS", ["OLS-All", "Filter", "Ridge", "Lasso"])]:
                W, pW = w_perm_p([mean_abs[m] for m in members], rng=np.random.default_rng(SEED))
                conc_rows.append(dict(universe=uname, surrogate=sname, set=label,
                                      n_strategies=len(members), kendall_W=round(W, 4),
                                      perm_p=round(pW, 4)))
            min_rho = min(s[1] for s in stats)
            print(f"\n  [{uname} | {sname}]  min pairwise rho={min_rho:.3f}  "
                  f"(threshold {THRESH})")
            for pair, rho, rp_, tau, tp_ in stats:
                print(f"    {pair:<16} rho={rho:.3f} tau={tau:.3f}  "
                      f"{'PASS' if rho>THRESH else 'fail'}@0.85")

    rob = pd.DataFrame(rows_out)
    conc = pd.DataFrame(conc_rows)
    rob.to_csv(ROOT / "data" / "h3_rank_consistency_robustness.csv", index=False)
    conc.to_csv(ROOT / "data" / "h3_concordance.csv", index=False)

    # sanity vs h2_robustness_h3.csv (ALL22 baseline rho)
    try:
        prev = pd.read_csv(ROOT / "data" / "h2_robustness_h3.csv")
        prev = prev[prev.protocol == "baseline"].set_index("pair")["spearman_rho"]
        cur = rob[(rob.universe == "ALL22") & (rob.surrogate == "oos_baseline")].set_index("pair")["spearman_rho"]
        worst = max(abs(cur[p] - prev[p]) for p in cur.index if p in prev.index)
        print(f"\n[sanity] ALL22 baseline rho vs h2_robustness_h3.csv: max|Δ|={worst:.3f} "
              f"({'OK' if worst < 0.005 else 'CHECK'})")
    except Exception as e:
        print(f"\n[sanity] skipped ({e})")

    print("\n=== Block-bootstrap CIs (ALL22, oos_baseline, block=30) ===")
    sub = rob[(rob.universe == "ALL22") & (rob.surrogate == "oos_baseline")]
    print(sub[["pair", "spearman_rho", "rho_ci30_lo", "rho_ci30_hi",
               "kendall_tau", "tau_ci30_lo", "tau_ci30_hi"]].to_string(index=False))
    print("\n=== Kendall's W concordance ===")
    print(conc.to_string(index=False))
    print("\nSaved:")
    print("  data/h3_rank_consistency_robustness.csv")
    print("  data/h3_concordance.csv")


if __name__ == "__main__":
    main()
