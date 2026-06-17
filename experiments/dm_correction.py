"""
Experiment 9 — Corrected Diebold-Mariano variance estimator.

The notebook-03 DM implementation (`dm_test_hln`, cell 36) has two flaws:
  1. Long-run variance = var(d) only — no autocovariance terms. Forecast
     origins are spaced 15 days apart, so windows overlap (and loss
     differentials are serially correlated) whenever h > 15.
  2. The HLN small-sample factor is applied with h in CALENDAR DAYS although
     the d-series is sampled per ORIGIN. At h=30 the factor inflates the DM
     statistic ~2.3x (anti-conservative); at h=180 it deflates it ~3.2x.

Corrected protocol (origin units, m = ceil(h/15) = horizon in origin steps):
  - LRV via Newey-West (Bartlett kernel) with truncation L = m-1, floored at
    gamma0 if the kernel sum turns non-positive.
  - HLN factor computed with m, t-distribution with df = n-1.
  - h <= 15 (m=1) reduces to the classical no-overlap DM with no HLN
    inflation — the corrected test is *less* conservative than the notebook
    at h=180 and *more* conservative at h=7/30.

Per-origin absolute errors are regenerated with the exact notebook-03
protocol (same strategy CSVs, universal SARIMA(2,0,1)x(0,1,1,7), oracle
exog, eval_step=15); the notebook-style DM statistics are recomputed as a
sanity check against data/sarimax_dm_tests.csv before correcting.

Outputs:
  - data/sarimax_dm_corrected.csv   original vs corrected stat/p per pair x
                                    horizon + Holm (within-pair and joint)
  - plots/dm_holm_grid_corrected.png  significance-survival grid (corrected)
"""
from __future__ import annotations

import json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

from scipy import stats as scipy_stats
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 3, 7, 30, 90, 180]
MAX_H = max(HORIZONS)
EVAL_STEP = 15
UNIVERSAL_ORDER = (2, 0, 1)
UNIVERSAL_SEASONAL = (0, 1, 1, 7)
ALPHA = 0.05

PAIRS = [
    ("OLS-All", "Filter"),
    ("OLS-All", "Ridge"),
    ("Filter", "Ridge"),
    ("Filter", "Lasso"),
    ("Ridge", "Lasso"),
    ("Filter", "PCA"),
    ("PCA", "Ridge"),
]


def load_strategy(name):
    return {s: pd.read_csv(ROOT / "data" / f"strategy_{name}_{s}.csv",
                           parse_dates=["date"]).drop(columns=["date"]).values
            for s in ("train", "test")}


def build_strategies():
    train = pd.read_csv(ROOT / "data" / "train.csv", parse_dates=["date"])
    test = pd.read_csv(ROOT / "data" / "test.csv", parse_dates=["date"])
    y_tr = train["demand_MW"].values
    y_te = test["demand_MW"].values

    X_filtered = load_strategy("filtered")
    X_pca = load_strategy("pca")
    X_ridge = load_strategy("ridge")
    X_enet = load_strategy("elasticnet")

    with open(ROOT / "data" / "strategy_ridge_params.json") as f:
        rp = json.load(f)
    with open(ROOT / "data" / "strategy_elasticnet_params.json") as f:
        ep = json.load(f)

    specs = {
        "OLS-All": (X_ridge, LinearRegression()),
        "Filter":  (X_filtered, LinearRegression()),
        "PCA":     (X_pca, LinearRegression()),
        "Ridge":   (X_ridge, Ridge(alpha=rp["lambda"], fit_intercept=True)),
        "Lasso":   (X_enet, ElasticNet(alpha=ep["alpha"], l1_ratio=ep["l1_ratio"],
                                       max_iter=5000, fit_intercept=True)),
    }
    out = {}
    for name, (X, model) in specs.items():
        out[name] = {"stage1": model.fit(X["train"], y_tr),
                     "X": X, "y_tr": y_tr, "y_te": y_te}
    return out


def abs_errors_per_origin(s):
    """Replicates notebook-03 rolling_evaluate; returns {h: abs error array}."""
    stage1, X, y_tr, y_te = s["stage1"], s["X"], s["y_tr"], s["y_te"]
    resid_tr = y_tr - stage1.predict(X["train"])
    sarima_r = SARIMAX(resid_tr, order=UNIVERSAL_ORDER,
                       seasonal_order=UNIVERSAL_SEASONAL,
                       enforce_stationarity=False,
                       enforce_invertibility=False).fit(disp=False)
    exog_te = stage1.predict(X["test"])
    resid_te = y_te - exog_te

    errs = {h: [] for h in HORIZONS}
    for origin in range(0, len(resid_te) - MAX_H, EVAL_STEP):
        res_ext = sarima_r if origin == 0 else sarima_r.append(
            endog=resid_te[:origin], refit=False)
        fc = res_ext.forecast(steps=MAX_H)
        for h in HORIZONS:
            t = origin + h
            if t >= len(resid_te):
                continue
            errs[h].append(abs(y_te[t] - (fc[h - 1] + exog_te[t])))
    return {h: np.array(v) for h, v in errs.items()}


def dm_notebook(e1, e2, h):
    """Original notebook-03 dm_test_hln (calendar-day h, gamma0 only)."""
    d = e1 - e2
    n = len(d)
    d_bar = d.mean()
    gamma0 = np.var(d, ddof=1)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm = d_bar / (np.sqrt(gamma0 / n) * hln)
    p = 2 * (1 - scipy_stats.t.cdf(abs(dm), df=n - 1))
    return float(dm), float(p)


def dm_corrected(e1, e2, h, step=EVAL_STEP):
    """DM with Newey-West LRV over the origin-spaced d-series and the HLN
    factor in origin units m = ceil(h/step)."""
    d = e1 - e2
    n = len(d)
    d_bar = d.mean()
    m = int(np.ceil(h / step))          # horizon in origin steps
    L = m - 1                            # NW truncation (overlap depth)
    dc = d - d_bar
    gamma0 = float(dc @ dc) / n
    lrv = gamma0
    for k in range(1, L + 1):
        gamma_k = float(dc[k:] @ dc[:-k]) / n
        lrv += 2 * (1 - k / (L + 1)) * gamma_k
    if lrv <= 0:                         # kernel sum can go negative in
        lrv = gamma0                     # finite samples; floor at gamma0
    hln = np.sqrt(max((n + 1 - 2 * m + m * (m - 1) / n) / n, 1e-12))
    dm = d_bar / (np.sqrt(lrv / n) * hln)
    p = 2 * (1 - scipy_stats.t.cdf(abs(dm), df=n - 1))
    return float(dm), float(p), m, L


def main():
    print("=" * 70)
    print("EXPERIMENT 9 — Corrected DM variance estimator")
    print("=" * 70)

    print("\nRegenerating per-origin absolute errors (notebook-03 protocol)...")
    strategies = build_strategies()
    errors = {}
    for name, s in strategies.items():
        t0 = time.time()
        errors[name] = abs_errors_per_origin(s)
        n_evals = {h: len(v) for h, v in errors[name].items()}
        print(f"  {name:<8} n per horizon = {n_evals}  [{time.time()-t0:.1f}s]")

    # Sanity check vs the saved notebook DM stats
    ref = pd.read_csv(ROOT / "data" / "sarimax_dm_tests.csv")
    ref["strategy_A"] = ref["strategy_A"].str.replace(r"Lasso.*", "Lasso", regex=True)
    ref["strategy_B"] = ref["strategy_B"].str.replace(r"Lasso.*", "Lasso", regex=True)
    print("\nSanity check vs data/sarimax_dm_tests.csv (notebook-style recompute):")
    mismatches = 0
    for (a, b) in PAIRS:
        for h in HORIZONS:
            n = min(len(errors[a][h]), len(errors[b][h]))
            dm_nb, _ = dm_notebook(errors[a][h][:n], errors[b][h][:n], h)
            row = ref[(ref.strategy_A == a) & (ref.strategy_B == b)
                      & (ref.horizon_days == h)]
            if len(row):
                saved = row.iloc[0]["dm_stat"]
                if abs(dm_nb - saved) > 0.05:
                    mismatches += 1
                    print(f"  MISMATCH {a} vs {b} h={h}: recomputed {dm_nb:.3f} "
                          f"vs saved {saved:.3f}")
    print(f"  {'all 42 cells match (|Δ| ≤ 0.05)' if mismatches == 0 else f'{mismatches} mismatches'}")

    # Original vs corrected DM for all pairs × horizons
    rows = []
    for (a, b) in PAIRS:
        for h in HORIZONS:
            n = min(len(errors[a][h]), len(errors[b][h]))
            e1, e2 = errors[a][h][:n], errors[b][h][:n]
            dm_o, p_o = dm_notebook(e1, e2, h)
            dm_c, p_c, m, L = dm_corrected(e1, e2, h)
            rows.append({
                "pair": f"{a} vs {b}", "strategy_A": a, "strategy_B": b,
                "horizon_days": h, "n_origins": n,
                "m_origin_units": m, "nw_truncation": L,
                "dm_stat_original": round(dm_o, 4),
                "p_original": round(p_o, 4),
                "dm_stat_corrected": round(dm_c, 4),
                "p_corrected": round(p_c, 4),
                "sig_original": p_o < ALPHA,
                "sig_corrected": p_c < ALPHA,
            })
    dm = pd.DataFrame(rows)

    # Holm corrections on the corrected p-values (mirrors statistical_depth.py)
    p_within = np.full(len(dm), np.nan)
    for pair, sub in dm.groupby("pair"):
        idx = sub.index.values
        _, p_adj, _, _ = multipletests(sub["p_corrected"].values,
                                       alpha=ALPHA, method="holm")
        p_within[idx] = p_adj
    dm["p_holm_within_pair"] = np.round(p_within, 4)
    dm["significant_holm_within_pair"] = dm["p_holm_within_pair"] < ALPHA
    _, p_joint, _, _ = multipletests(dm["p_corrected"].values,
                                     alpha=ALPHA, method="holm")
    dm["p_holm_joint"] = np.round(p_joint, 4)
    dm["significant_holm_joint"] = dm["p_holm_joint"] < ALPHA

    dm.to_csv(ROOT / "data" / "sarimax_dm_corrected.csv", index=False)

    # Report: cells whose 5% verdict flips
    print("\n" + "=" * 70)
    print("Original vs corrected — full grid")
    print("=" * 70)
    cols = ["pair", "horizon_days", "dm_stat_original", "p_original",
            "dm_stat_corrected", "p_corrected", "p_holm_within_pair"]
    print(dm[cols].to_string(index=False))

    flips = dm[dm.sig_original != dm.sig_corrected]
    print("\nVerdict flips at raw 5% level:")
    if len(flips):
        print(flips[["pair", "horizon_days", "p_original", "p_corrected"]]
              .to_string(index=False))
    else:
        print("  none")

    # Corrected significance-survival grid (raw vs Holm-within vs Holm-joint)
    pivot_rows = sorted(dm["pair"].unique())
    grid = np.zeros((len(pivot_rows), len(HORIZONS)), dtype=int)
    for i, pair in enumerate(pivot_rows):
        for j, h in enumerate(HORIZONS):
            r = dm[(dm["pair"] == pair) & (dm["horizon_days"] == h)].iloc[0]
            if r["significant_holm_joint"]:
                grid[i, j] = 3
            elif r["significant_holm_within_pair"]:
                grid[i, j] = 2
            elif r["sig_corrected"]:
                grid[i, j] = 1
            else:
                grid[i, j] = 0

    cmap = ListedColormap(["#e8e8e8", "#fdd49e", "#fc8d59", "#b30000"])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=3)
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels([f"h={h}" for h in HORIZONS])
    ax.set_yticks(range(len(pivot_rows)))
    ax.set_yticklabels(pivot_rows, fontsize=8)
    for i in range(len(pivot_rows)):
        for j in range(len(HORIZONS)):
            lab = ["n.s.", "raw", "Holm-pair", "Holm-joint"][grid[i, j]]
            ax.text(j, i, lab, ha="center", va="center", fontsize=7,
                    color="white" if grid[i, j] == 3 else "black")
    ax.set_title("DM significance survival — corrected variance estimator\n"
                 "(Newey–West LRV in origin units + HLN(m), t df=n−1)",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(ROOT / "plots" / "dm_holm_grid_corrected.png",
                bbox_inches="tight", dpi=120)
    plt.close()

    print("\nSaved:")
    print("  data/sarimax_dm_corrected.csv")
    print("  plots/dm_holm_grid_corrected.png")


if __name__ == "__main__":
    main()
