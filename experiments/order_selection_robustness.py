"""
Experiment 7 — SARIMA order-selection robustness.

Addresses five review points on the Stage-2 order protocol:
  1. Universal order vs per-strategy orders
  2. Seasonal component (0,1,1)_7 fixed a priori, never searched
  3. Small non-seasonal grid (6 candidates) with a boundary winner (2,0,1)
  4. AIC-only selection (BIC computed but not used)
  5. Order selected on Strategy-1 residuals only (similar-structure assumption)

Protocol (train residuals only — no test data touched in selection):
  A. Expanded stepwise search on Strategy-1 (Filter) residuals:
     Step 1 — seasonal fixed at (0,1,1)_7; non-seasonal p∈{0..3} × q∈{0..2},
              d=0 (residuals strongly stationary), plus the two original d=1
              candidates for continuity with the notebook grid.
     Step 2 — best non-seasonal fixed; 7 seasonal (P,D,Q)_7 alternatives.
     Every fit records AIC, BIC, Ljung-Box p at lags 7/14/21/28/365.
  B. Per-strategy selection: the Step-1 grid is rerun on each of the five
     strategies' Stage-1 train residuals (with the universal seasonal order
     and, if different, the Step-2 winner). AIC/BIC winners are reported
     per strategy. AIC is NOT compared across strategies (different endog).
  C. If any strategy's winner differs from the universal (2,0,1)x(0,1,1)_7,
     rolling-origin test MAPE (oracle exog, eval_step=15, notebook-03
     protocol) is computed under both orders to quantify the impact.

Outputs:
  - data/order_grid_expanded.csv     all Part-A fits (stage = nonseasonal/seasonal)
  - data/order_per_strategy.csv      Part-B winners per strategy
  - data/order_sensitivity_mape.csv  Part-C MAPE deltas (always written;
                                     zero rows if all winners match)
"""
from __future__ import annotations

import json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
HORIZONS = [1, 3, 7, 30, 90, 180]
MAX_H = max(HORIZONS)
EVAL_STEP = 15  # notebook-03 rolling-origin protocol

UNIVERSAL_ORDER = (2, 0, 1)
UNIVERSAL_SEASONAL = (0, 1, 1, 7)

LB_LAGS = [7, 14, 21, 28, 365]

# Step-1 non-seasonal grid: d=0 over p<=3, q<=2 (minus the degenerate
# (0,0,0)) + the two d=1 candidates from the original notebook grid.
NONSEASONAL_GRID = (
    [(p, 0, q) for p in range(4) for q in range(3) if (p, q) != (0, 0)]
    + [(1, 1, 1), (0, 1, 1)]
)

# Step-2 seasonal grid (s=7). (0,1,1) is the incumbent.
SEASONAL_GRID = [
    (0, 1, 1, 7), (1, 1, 0, 7), (1, 1, 1, 7), (0, 1, 2, 7),
    (1, 0, 1, 7), (1, 0, 0, 7), (0, 0, 1, 7),
]

# Original notebook AICs for the sanity check (cell 7 of 03_sarimax_models)
NOTEBOOK_AICS = {
    (1, 0, 0): 61109.4, (0, 0, 1): 62392.6, (1, 0, 1): 60898.1,
    (2, 0, 1): 60885.8, (1, 1, 1): 60998.0, (0, 1, 1): 61134.6,
}


def load_strategy(name):
    return {s: pd.read_csv(ROOT / "data" / f"strategy_{name}_{s}.csv",
                           parse_dates=["date"]).drop(columns=["date"]).values
            for s in ("train", "test")}


def build_strategies():
    """Stage-1 train residuals + fitted models per strategy, exactly as in
    notebook 03 (same strategy CSVs, same hyperparameters)."""
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

    strategies = {}
    for name, (X, model) in specs.items():
        stage1 = model.fit(X["train"], y_tr)
        resid_tr = y_tr - stage1.predict(X["train"])
        strategies[name] = {
            "stage1": stage1, "resid_tr": resid_tr,
            "X_test": X["test"], "y_test": y_te,
        }
        print(f"  {name:<8} resid_std={resid_tr.std():.1f} MW")
    return strategies


def fit_candidate(resid, order, seasonal_order):
    """Fit one SARIMA candidate; return metrics row (AIC, BIC, Ljung-Box)."""
    t0 = time.time()
    try:
        m = SARIMAX(resid, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
        r = m.fit(disp=False)
        lb = acorr_ljungbox(r.resid, lags=LB_LAGS, return_df=True)
        row = {
            "order": str(order), "seasonal_order": str(seasonal_order),
            "aic": r.aic, "bic": r.bic,
            "converged": bool(r.mle_retvals.get("converged", True)),
            "fit_s": round(time.time() - t0, 1),
        }
        for lag in LB_LAGS:
            row[f"lb_p{lag}"] = float(lb.loc[lag, "lb_pvalue"])
        return row, r
    except Exception as e:
        return {"order": str(order), "seasonal_order": str(seasonal_order),
                "aic": np.nan, "bic": np.nan, "converged": False,
                "fit_s": round(time.time() - t0, 1), "error": str(e)[:80]}, None


def grid_search(resid, nonseasonal_grid, seasonal_orders, label=""):
    rows = []
    for seas in seasonal_orders:
        for order in nonseasonal_grid:
            row, _ = fit_candidate(resid, order, seas)
            rows.append(row)
            aic = row["aic"]
            aic_s = f"{aic:.1f}" if np.isfinite(aic) else "FAILED"
            print(f"    {label}SARIMA{order}x{seas}  AIC={aic_s}  "
                  f"BIC={row['bic']:.1f}  LB7p={row.get('lb_p7', np.nan):.3f}  "
                  f"[{row['fit_s']}s]" if np.isfinite(aic) else
                  f"    {label}SARIMA{order}x{seas}  FAILED")
    return pd.DataFrame(rows)


def rolling_eval(stage1, resid_tr, order, seasonal_order, X_te, y_te):
    """Notebook-03 oracle rolling-origin MAPE (Kalman append, no refit)."""
    sarima_r = SARIMAX(resid_tr, order=order, seasonal_order=seasonal_order,
                       enforce_stationarity=False,
                       enforce_invertibility=False).fit(disp=False)
    exog_te = stage1.predict(X_te)
    resid_te = y_te - exog_te
    results = {h: {"pred": [], "actual": []} for h in HORIZONS}
    for origin in range(0, len(resid_te) - MAX_H, EVAL_STEP):
        res_ext = sarima_r if origin == 0 else sarima_r.append(
            endog=resid_te[:origin], refit=False)
        fc = res_ext.forecast(steps=MAX_H)
        for h in HORIZONS:
            t = origin + h
            if t >= len(resid_te):
                continue
            results[h]["pred"].append(fc[h - 1] + exog_te[t])
            results[h]["actual"].append(y_te[t])
    return {h: 100 * np.mean(np.abs((np.array(r["actual"]) - np.array(r["pred"]))
                                    / np.array(r["actual"])))
            for h, r in results.items() if r["actual"]}


def main():
    print("=" * 70)
    print("EXPERIMENT 7 — SARIMA order-selection robustness")
    print("=" * 70)

    print("\nBuilding Stage-1 residuals (notebook-03 strategy matrices)...")
    strategies = build_strategies()
    resid_filter = strategies["Filter"]["resid_tr"]

    # ---------------- Part A: expanded stepwise search (Filter) -----------
    print(f"\n[A] Step 1 — non-seasonal grid ({len(NONSEASONAL_GRID)} candidates), "
          f"seasonal fixed {UNIVERSAL_SEASONAL}")
    step1 = grid_search(resid_filter, NONSEASONAL_GRID, [UNIVERSAL_SEASONAL])
    step1["stage"] = "nonseasonal"

    ok1 = step1.dropna(subset=["aic"]).sort_values("aic")
    best_ns = eval(ok1.iloc[0]["order"])
    print(f"\n  Step-1 winner: {best_ns}  "
          f"(AIC {ok1.iloc[0]['aic']:.1f}, BIC winner: "
          f"{ok1.sort_values('bic').iloc[0]['order']})")

    # Sanity check against the original notebook grid
    print("\n  Sanity check vs notebook cell-7 AICs:")
    for order, ref in NOTEBOOK_AICS.items():
        got = step1.loc[step1["order"] == str(order), "aic"]
        if len(got) and np.isfinite(got.iloc[0]):
            flag = "OK" if abs(got.iloc[0] - ref) < 1.0 else "MISMATCH"
            print(f"    {order}: script={got.iloc[0]:.1f}  notebook={ref}  {flag}")

    print(f"\n[A] Step 2 — seasonal grid ({len(SEASONAL_GRID)} candidates), "
          f"non-seasonal fixed {best_ns}")
    step2 = grid_search(resid_filter, [best_ns], SEASONAL_GRID)
    step2["stage"] = "seasonal"

    ok2 = step2.dropna(subset=["aic"]).sort_values("aic")
    best_seas = eval(ok2.iloc[0]["seasonal_order"])
    print(f"\n  Step-2 winner: {best_seas}  (AIC {ok2.iloc[0]['aic']:.1f})")

    grid_df = pd.concat([step1, step2], ignore_index=True)
    grid_df["strategy"] = "Filter"
    grid_df.to_csv(ROOT / "data" / "order_grid_expanded.csv", index=False)

    # ---------------- Part B: per-strategy selection ----------------------
    seasonal_set = [UNIVERSAL_SEASONAL]
    if best_seas != UNIVERSAL_SEASONAL:
        seasonal_set.append(best_seas)

    print(f"\n[B] Per-strategy non-seasonal grid "
          f"(seasonal orders tested: {seasonal_set})")
    per_rows, per_grids = [], {}
    for name, s in strategies.items():
        print(f"\n  --- {name} ---")
        g = grid_search(s["resid_tr"], NONSEASONAL_GRID, seasonal_set,
                        label=f"{name[:3]} ")
        g["strategy"] = name
        per_grids[name] = g
        ok = g.dropna(subset=["aic"])
        win_aic = ok.sort_values("aic").iloc[0]
        win_bic = ok.sort_values("bic").iloc[0]
        match = (eval(win_aic["order"]) == UNIVERSAL_ORDER
                 and eval(win_aic["seasonal_order"]) == UNIVERSAL_SEASONAL)
        per_rows.append({
            "strategy": name,
            "best_order_aic": win_aic["order"],
            "best_seasonal_aic": win_aic["seasonal_order"],
            "aic": win_aic["aic"],
            "best_order_bic": win_bic["order"],
            "best_seasonal_bic": win_bic["seasonal_order"],
            "bic": win_bic["bic"],
            "lb_p7": win_aic["lb_p7"], "lb_p28": win_aic["lb_p28"],
            "lb_p365": win_aic["lb_p365"],
            "matches_universal": match,
        })
        print(f"  {name}: AIC winner {win_aic['order']}x{win_aic['seasonal_order']}"
              f"  BIC winner {win_bic['order']}x{win_bic['seasonal_order']}"
              f"  matches_universal={match}")

    per_df = pd.DataFrame(per_rows)
    per_df.to_csv(ROOT / "data" / "order_per_strategy.csv", index=False)
    pd.concat(per_grids.values(), ignore_index=True).to_csv(
        ROOT / "data" / "order_per_strategy_grid.csv", index=False)

    # ---------------- Part C: conditional accuracy sensitivity ------------
    deviating = per_df[~per_df["matches_universal"]]
    sens_rows = []
    if len(deviating):
        print(f"\n[C] {len(deviating)} strategies deviate from the universal "
              f"order — rolling-origin sensitivity (oracle, step={EVAL_STEP})")
        for _, row in deviating.iterrows():
            name = row["strategy"]
            s = strategies[name]
            own_order = eval(row["best_order_aic"])
            own_seas = eval(row["best_seasonal_aic"])
            print(f"  {name}: universal {UNIVERSAL_ORDER}x{UNIVERSAL_SEASONAL} "
                  f"vs own {own_order}x{own_seas}")
            m_uni = rolling_eval(s["stage1"], s["resid_tr"], UNIVERSAL_ORDER,
                                 UNIVERSAL_SEASONAL, s["X_test"], s["y_test"])
            m_own = rolling_eval(s["stage1"], s["resid_tr"], own_order,
                                 own_seas, s["X_test"], s["y_test"])
            for h in HORIZONS:
                sens_rows.append({
                    "strategy": name, "horizon": h,
                    "mape_universal": m_uni[h], "mape_own_order": m_own[h],
                    "delta": m_own[h] - m_uni[h],
                    "own_order": str(own_order), "own_seasonal": str(own_seas),
                })
                print(f"    h={h:<3} universal={m_uni[h]:.3f}%  "
                      f"own={m_own[h]:.3f}%  Δ={m_own[h]-m_uni[h]:+.3f} pp")
    else:
        print("\n[C] All strategies select the universal order — no "
              "sensitivity rerun needed (zero-delta table written).")

    pd.DataFrame(sens_rows, columns=[
        "strategy", "horizon", "mape_universal", "mape_own_order",
        "delta", "own_order", "own_seasonal",
    ]).to_csv(ROOT / "data" / "order_sensitivity_mape.csv", index=False)

    # ---------------- Headline summary ------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Universal order        : {UNIVERSAL_ORDER}x{UNIVERSAL_SEASONAL}")
    print(f"Expanded grid winner   : {best_ns}x{best_seas}  (Filter residuals)")
    print(f"Top 5 by AIC (step 1):")
    print(ok1[["order", "aic", "bic", "lb_p7", "lb_p28", "lb_p365"]]
          .head(5).to_string(index=False))
    print(f"\nSeasonal alternatives (step 2):")
    print(ok2[["seasonal_order", "aic", "bic", "lb_p7", "lb_p28", "lb_p365"]]
          .to_string(index=False))
    print(f"\nPer-strategy winners:")
    print(per_df.to_string(index=False))
    print("\nSaved:")
    for f in ["order_grid_expanded.csv", "order_per_strategy.csv",
              "order_per_strategy_grid.csv", "order_sensitivity_mape.csv"]:
        print(f"  data/{f}")


if __name__ == "__main__":
    main()
