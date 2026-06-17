"""
Experiment 11 — Stage-1 re-estimation sensitivity (Review point: "Stage 1 is
never re-estimated on incoming data; coefficient drift is absorbed into
errors rather than corrected for").

Two arms, oracle exog protocol (comparable with data/sarimax_results.csv),
identical rolling origins (every 15 days, n=50 per horizon):

  static — current headline protocol: Stage 1 + SARIMA fit once on TRAIN
           (2009-01..2020-09); SARIMA state advanced through test by Kalman
           append only. Must reproduce data/sarimax_results.csv (sanity).
  refit  — operator-update scenario: every 6th origin (~90 days) BOTH stages
           are refit on the expanding window train + val + test[:origin].
           Note this folds in the val period (2020-09..2023-03: COVID
           recovery + 2022 energy crisis) that the headline protocol never
           trains on — deliberately, since that is exactly the drift the
           static design absorbs into errors. Between refits the SARIMA state
           is advanced by appending the residuals implied by the CURRENT
           Stage-1 fit. Feature sets, scaler and hyperparameters stay fixed
           from the original train-only fit (isolates coefficient drift from
           selection/tuning drift).

Also reports Stage-1 coefficient drift directly: coefficients fit on train
vs train+val vs train+val+test for the interpretable strategies.

Outputs:
  - data/stage1_refit_sensitivity.csv  MAPE x arm x strategy x horizon + delta
  - data/stage1_coef_drift.csv         coefficients x fit window x strategy
"""
from __future__ import annotations

import json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
HORIZONS = [1, 3, 7, 30, 90, 180]
MAX_H = max(HORIZONS)
EVAL_STEP = 15
REFIT_EVERY = 6 * EVAL_STEP        # refit every 6th origin = 90 days
SEASONAL_ORDER = (0, 1, 1, 7)
BEST_ORDER = (2, 0, 1)
PCA_N_COMPONENTS = 9

CLIMATE = ["temp_c", "wind_ms", "precip_mm", "sunshine_h",
           "global_rad", "pressure_hpa", "humidity_pct", "cloudiness", "nao"]
STRUCTURAL = ["price_eur_kwh", "gdp_mln_eur", "population",
              "wind_energy_gwh", "solar_energy_gwh"]
CONTINUOUS = CLIMATE + STRUCTURAL
RAW_CAL = ["day_of_week", "month", "is_weekend", "is_holiday"]
FOURIER = ["sin1_ann", "cos1_ann", "sin2_ann", "cos2_ann"]
CALENDAR = RAW_CAL + FOURIER


def add_fourier(df):
    doy = df["date"].dt.dayofyear
    out = df.copy()
    out["sin1_ann"] = np.sin(2 * np.pi * doy / 365.25)
    out["cos1_ann"] = np.cos(2 * np.pi * doy / 365.25)
    out["sin2_ann"] = np.sin(4 * np.pi * doy / 365.25)
    out["cos2_ann"] = np.cos(4 * np.pi * doy / 365.25)
    return out


def correlation_filter(X_df, y, threshold=0.80):
    cols = list(X_df.columns)
    target_corr = {c: abs(np.corrcoef(X_df[c].values, y)[0, 1]) for c in cols}
    to_remove = set()
    cm = X_df.corr().abs()
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if j <= i or c1 in to_remove or c2 in to_remove:
                continue
            if cm.loc[c1, c2] > threshold:
                drop = c1 if target_corr[c1] < target_corr[c2] else c2
                to_remove.add(drop)
    return [c for c in cols if c not in to_remove]


def mape(y_true, y_pred):
    return 100 * np.mean(np.abs((y_true - y_pred) / y_true))


def load_strategy_splits(name):
    """Canonical notebook-02 strategy matrices (already train-scaled)."""
    out = {}
    for s in ("train", "val", "test"):
        df = pd.read_csv(ROOT / "data" / f"strategy_{name}_{s}.csv",
                         parse_dates=["date"]).drop(columns=["date"])
        out[s] = df.reset_index(drop=True)
    return out


def build_data():
    y = {}
    for s in ("train", "val", "test"):
        y[s] = pd.read_csv(ROOT / "data" / f"{s}.csv")["demand_MW"].values
    mats = {
        "Filter": load_strategy_splits("filtered"),
        "PCA":    load_strategy_splits("pca"),
        "Ridge":  load_strategy_splits("ridge"),
        "Lasso":  load_strategy_splits("elasticnet"),
    }
    mats["OLS-All"] = mats["Ridge"]      # same 18-feature matrices
    return mats, y


def eval_static(X_tr_s, X_te_s, y_tr, y_te, mfact):
    stage1 = mfact().fit(X_tr_s, y_tr)
    resid_tr = y_tr - stage1.predict(X_tr_s)
    sarima_r = SARIMAX(resid_tr, order=BEST_ORDER, seasonal_order=SEASONAL_ORDER,
                       enforce_stationarity=False,
                       enforce_invertibility=False).fit(disp=False)
    exog_te = stage1.predict(X_te_s)
    resid_te = y_te - exog_te
    res = {h: {"p": [], "a": []} for h in HORIZONS}
    for origin in range(0, len(y_te) - MAX_H, EVAL_STEP):
        ext = sarima_r if origin == 0 else sarima_r.append(
            endog=resid_te[:origin], refit=False)
        fc = ext.forecast(steps=MAX_H)
        for h in HORIZONS:
            t = origin + h
            if t >= len(y_te):
                continue
            res[h]["p"].append(fc[h - 1] + exog_te[t])
            res[h]["a"].append(y_te[t])
    return {h: mape(np.array(r["a"]), np.array(r["p"])) for h, r in res.items()}


def eval_refit(X_tr_s, X_va_s, X_te_s, y_tr, y_va, y_te, mfact):
    """Expanding-window refit of both stages every REFIT_EVERY days."""
    X_hist = pd.concat([X_tr_s, X_va_s, X_te_s], ignore_index=True)
    y_hist = np.concatenate([y_tr, y_va, y_te])
    n_pre = len(y_tr) + len(y_va)            # index of test[0] in history

    res = {h: {"p": [], "a": []} for h in HORIZONS}
    stage1 = sarima_cur = None
    last_refit = None
    n_refits = 0
    for origin in range(0, len(y_te) - MAX_H, EVAL_STEP):
        if stage1 is None or origin % REFIT_EVERY == 0:
            hist_end = n_pre + origin
            Xw = X_hist.iloc[:hist_end]
            yw = y_hist[:hist_end]
            stage1 = mfact().fit(Xw, yw)
            resid_w = yw - stage1.predict(Xw)
            sarima_cur = SARIMAX(resid_w, order=BEST_ORDER,
                                 seasonal_order=SEASONAL_ORDER,
                                 enforce_stationarity=False,
                                 enforce_invertibility=False).fit(disp=False)
            last_refit = origin
            n_refits += 1
            exog_te = stage1.predict(X_te_s)
            resid_te = y_te - exog_te
        if origin == last_refit:
            ext = sarima_cur
        else:
            ext = sarima_cur.append(endog=resid_te[last_refit:origin],
                                    refit=False)
        fc = ext.forecast(steps=MAX_H)
        for h in HORIZONS:
            t = origin + h
            if t >= len(y_te):
                continue
            res[h]["p"].append(fc[h - 1] + exog_te[t])
            res[h]["a"].append(y_te[t])
    return ({h: mape(np.array(r["a"]), np.array(r["p"])) for h, r in res.items()},
            n_refits)


def main():
    print("=" * 70)
    print("EXPERIMENT 11 — Stage-1 re-estimation sensitivity")
    print("=" * 70)

    mats, y = build_data()
    with open(ROOT / "data" / "strategy_ridge_params.json") as f:
        rp = json.load(f)
    with open(ROOT / "data" / "strategy_elasticnet_params.json") as f:
        ep = json.load(f)

    factories = {
        "OLS-All": lambda: LinearRegression(),
        "Filter":  lambda: LinearRegression(),
        "PCA":     lambda: LinearRegression(),
        "Ridge":   lambda: Ridge(alpha=rp["lambda"]),
        "Lasso":   lambda: ElasticNet(alpha=ep["alpha"], l1_ratio=ep["l1_ratio"],
                                      max_iter=5000, random_state=SEED),
    }

    # --- Coefficient drift table (interpretable strategies) ---------------
    print("\nStage-1 coefficient drift (train vs train+val vs train+val+test):")
    drift_rows = []
    for name in ["OLS-All", "Filter", "Ridge", "Lasso"]:
        M = mats[name]
        windows = {
            "train":          (M["train"], y["train"]),
            "train+val":      (pd.concat([M["train"], M["val"]], ignore_index=True),
                               np.concatenate([y["train"], y["val"]])),
            "train+val+test": (pd.concat([M["train"], M["val"], M["test"]],
                                         ignore_index=True),
                               np.concatenate([y["train"], y["val"], y["test"]])),
        }
        for wname, (Xw, yw) in windows.items():
            m = factories[name]().fit(Xw, yw)
            for feat, coef in zip(Xw.columns, m.coef_):
                drift_rows.append({"strategy": name, "window": wname,
                                   "feature": feat, "coef": float(coef)})
    drift = pd.DataFrame(drift_rows)
    drift.to_csv(ROOT / "data" / "stage1_coef_drift.csv", index=False)
    piv = drift[drift.strategy == "Ridge"].pivot(index="feature",
                                                 columns="window", values="coef")
    piv = piv[["train", "train+val", "train+val+test"]]
    piv["abs_change"] = (piv["train+val+test"] - piv["train"]).abs()
    print("\nRidge coefficients, largest drift (top 8):")
    print(piv.sort_values("abs_change", ascending=False).head(8).round(1).to_string())

    # --- MAPE: static vs refit ---------------------------------------------
    rows = []
    for name, mfact in factories.items():
        t0 = time.time()
        M = mats[name]
        m_static = eval_static(M["train"], M["test"], y["train"], y["test"], mfact)
        m_refit, n_refits = eval_refit(M["train"], M["val"], M["test"],
                                       y["train"], y["val"], y["test"], mfact)
        for h in HORIZONS:
            rows.append({"strategy": name, "horizon": h,
                         "mape_static": m_static[h], "mape_refit": m_refit[h],
                         "delta": m_refit[h] - m_static[h]})
        print(f"  {name:<8} ({n_refits} refits) [{time.time()-t0:5.1f}s]  "
              + "  ".join(f"h={h}: {m_refit[h]-m_static[h]:+.3f}pp" for h in HORIZONS))

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data" / "stage1_refit_sensitivity.csv", index=False)

    # Sanity: static arm vs headline oracle results
    oracle = pd.read_csv(ROOT / "data" / "sarimax_results.csv")
    oracle["strategy"] = oracle["model"].str.extract(r"\((.*?)\)$")[0] \
        .str.replace(r"\s*\(.*\)$", "", regex=True)
    o_map = oracle.set_index(["strategy", "horizon_days"])["mape_pct"]
    worst = max(abs(r["mape_static"] - o_map.loc[(r["strategy"], r["horizon"])])
                for _, r in out.iterrows())
    print(f"\nSanity: static arm vs sarimax_results.csv — max |Δ| = {worst:.4f} pp "
          f"({'OK' if worst < 0.05 else 'MISMATCH — investigate'})")

    print("\nΔ MAPE (refit − static, pp; negative = refitting helps):")
    print(out.pivot_table(index="strategy", columns="horizon",
                          values="delta").round(3).to_string())
    print("\nSaved:")
    print("  data/stage1_refit_sensitivity.csv")
    print("  data/stage1_coef_drift.csv")


if __name__ == "__main__":
    main()
