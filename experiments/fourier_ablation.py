"""
Experiment 1 — Fourier with/without ablation (Note #2).

For each Fourier condition (with / without the 4 annual Fourier terms), refits
Stage-1 for each strategy, runs rolling-origin SARIMAX MAPE at all horizons,
computes 10-fold coefficient-stability CV, and computes XGBoost-on-residual
SHAP attribution for Ridge to test whether removing Fourier shifts the
seasonal-signal credit toward the climate predictors (temp_c, global_rad).

Outputs:
  - data/fourier_ablation_mape.csv      MAPE × strategy × horizon × condition
  - data/fourier_ablation_cv.csv        per-feature CV × strategy × condition
  - data/fourier_ablation_stage1.csv    Stage-1 R² and residual std per strategy × condition
  - data/fourier_ablation_shap.csv      mean |SHAP| per feature × condition (Ridge only)
  - plots/fourier_ablation_shap.png     bar chart of Ridge mean |SHAP| with/without Fourier
  - plots/fourier_ablation_mape.png     MAPE × horizon × strategy under both conditions
"""
from __future__ import annotations

import json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX

import builtins
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")


def _shap_tree_explainer(xgb_model):
    """Workaround for XGBoost 2.x+ / SHAP TreeExplainer incompatibility.

    XGBoost serialises base_score as '[value]' in its internal JSON; SHAP
    calls float() on that string and crashes. Temporarily replace
    builtins.float to strip the brackets, then restore.
    """
    _orig = builtins.float

    def _float(x):
        if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
            return _orig(x[1:-1])
        return _orig(x)

    builtins.float = _float
    try:
        explainer = shap.TreeExplainer(xgb_model.get_booster())
    finally:
        builtins.float = _orig
    return explainer

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
HORIZONS = [1, 3, 7, 30, 90, 180]
MAX_H = max(HORIZONS)
SEASONAL_ORDER = (0, 1, 1, 7)
BEST_ORDER = (2, 0, 1)

CLIMATE = ["temp_c", "wind_ms", "precip_mm", "sunshine_h",
           "global_rad", "pressure_hpa", "humidity_pct", "cloudiness", "nao"]
STRUCTURAL = ["price_eur_kwh", "gdp_mln_eur", "population",
              "wind_energy_gwh", "solar_energy_gwh"]
CONTINUOUS = CLIMATE + STRUCTURAL
RAW_CAL = ["day_of_week", "month", "is_weekend", "is_holiday"]
FOURIER = ["sin1_ann", "cos1_ann", "sin2_ann", "cos2_ann"]


def add_fourier(df):
    doy = df["date"].dt.dayofyear
    df = df.copy()
    df["sin1_ann"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos1_ann"] = np.cos(2 * np.pi * doy / 365.25)
    df["sin2_ann"] = np.sin(4 * np.pi * doy / 365.25)
    df["cos2_ann"] = np.cos(4 * np.pi * doy / 365.25)
    return df


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


def build_splits(include_fourier: bool):
    """Build train/val/test feature matrices for the given Fourier condition."""
    train = pd.read_csv(ROOT / "data" / "train.csv", parse_dates=["date"])
    val   = pd.read_csv(ROOT / "data" / "val.csv",   parse_dates=["date"])
    test  = pd.read_csv(ROOT / "data" / "test.csv",  parse_dates=["date"])

    if include_fourier:
        train = add_fourier(train)
        val   = add_fourier(val)
        test  = add_fourier(test)
        calendar = RAW_CAL + FOURIER
    else:
        calendar = RAW_CAL

    sx = StandardScaler()
    X_tr_c = sx.fit_transform(train[CONTINUOUS])
    X_va_c = sx.transform(val[CONTINUOUS])
    X_te_c = sx.transform(test[CONTINUOUS])

    def stack(arr, df):
        cont = pd.DataFrame(arr, columns=CONTINUOUS, index=df.index)
        cal  = df[calendar].reset_index(drop=True)
        return pd.concat([cont.reset_index(drop=True), cal], axis=1)

    X_tr = stack(X_tr_c, train)
    X_va = stack(X_va_c, val)
    X_te = stack(X_te_c, test)

    y_tr = train["demand_MW"].values
    y_va = val["demand_MW"].values
    y_te = test["demand_MW"].values

    # Strategy feature sets — refit Filter/Lasso under this condition
    retained = correlation_filter(train[CONTINUOUS], y_tr, threshold=0.80)
    filter_cols = retained + calendar

    # Lasso — refit on the new X to pick zero features under this condition
    with open(ROOT / "data" / "strategy_elasticnet_params.json") as f:
        ep = json.load(f)
    enet = ElasticNet(alpha=ep["alpha"], l1_ratio=ep["l1_ratio"],
                      max_iter=5000, random_state=SEED).fit(X_tr.values, y_tr)
    nonzero = [c for c, k in zip(X_tr.columns, enet.coef_) if k != 0]
    lasso_cols = sorted(set(nonzero) | set(calendar),
                        key=lambda c: list(X_tr.columns).index(c))

    return {
        "train": {"X": X_tr, "y": y_tr},
        "val":   {"X": X_va, "y": y_va},
        "test":  {"X": X_te, "y": y_te},
        "calendar": calendar,
        "filter_cols": filter_cols,
        "lasso_cols": lasso_cols,
    }


def mape(y_true, y_pred):
    return 100 * np.mean(np.abs((y_true - y_pred) / y_true))


def fit_two_stage(X_tr, y_tr, stage1):
    s = stage1.fit(X_tr, y_tr)
    resid = y_tr - s.predict(X_tr)
    sx = SARIMAX(resid, order=BEST_ORDER, seasonal_order=SEASONAL_ORDER,
                 enforce_stationarity=False, enforce_invertibility=False)
    return s, sx.fit(disp=False), resid


def rolling_eval(stage1, sarima_r, X_te, y_te, eval_step=30):
    exog_te = stage1.predict(X_te)
    resid_te = y_te - exog_te
    results = {h: {"pred": [], "actual": []} for h in HORIZONS}
    for origin in range(0, len(resid_te) - MAX_H, eval_step):
        if origin == 0:
            res_ext = sarima_r
        else:
            res_ext = sarima_r.append(endog=resid_te[:origin], refit=False)
        fc = res_ext.forecast(steps=MAX_H)
        for h in HORIZONS:
            t = origin + h
            if t >= len(resid_te):
                continue
            results[h]["pred"].append(fc[h - 1] + exog_te[t])
            results[h]["actual"].append(y_te[t])
    return {h: mape(np.array(r["actual"]), np.array(r["pred"]))
            for h, r in results.items() if r["actual"]}


def coef_cv(X, y, model_factory):
    tscv = TimeSeriesSplit(n_splits=10)
    coefs = []
    for tr_idx, _ in tscv.split(X):
        m = model_factory()
        m.fit(X.iloc[tr_idx], y[tr_idx])
        coefs.append(pd.Series(m.coef_, index=X.columns))
    df = pd.DataFrame(coefs)
    cv = (df.std() / df.mean().abs()).clip(upper=5.0)
    return cv


def run_strategies(feats):
    """Refit all four strategies; return Stage-1 models, MAPE, CV."""
    with open(ROOT / "data" / "strategy_ridge_params.json") as f:
        rp = json.load(f)
    with open(ROOT / "data" / "strategy_elasticnet_params.json") as f:
        ep = json.load(f)

    X_tr = feats["train"]["X"]
    y_tr = feats["train"]["y"]
    X_te = feats["test"]["X"]
    y_te = feats["test"]["y"]

    strategies = {
        "OLS-All": (X_tr, X_te, lambda: LinearRegression()),
        "Filter":  (X_tr[feats["filter_cols"]], X_te[feats["filter_cols"]],
                    lambda: LinearRegression()),
        "Ridge":   (X_tr, X_te, lambda: Ridge(alpha=rp["lambda"])),
        "Lasso":   (X_tr[feats["lasso_cols"]], X_te[feats["lasso_cols"]],
                    lambda: ElasticNet(alpha=ep["alpha"],
                                       l1_ratio=ep["l1_ratio"],
                                       max_iter=5000, random_state=SEED)),
    }

    mape_rows, cv_rows, stage1_rows = [], [], []
    stage1_store = {}

    for name, (X_tr_s, X_te_s, mfact) in strategies.items():
        t0 = time.time()
        stage1, sarima_r, resid_tr = fit_two_stage(X_tr_s, y_tr, mfact())
        r2 = stage1.score(X_tr_s, y_tr)
        mape_dict = rolling_eval(stage1, sarima_r, X_te_s, y_te)
        cv_series = coef_cv(X_tr_s, y_tr, mfact)
        stage1_store[name] = {"stage1": stage1, "sarima_r": sarima_r,
                              "X_tr": X_tr_s, "X_te": X_te_s, "y_te": y_te,
                              "resid_tr": resid_tr, "stage1_r2": r2,
                              "resid_std": resid_tr.std()}
        print(f"    {name:<8}  R²={r2:.4f}  resid_std={resid_tr.std():.1f}MW  "
              + "  ".join(f"h={h}: {v:.2f}%" for h, v in mape_dict.items())
              + f"  [{time.time()-t0:.1f}s]")
        for h, v in mape_dict.items():
            mape_rows.append({"strategy": name, "horizon": h, "mape": v})
        for feat, val in cv_series.items():
            cv_rows.append({"strategy": name, "feature": feat, "cv": float(val)})
        stage1_rows.append({"strategy": name, "stage1_r2": r2,
                             "resid_std": resid_tr.std()})

    return (pd.DataFrame(mape_rows), pd.DataFrame(cv_rows),
            pd.DataFrame(stage1_rows), stage1_store)


def shap_ridge(stage1_store, feats, condition):
    """XGBoost surrogate on Ridge Stage-1 test residuals + TreeSHAP."""
    s = stage1_store["Ridge"]
    X_te = s["X_te"]
    y_te = s["y_te"]
    resid_te = y_te - s["stage1"].predict(X_te)

    model = xgb.XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        subsample=0.8, random_state=SEED, n_jobs=-1, tree_method="hist"
    )
    model.fit(X_te, resid_te)
    r2 = model.score(X_te, resid_te)

    explainer = _shap_tree_explainer(model)
    shap_vals = explainer.shap_values(X_te)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    return pd.DataFrame({
        "condition": condition,
        "feature": X_te.columns,
        "mean_abs_shap": mean_abs,
        "surrogate_r2": r2,
    })


def main():
    print("=" * 70)
    print("EXPERIMENT 1 — Fourier with/without ablation")
    print("=" * 70)

    all_mape, all_cv, all_stage1, all_shap = [], [], [], []

    for include_fourier in (True, False):
        condition = "with_fourier" if include_fourier else "no_fourier"
        print(f"\n=== Condition: {condition} ===")
        feats = build_splits(include_fourier)
        print(f"  Calendar features: {feats['calendar']}")
        print(f"  Filter cols ({len(feats['filter_cols'])})")
        print(f"  Lasso cols  ({len(feats['lasso_cols'])})")

        print("  Running rolling-origin SARIMAX + CV stability...")
        m, c, s, store = run_strategies(feats)
        m["condition"] = condition; c["condition"] = condition; s["condition"] = condition
        all_mape.append(m); all_cv.append(c); all_stage1.append(s)

        print("  Computing SHAP (Ridge surrogate)...")
        t0 = time.time()
        sh = shap_ridge(store, feats, condition)
        print(f"    XGB R² on test residuals: {sh['surrogate_r2'].iloc[0]:.4f}  [{time.time()-t0:.1f}s]")
        all_shap.append(sh)

    mape_df = pd.concat(all_mape, ignore_index=True)
    cv_df   = pd.concat(all_cv, ignore_index=True)
    stage1_df = pd.concat(all_stage1, ignore_index=True)
    shap_df = pd.concat(all_shap, ignore_index=True)

    mape_df.to_csv(ROOT / "data" / "fourier_ablation_mape.csv", index=False)
    cv_df.to_csv(ROOT / "data" / "fourier_ablation_cv.csv", index=False)
    stage1_df.to_csv(ROOT / "data" / "fourier_ablation_stage1.csv", index=False)
    shap_df.to_csv(ROOT / "data" / "fourier_ablation_shap.csv", index=False)

    # ---------- Headline tables ----------
    print()
    print("=" * 70)
    print("MAPE — Δ (no_fourier − with_fourier) per strategy × horizon")
    print("=" * 70)
    piv = mape_df.pivot_table(index=["strategy","horizon"],
                              columns="condition", values="mape").round(3)
    piv["delta"] = (piv["no_fourier"] - piv["with_fourier"]).round(3)
    print(piv)

    print()
    print("=" * 70)
    print("Stage-1 R² — with vs without Fourier per strategy")
    print("=" * 70)
    s_piv = stage1_df.pivot(index="strategy", columns="condition",
                            values="stage1_r2").round(4)
    s_piv["delta"] = (s_piv["no_fourier"] - s_piv["with_fourier"]).round(4)
    print(s_piv)

    print()
    print("=" * 70)
    print("Ridge mean |SHAP| — Δ no_fourier − with_fourier per feature")
    print("=" * 70)
    sh_piv = shap_df.pivot(index="feature", columns="condition",
                           values="mean_abs_shap")
    sh_piv = sh_piv.reindex(columns=["with_fourier", "no_fourier"])
    sh_piv["delta"] = sh_piv["no_fourier"] - sh_piv.get("with_fourier", 0)
    sh_piv = sh_piv.sort_values("delta", ascending=False).round(2)
    print(sh_piv.head(15))
    print("...")
    print(sh_piv.tail(8))

    # ---------- Plots ----------
    print("\nGenerating plots...")

    # SHAP comparison
    common = sh_piv.dropna().sort_values("with_fourier", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(common))
    w = 0.4
    ax.barh(y - w/2, common["with_fourier"], w, label="with Fourier",
            color="#4477aa", edgecolor="white")
    ax.barh(y + w/2, common["no_fourier"], w, label="without Fourier",
            color="#cc6677", edgecolor="white")
    ax.set_yticks(y); ax.set_yticklabels(common.index, fontsize=8)
    ax.set_xlabel("Mean |SHAP| value")
    ax.set_title("Ridge Stage-1 Residual SHAP — with vs without Fourier terms")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(ROOT / "plots" / "fourier_ablation_shap.png",
                bbox_inches="tight", dpi=120)
    plt.close()
    print(f"  Saved: plots/fourier_ablation_shap.png")

    # MAPE × horizon
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"OLS-All": "#888888", "Filter": "#4477aa",
              "Ridge": "#117733", "Lasso": "#cc6677"}
    for strat, sub in mape_df.groupby("strategy"):
        for cond, marker, ls in [("with_fourier", "o", "-"),
                                  ("no_fourier", "s", "--")]:
            d = sub[sub["condition"] == cond].sort_values("horizon")
            ax.plot(d["horizon"], d["mape"], marker=marker, linestyle=ls,
                    color=colors[strat], alpha=0.9,
                    label=f"{strat} ({cond.replace('_', ' ')})")
    ax.set_xscale("log")
    ax.set_xticks(HORIZONS); ax.set_xticklabels(HORIZONS)
    ax.set_xlabel("Forecast horizon (days, log scale)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("MAPE × Horizon — With vs Without Fourier Terms")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROOT / "plots" / "fourier_ablation_mape.png",
                bbox_inches="tight", dpi=120)
    plt.close()
    print(f"  Saved: plots/fourier_ablation_mape.png")

    print("\nSaved:")
    for f in ["fourier_ablation_mape.csv", "fourier_ablation_cv.csv",
              "fourier_ablation_stage1.csv", "fourier_ablation_shap.csv"]:
        print(f"  data/{f}")


if __name__ == "__main__":
    main()
