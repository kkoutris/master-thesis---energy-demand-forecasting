"""
Experiment 4 — XGBoost surrogate cross-validation (Note #8).

Current pattern in notebook 04 cell 14 fits an XGBoost surrogate on the
Stage-1 *test* residuals with no hold-out:

    xgb_model.fit(X_shap_test, stage1_resid_test)   # in-sample
    r2 = 1 - var(resid - xgb_model.predict(X_shap_test)) / var(resid)
    shap_values = explainer.shap_values(X_shap_test)

The R² of 0.96–0.97 is in-sample on the very data the explainer then
attributes, which inflates both the headline R² and the apparent confidence
of the SHAP numbers (used in H3 and in Experiment 1).

This script reruns the same surrogate under two out-of-sample protocols:

  (1) PRIMARY FIX — train→test fit: refit XGBoost on Stage-1 *training*
      residuals (y_train - stage1.predict(X_train)) and evaluate / explain
      on test residuals. SHAP is computed on the train-fitted model.

  (2) ROBUSTNESS — 5-fold TimeSeriesSplit CV on the test residuals: split
      the test window into 5 forward-chained folds, fit on the past slice,
      evaluate on the next; report mean and per-fold OOS R².

The original in-sample R² is also recomputed here so the three numbers can
be compared side-by-side per strategy.

Outputs:
  - data/xgb_surrogate_r2.csv   one row per strategy with three R²s + per-fold CV
  - data/xgb_surrogate_shap.csv mean |SHAP| per (strategy, feature) under the
                                train→test surrogate (the "corrected" attributions)
  - plots/xgb_surrogate_r2.png  bar chart: in-sample vs train→test vs CV mean
  - plots/xgb_surrogate_shap_compare.png Ridge mean|SHAP| in-sample vs train→test
"""
from __future__ import annotations

import builtins
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
PCA_N_COMPONENTS = 9
CV_SPLITS = 5

CLIMATE = ["temp_c", "wind_ms", "precip_mm", "sunshine_h",
           "global_rad", "pressure_hpa", "humidity_pct", "cloudiness", "nao"]
STRUCTURAL = ["price_eur_kwh", "gdp_mln_eur", "population",
              "wind_energy_gwh", "solar_energy_gwh"]
CONTINUOUS = CLIMATE + STRUCTURAL
RAW_CAL = ["day_of_week", "month", "is_weekend", "is_holiday"]
FOURIER = ["sin1_ann", "cos1_ann", "sin2_ann", "cos2_ann"]
CALENDAR = RAW_CAL + FOURIER

# Matches notebook 04 cell 14
XGB_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbosity=0, n_jobs=-1, tree_method="hist",
)


def _shap_tree_explainer(xgb_model):
    """XGBoost 2.x base_score-as-JSON workaround for SHAP TreeExplainer."""
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


def add_fourier(df: pd.DataFrame) -> pd.DataFrame:
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


def build_splits():
    train = add_fourier(pd.read_csv(ROOT / "data" / "train.csv",    parse_dates=["date"]))
    test = add_fourier(pd.read_csv(ROOT / "data" / "test.csv", parse_dates=["date"]))

    sx = StandardScaler()
    X_tr_c = sx.fit_transform(train[CONTINUOUS])
    X_te_c = sx.transform(test[CONTINUOUS])

    def stack(arr, df):
        cont = pd.DataFrame(arr, columns=CONTINUOUS, index=df.index)
        cal = df[CALENDAR].reset_index(drop=True)
        return pd.concat([cont.reset_index(drop=True), cal], axis=1)

    X_tr = stack(X_tr_c, train)
    X_te = stack(X_te_c, test)
    y_tr = train["demand_MW"].values
    y_te = test["demand_MW"].values

    retained = correlation_filter(train[CONTINUOUS], y_tr, threshold=0.80)
    filter_cols = retained + CALENDAR

    with open(ROOT / "data" / "strategy_elasticnet_params.json") as f:
        ep = json.load(f)
    enet = ElasticNet(alpha=ep["alpha"], l1_ratio=ep["l1_ratio"],
                      max_iter=5000, random_state=SEED).fit(X_tr.values, y_tr)
    nonzero = [c for c, k in zip(X_tr.columns, enet.coef_) if k != 0]
    lasso_cols = sorted(set(nonzero) | set(CALENDAR),
                        key=lambda c: list(X_tr.columns).index(c))

    pca_obj = PCA(n_components=PCA_N_COMPONENTS, random_state=SEED).fit(X_tr[CONTINUOUS].values)
    pc_cols = [f"PC{i+1}" for i in range(PCA_N_COMPONENTS)]

    def to_pca(X_wide):
        pcs = pca_obj.transform(X_wide[CONTINUOUS].values)
        pc_df = pd.DataFrame(pcs, columns=pc_cols, index=X_wide.index)
        return pd.concat([pc_df, X_wide[CALENDAR].reset_index(drop=True)], axis=1)

    return {
        "X_tr_wide": X_tr, "X_te_wide": X_te,
        "y_tr": y_tr, "y_te": y_te,
        "filter_cols": filter_cols, "lasso_cols": lasso_cols,
        "to_pca": to_pca, "pc_cols": pc_cols,
    }


def stage1_resid_pair(X_tr_strategy, X_te_strategy, y_tr, y_te, model):
    """Fit Stage-1 model; return (train_resid, test_resid)."""
    model.fit(X_tr_strategy, y_tr)
    return y_tr - model.predict(X_tr_strategy), y_te - model.predict(X_te_strategy)


def fit_xgb(X, y):
    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(X, y)
    return m


def r2_var(y_true, y_pred):
    """Variance-form R² to match notebook 04 cell 14."""
    return 1.0 - np.var(y_true - y_pred) / np.var(y_true)


def cv_test_residuals(X_te_shap, resid_te, n_splits=CV_SPLITS):
    """TimeSeriesSplit CV on the test residuals. Returns per-fold OOS R²."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_r2 = []
    for tr_idx, va_idx in tscv.split(X_te_shap):
        m = fit_xgb(X_te_shap.iloc[tr_idx], resid_te[tr_idx])
        pred = m.predict(X_te_shap.iloc[va_idx])
        fold_r2.append(r2_var(resid_te[va_idx], pred))
    return fold_r2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("EXPERIMENT 4 — XGBoost surrogate cross-validation")
    print("=" * 70)

    s = build_splits()
    X_tr_wide, X_te_wide = s["X_tr_wide"], s["X_te_wide"]
    y_tr, y_te = s["y_tr"], s["y_te"]

    # The XGBoost surrogate (and SHAP) always sees the full 22-column space,
    # matching notebook 04 cell 14's choice of X_shap = s3_te (Ridge test).
    X_shap_tr = X_tr_wide
    X_shap_te = X_te_wide

    print(f"  Train n={len(X_shap_tr)}  Test n={len(X_shap_te)}  "
          f"Features={X_shap_tr.shape[1]}")
    print(f"  XGB: {XGB_PARAMS}")
    print()

    with open(ROOT / "data" / "strategy_ridge_params.json") as f:
        rp = json.load(f)
    with open(ROOT / "data" / "strategy_elasticnet_params.json") as f:
        ep = json.load(f)

    strategies = [
        ("OLS-All", X_tr_wide, X_te_wide, LinearRegression()),
        ("Filter", X_tr_wide[s["filter_cols"]], X_te_wide[s["filter_cols"]],
         LinearRegression()),
        ("PCA", s["to_pca"](X_tr_wide), s["to_pca"](X_te_wide), LinearRegression()),
        ("Ridge", X_tr_wide, X_te_wide, Ridge(alpha=rp["lambda"])),
        ("Lasso", X_tr_wide[s["lasso_cols"]], X_te_wide[s["lasso_cols"]],
         ElasticNet(alpha=ep["alpha"], l1_ratio=ep["l1_ratio"],
                    max_iter=5000, random_state=SEED)),
    ]

    rows = []
    shap_rows = []

    for name, X_tr_s, X_te_s, model in strategies:
        t0 = time.time()
        resid_tr, resid_te = stage1_resid_pair(X_tr_s, X_te_s, y_tr, y_te, model)

        # (a) IN-SAMPLE on test — the current 0.96–0.97 number
        m_in = fit_xgb(X_shap_te, resid_te)
        r2_in = r2_var(resid_te, m_in.predict(X_shap_te))

        # (b) PRIMARY FIX — train→test
        m_oos = fit_xgb(X_shap_tr, resid_tr)
        r2_oos = r2_var(resid_te, m_oos.predict(X_shap_te))
        r2_oos_train = r2_var(resid_tr, m_oos.predict(X_shap_tr))

        # (c) ROBUSTNESS — 5-fold TimeSeriesSplit CV on the test residuals
        fold_r2 = cv_test_residuals(X_shap_te, resid_te, CV_SPLITS)
        r2_cv_mean = float(np.mean(fold_r2))
        r2_cv_std = float(np.std(fold_r2))

        # SHAP under the train→test surrogate (the "corrected" attributions)
        explainer = _shap_tree_explainer(m_oos)
        sv = explainer.shap_values(X_shap_te)
        mean_abs = np.abs(sv).mean(axis=0)

        # Also SHAP under the in-sample surrogate so we can quantify the shift
        explainer_in = _shap_tree_explainer(m_in)
        sv_in = explainer_in.shap_values(X_shap_te)
        mean_abs_in = np.abs(sv_in).mean(axis=0)

        rows.append({
            "strategy": name,
            "r2_in_sample_test": r2_in,
            "r2_train_to_test": r2_oos,
            "r2_train_in_sample": r2_oos_train,
            "r2_tscv_mean": r2_cv_mean,
            "r2_tscv_std": r2_cv_std,
            "r2_tscv_per_fold": ";".join(f"{x:.4f}" for x in fold_r2),
            "total_abs_shap_in_sample": float(mean_abs_in.sum()),
            "total_abs_shap_train_to_test": float(mean_abs.sum()),
        })

        for feat, ms, ms_in in zip(X_shap_te.columns, mean_abs, mean_abs_in):
            shap_rows.append({
                "strategy": name, "feature": feat,
                "mean_abs_shap_in_sample": float(ms_in),
                "mean_abs_shap_train_to_test": float(ms),
            })

        print(f"  {name:<8}  R²(in-sample) = {r2_in:.4f}   "
              f"R²(train→test) = {r2_oos:.4f}   "
              f"R²(CV mean) = {r2_cv_mean:.4f} ± {r2_cv_std:.4f}   "
              f"[{time.time()-t0:5.1f}s]")
        print(f"            per-fold CV R²: "
              f"[{', '.join(f'{x:+.3f}' for x in fold_r2)}]")

    r2_df = pd.DataFrame(rows)
    shap_df = pd.DataFrame(shap_rows)

    r2_df.to_csv(ROOT / "data" / "xgb_surrogate_r2.csv", index=False)
    shap_df.to_csv(ROOT / "data" / "xgb_surrogate_shap.csv", index=False)

    # ---------- Headline tables ----------
    print()
    print("=" * 70)
    print("R² summary — in-sample (test) vs train→test vs 5-fold TS-CV")
    print("=" * 70)
    print(r2_df[["strategy", "r2_in_sample_test", "r2_train_to_test",
                 "r2_tscv_mean", "r2_tscv_std"]].round(4).to_string(index=False))

    print()
    print("=" * 70)
    print("Ridge mean |SHAP| — in-sample vs train→test (top 15 by in-sample)")
    print("=" * 70)
    rd = shap_df[shap_df["strategy"] == "Ridge"].copy()
    rd["delta"] = rd["mean_abs_shap_train_to_test"] - rd["mean_abs_shap_in_sample"]
    rd["pct_change"] = 100 * rd["delta"] / rd["mean_abs_shap_in_sample"]
    rd = rd.sort_values("mean_abs_shap_in_sample", ascending=False)
    print(rd[["feature", "mean_abs_shap_in_sample",
              "mean_abs_shap_train_to_test", "delta", "pct_change"]]
          .head(15).round(2).to_string(index=False))

    # ---------- Plots ----------
    print("\nGenerating plots...")

    # R² bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(r2_df))
    w = 0.27
    ax.bar(x - w, r2_df["r2_in_sample_test"], w,
           label="In-sample on test (current)", color="#cc6677", edgecolor="white")
    ax.bar(x, r2_df["r2_train_to_test"], w,
           label="Train → test (primary fix)", color="#4477aa", edgecolor="white")
    ax.bar(x + w, r2_df["r2_tscv_mean"], w,
           yerr=r2_df["r2_tscv_std"], capsize=4,
           label="5-fold TS-CV on test (mean ± sd)",
           color="#117733", edgecolor="white")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(r2_df["strategy"])
    ax.set_ylabel("XGBoost surrogate R² on Stage-1 residuals")
    ax.set_title("Surrogate R² — in-sample vs out-of-sample protocols")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(ROOT / "plots" / "xgb_surrogate_r2.png",
                bbox_inches="tight", dpi=120)
    plt.close()
    print("  Saved: plots/xgb_surrogate_r2.png")

    # Ridge SHAP comparison: in-sample vs train→test
    common = rd.set_index("feature")[
        ["mean_abs_shap_in_sample", "mean_abs_shap_train_to_test"]
    ].sort_values("mean_abs_shap_in_sample", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(common))
    w = 0.4
    ax.barh(y - w/2, common["mean_abs_shap_in_sample"], w,
            label="In-sample (current)", color="#cc6677", edgecolor="white")
    ax.barh(y + w/2, common["mean_abs_shap_train_to_test"], w,
            label="Train → test (corrected)", color="#4477aa", edgecolor="white")
    ax.set_yticks(y); ax.set_yticklabels(common.index, fontsize=8)
    ax.set_xlabel("Mean |SHAP| value (MW)")
    ax.set_title("Ridge surrogate SHAP — in-sample vs train→test attribution")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(ROOT / "plots" / "xgb_surrogate_shap_compare.png",
                bbox_inches="tight", dpi=120)
    plt.close()
    print("  Saved: plots/xgb_surrogate_shap_compare.png")

    print("\nSaved data:")
    print("  data/xgb_surrogate_r2.csv")
    print("  data/xgb_surrogate_shap.csv")


if __name__ == "__main__":
    main()
