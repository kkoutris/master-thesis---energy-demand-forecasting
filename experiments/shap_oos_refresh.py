"""
OOS-SHAP refresh — corrects two downstream uses of in-sample SHAP arrays.

Experiment 4 (xgb_surrogate_cv.py) established that the original surrogate
diagnostic (fit XGBoost on test residuals, explain on the same test
residuals) overfits — R² collapses 0.96 → 0.35 under train→test, and the
per-feature mean |SHAP| values shift by ±50–400 %. Two other places in the
thesis are still using the in-sample SHAP arrays:

  (A) Experiment 1's Ridge mean |SHAP| with/without Fourier
      (data/fourier_ablation_shap.csv). The "+109 % climate-SHAP inflation
      when Fourier is removed" headline was computed from two in-sample
      surrogates and inherits the same overfit.

  (B) Notebook 04 cell 19's H2 information-loss bootstrap CI
      (data/interpretability_h2_bootstrap.csv). The observed gap (+8.59 pp)
      and CI [+8.01, +9.18] were bootstrapped from the in-sample
      shap_values arrays for Filter and Ridge.

This script reruns both under the train→test surrogate (fit XGB on
training residuals, explain on test) so all SHAP-derived numbers in the
thesis come from one consistent methodology.

Outputs (parallel to the in-sample versions, suffix `_oos`):
  - data/fourier_ablation_shap_oos.csv
  - data/interpretability_h2_bootstrap_oos.csv
  - plots/fourier_ablation_shap_oos.png
  - plots/h2_bootstrap_oos_ci.png
  - plots/h2_bootstrap_oos_ci.pdf  vector copy of the above
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

from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N_BOOT_H2 = 2000
FILTERED_OUT_S1 = ["population", "sunshine_h"]  # matches notebook 04 cell 19

CLIMATE = ["temp_c", "wind_ms", "precip_mm", "sunshine_h",
           "global_rad", "pressure_hpa", "humidity_pct", "cloudiness", "nao"]
STRUCTURAL = ["price_eur_kwh", "gdp_mln_eur", "population",
              "wind_energy_gwh", "solar_energy_gwh"]
CONTINUOUS = CLIMATE + STRUCTURAL
RAW_CAL = ["day_of_week", "month", "is_weekend", "is_holiday"]
FOURIER = ["sin1_ann", "cos1_ann", "sin2_ann", "cos2_ann"]

# Matches notebook 04 cell 14 — the canonical surrogate hyperparameters.
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


def build_splits(include_fourier: bool):
    """Train/test feature matrices under the Fourier flag."""
    train = pd.read_csv(ROOT / "data" / "train.csv", parse_dates=["date"])
    test  = pd.read_csv(ROOT / "data" / "test.csv",  parse_dates=["date"])
    if include_fourier:
        train = add_fourier(train)
        test  = add_fourier(test)
        calendar = RAW_CAL + FOURIER
    else:
        calendar = RAW_CAL

    sx = StandardScaler()
    X_tr_c = sx.fit_transform(train[CONTINUOUS])
    X_te_c = sx.transform(test[CONTINUOUS])

    def stack(arr, df):
        cont = pd.DataFrame(arr, columns=CONTINUOUS, index=df.index)
        cal  = df[calendar].reset_index(drop=True)
        return pd.concat([cont.reset_index(drop=True), cal], axis=1)

    X_tr = stack(X_tr_c, train)
    X_te = stack(X_te_c, test)
    y_tr = train["demand_MW"].values
    y_te = test["demand_MW"].values

    retained = correlation_filter(train[CONTINUOUS], y_tr, threshold=0.80)
    filter_cols = retained + calendar

    return {
        "X_tr": X_tr, "X_te": X_te,
        "y_tr": y_tr, "y_te": y_te,
        "calendar": calendar,
        "filter_cols": filter_cols,
    }


def oos_shap(X_tr_strat, X_te_strat, X_shap_tr, X_shap_te, y_tr, y_te, stage1_model):
    """Train→test surrogate: fit Stage-1, then XGBoost on training residuals,
    explain on test residuals. Returns (sv_test, mean_abs, r2_oos).
    """
    stage1_model.fit(X_tr_strat, y_tr)
    resid_tr = y_tr - stage1_model.predict(X_tr_strat)
    resid_te = y_te - stage1_model.predict(X_te_strat)

    m_oos = xgb.XGBRegressor(**XGB_PARAMS)
    m_oos.fit(X_shap_tr, resid_tr)
    pred_te = m_oos.predict(X_shap_te)
    r2_oos = 1.0 - np.var(resid_te - pred_te) / np.var(resid_te)

    explainer = _shap_tree_explainer(m_oos)
    sv = explainer.shap_values(X_shap_te)
    return sv, np.abs(sv).mean(axis=0), float(r2_oos)


# ---------------------------------------------------------------------------
# (A) Experiment 1's SHAP table under OOS surrogate
# ---------------------------------------------------------------------------
def refresh_exp1_shap():
    print("=" * 70)
    print("(A) Exp 1 SHAP refresh — Ridge with vs without Fourier under OOS")
    print("=" * 70)

    with open(ROOT / "data" / "strategy_ridge_params.json") as f:
        rp = json.load(f)

    rows = []
    for include_fourier in (True, False):
        condition = "with_fourier" if include_fourier else "no_fourier"
        s = build_splits(include_fourier)
        # Ridge uses full feature set (continuous + calendar)
        # Surrogate is fit on the same shape as Stage-1 features.
        t0 = time.time()
        _, mean_abs, r2 = oos_shap(
            s["X_tr"], s["X_te"], s["X_tr"], s["X_te"],
            s["y_tr"], s["y_te"], Ridge(alpha=rp["lambda"]),
        )
        print(f"  Ridge / {condition:<13} OOS R² = {r2:+.4f}   "
              f"total mean|SHAP| = {mean_abs.sum():.1f} MW   [{time.time()-t0:.1f}s]")
        for feat, m in zip(s["X_tr"].columns, mean_abs):
            rows.append({"condition": condition, "feature": feat,
                          "mean_abs_shap_oos": float(m),
                          "surrogate_r2_oos": r2})

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data" / "fourier_ablation_shap_oos.csv", index=False)

    # Compare to in-sample version
    in_sample = pd.read_csv(ROOT / "data" / "fourier_ablation_shap.csv")
    in_sample = in_sample.rename(columns={"mean_abs_shap": "mean_abs_shap_insample",
                                            "surrogate_r2": "surrogate_r2_insample"})
    merged = df.merge(in_sample[["condition", "feature",
                                  "mean_abs_shap_insample",
                                  "surrogate_r2_insample"]],
                       on=["condition", "feature"], how="left")
    merged["delta_oos_minus_insample"] = (merged["mean_abs_shap_oos"]
                                            - merged["mean_abs_shap_insample"])

    print()
    print("Per-condition surrogate R²: in-sample → OOS")
    r2_tbl = merged.groupby("condition")[
        ["surrogate_r2_insample", "surrogate_r2_oos"]].first().round(4)
    print(r2_tbl)

    print()
    print("Ridge mean|SHAP| under OOS — δ (no_fourier − with_fourier):")
    piv = df.pivot(index="feature", columns="condition",
                    values="mean_abs_shap_oos")
    piv = piv.reindex(columns=["with_fourier", "no_fourier"])
    piv["delta"] = piv["no_fourier"] - piv["with_fourier"]
    piv["pct_change"] = 100 * piv["delta"] / piv["with_fourier"].replace(0, np.nan)
    piv = piv.sort_values("delta", ascending=False).round(2)
    print(piv.head(15))
    print("...")
    print(piv.tail(8))

    # Plot
    print("\nGenerating Exp-1 OOS plot...")
    common = piv.dropna(subset=["with_fourier", "no_fourier"]).sort_values(
        "with_fourier", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(common))
    w = 0.4
    ax.barh(y - w/2, common["with_fourier"], w, label="with Fourier",
            color="#4477aa", edgecolor="white")
    ax.barh(y + w/2, common["no_fourier"], w, label="without Fourier",
            color="#cc6677", edgecolor="white")
    ax.set_yticks(y); ax.set_yticklabels(common.index, fontsize=8)
    ax.set_xlabel("Mean |SHAP| value (MW)")
    ax.set_title("Ridge SHAP (OOS surrogate) — with vs without Fourier")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    out = ROOT / "plots" / "fourier_ablation_shap_oos.png"
    plt.savefig(out, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"  Saved: {out.relative_to(ROOT)}")

    return merged


# ---------------------------------------------------------------------------
# (B) H2 information-loss bootstrap under OOS surrogate
# ---------------------------------------------------------------------------
def refresh_h2_bootstrap():
    print()
    print("=" * 70)
    print("(B) H2 information-loss bootstrap refresh — OOS Filter vs Ridge")
    print("=" * 70)

    with open(ROOT / "data" / "strategy_ridge_params.json") as f:
        rp = json.load(f)

    s = build_splits(include_fourier=True)
    # Match notebook 04 cell 19's convention: the surrogate's input space is
    # the full Ridge feature matrix (so SHAP can attribute to any feature
    # including those Filter dropped).
    X_shap_tr = s["X_tr"]
    X_shap_te = s["X_te"]

    print("  Fitting OOS surrogate for Filter (Stage-1 on filter_cols only)...")
    t0 = time.time()
    sv_filter, mean_filter, r2_filter = oos_shap(
        s["X_tr"][s["filter_cols"]], s["X_te"][s["filter_cols"]],
        X_shap_tr, X_shap_te, s["y_tr"], s["y_te"], LinearRegression(),
    )
    print(f"    OOS R² = {r2_filter:+.4f}  [{time.time()-t0:.1f}s]")

    print("  Fitting OOS surrogate for Ridge (Stage-1 on full features)...")
    t0 = time.time()
    sv_ridge, mean_ridge, r2_ridge = oos_shap(
        s["X_tr"], s["X_te"], X_shap_tr, X_shap_te, s["y_tr"], s["y_te"],
        Ridge(alpha=rp["lambda"]),
    )
    print(f"    OOS R² = {r2_ridge:+.4f}  [{time.time()-t0:.1f}s]")

    s3_cols = list(X_shap_te.columns)
    filtered_idx = [s3_cols.index(f) for f in FILTERED_OUT_S1 if f in s3_cols]

    def pct_filtered(sv):
        mean_abs = np.abs(sv).mean(axis=0)
        return 100 * mean_abs[filtered_idx].sum() / mean_abs.sum()

    def info_loss_gap(sv_f, sv_r):
        return pct_filtered(sv_f) - pct_filtered(sv_r)

    rng = np.random.default_rng(SEED)
    n_test = sv_filter.shape[0]
    boot_gaps = np.empty(N_BOOT_H2)
    for i in range(N_BOOT_H2):
        idx = rng.integers(0, n_test, n_test)
        boot_gaps[i] = info_loss_gap(sv_filter[idx], sv_ridge[idx])

    obs_gap = info_loss_gap(sv_filter, sv_ridge)
    ci_lo = float(np.percentile(boot_gaps, 2.5))
    ci_hi = float(np.percentile(boot_gaps, 97.5))
    excludes_zero = not (ci_lo <= 0 <= ci_hi)

    print()
    print(f"  Filter %|SHAP| in filtered features (OOS): {pct_filtered(sv_filter):.2f}%")
    print(f"  Ridge  %|SHAP| in filtered features (OOS): {pct_filtered(sv_ridge):.2f}%")
    print(f"  Observed gap (OOS) : {obs_gap:+.4f} pp")
    print(f"  Bootstrap 95% CI   : [{ci_lo:+.4f}, {ci_hi:+.4f}] pp")
    print(f"  CI excludes 0      : {excludes_zero}")

    out_df = pd.DataFrame([{
        "observed_gap_pp": round(obs_gap, 4),
        "bootstrap_ci_lo_pp": round(ci_lo, 4),
        "bootstrap_ci_hi_pp": round(ci_hi, 4),
        "ci_excludes_zero": excludes_zero,
        "n_bootstrap": N_BOOT_H2,
        "filtered_features": str(FILTERED_OUT_S1),
        "surrogate_r2_filter_oos": round(r2_filter, 4),
        "surrogate_r2_ridge_oos": round(r2_ridge, 4),
    }])
    out_df.to_csv(ROOT / "data" / "interpretability_h2_bootstrap_oos.csv",
                   index=False)

    # Compare to in-sample baseline
    in_sample = pd.read_csv(ROOT / "data" / "interpretability_h2_bootstrap.csv")
    in_row = in_sample.iloc[0]
    print()
    print("Comparison vs in-sample (notebook 04 cell 19):")
    print(f"  observed_gap_pp     : in-sample {in_row['observed_gap_pp']:+.4f}  "
          f"→ OOS {obs_gap:+.4f}  (Δ = {obs_gap - in_row['observed_gap_pp']:+.4f})")
    print(f"  CI                  : in-sample [{in_row['bootstrap_ci_lo_pp']:+.4f}, "
          f"{in_row['bootstrap_ci_hi_pp']:+.4f}]  → OOS [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  CI excludes 0       : in-sample {bool(in_row['ci_excludes_zero'])}  "
          f"→ OOS {excludes_zero}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.hist(boot_gaps, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(obs_gap, color="crimson", lw=2,
                label=f"Observed gap ({obs_gap:+.2f} pp)")
    ax.axvline(ci_lo, color="grey", lw=1.5, ls="--",
                label=f"95% CI [{ci_lo:+.2f}, {ci_hi:+.2f}]")
    ax.axvline(ci_hi, color="grey", lw=1.5, ls="--")
    ax.axvline(0, color="black", lw=1, ls=":")
    # Overlay the in-sample observed gap for direct contrast
    ax.axvline(in_row["observed_gap_pp"], color="darkorange", lw=1.5, ls="-.",
                label=f"In-sample observed ({in_row['observed_gap_pp']:+.2f} pp)")
    ax.set_xlabel("Information-loss gap (pp): %|SHAP|_Filter − %|SHAP|_Ridge in filtered features")
    ax.set_ylabel("Bootstrap frequency")
    ax.set_title("H1 bootstrap distribution — OOS surrogate (Filter vs Ridge)")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    out = ROOT / "plots" / "h2_bootstrap_oos_ci.png"
    pdf_out = out.with_suffix(".pdf")
    plt.savefig(out, bbox_inches="tight", dpi=120)
    plt.savefig(pdf_out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.relative_to(ROOT)}")
    print(f"  Saved: {pdf_out.relative_to(ROOT)}")

    return out_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("OOS-SHAP REFRESH — Exp 1 + Notebook 04 H2 bootstrap")
    print("=" * 70)
    print()

    exp1_merged = refresh_exp1_shap()
    h2_oos      = refresh_h2_bootstrap()

    print()
    print("Saved:")
    print("  data/fourier_ablation_shap_oos.csv")
    print("  data/interpretability_h2_bootstrap_oos.csv")
    print("  plots/fourier_ablation_shap_oos.png")
    print("  plots/h2_bootstrap_oos_ci.png")


if __name__ == "__main__":
    main()
