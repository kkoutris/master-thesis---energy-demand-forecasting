"""
Experiment 2 — Realistic (non-oracle) forecast (Note #4).

The thesis's headline MAPE table assumes the true future exogenous matrix
X[t0+h] is known at forecast origin t0. This script reruns the rolling-origin
SARIMAX evaluation under a deployability-aware exog construction:

  - Calendar + Fourier (8 features) : true values (deterministic).
  - Climate (9 features: temp_c, wind_ms, precip_mm, sunshine_h, global_rad,
    pressure_hpa, humidity_pct, cloudiness, nao):
      h <= 7        : persistence (repeat the value at t0-1).
      h >= 30       : training day-of-year climatology mean.
      h in (7, 30)  : linear blend (weight w = (h - 7) / 23).
  - Economic (5 features: price_eur_kwh, gdp_mln_eur, population,
    wind_energy_gwh, solar_energy_gwh) : forward-fill from t0-1.

All five strategies (OLS-All, Filter, PCA, Ridge, Lasso) are rerun so the
output table can be compared row-for-row against data/sarimax_results.csv.

SARIMA state advancement at each rolling origin uses the *oracle* residuals
up to t0 (y_te[:t0] - stage1.predict(X_te[:t0])): at observed past times we
do see the true X, so the realistic restriction only applies to the
forecast window itself.

Interpretation caveats (see experiments/realistic_protocol_sensitivity.py
for the corresponding sensitivity variants):
  - This protocol is a CONSERVATIVE LOWER BOUND on operational accuracy:
    a real operator would use NWP weather forecasts (KNMI/ECMWF), which are
    skillful to ~5 days and far better than persistence at h <= 5. Oracle
    and realistic therefore bracket deployable performance from above/below.
  - Economic forward-fill inherits ~1 quarter of look-ahead from the daily
    GDP/population series (linearly interpolated from quarterly anchors in
    01_feature_engineering); publication-lag-safe values are tested in the
    sensitivity script (variant V4) and matter little.
  - Fixed 2026-06: exog rows were previously built for day origin+h-1 while
    the eval target was origin+h, feeding the previous day's calendar
    features into every forecast. That off-by-one inflated the apparent
    oracle->realistic penalty to +1.4..2.9 pp; the true penalty is ~+0.3 pp
    at h <= 3 and +0.3..0.6 pp at h >= 90.

Outputs:
  - data/sarimax_results_realistic.csv  schema mirrors sarimax_results.csv
  - data/realistic_vs_oracle_delta.csv  Δ MAPE per strategy × horizon
  - plots/realistic_vs_oracle_mape.png  MAPE × horizon, oracle vs realistic
  - plots/realistic_vs_oracle_mape.pdf  vector copy of the above
"""
from __future__ import annotations

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
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
HORIZONS = [1, 3, 7, 30, 90, 180]
MAX_H = max(HORIZONS)
SEASONAL_ORDER = (0, 1, 1, 7)
BEST_ORDER = (2, 0, 1)
PCA_N_COMPONENTS = 9
PERSIST_H = 7
CLIM_H = 30

CLIMATE = ["temp_c", "wind_ms", "precip_mm", "sunshine_h",
           "global_rad", "pressure_hpa", "humidity_pct", "cloudiness", "nao"]
STRUCTURAL = ["price_eur_kwh", "gdp_mln_eur", "population",
              "wind_energy_gwh", "solar_energy_gwh"]
CONTINUOUS = CLIMATE + STRUCTURAL
RAW_CAL = ["day_of_week", "month", "is_weekend", "is_holiday"]
FOURIER = ["sin1_ann", "cos1_ann", "sin2_ann", "cos2_ann"]
CALENDAR = RAW_CAL + FOURIER


def add_fourier(df: pd.DataFrame) -> pd.DataFrame:
    doy = df["date"].dt.dayofyear
    out = df.copy()
    out["sin1_ann"] = np.sin(2 * np.pi * doy / 365.25)
    out["cos1_ann"] = np.cos(2 * np.pi * doy / 365.25)
    out["sin2_ann"] = np.sin(4 * np.pi * doy / 365.25)
    out["cos2_ann"] = np.cos(4 * np.pi * doy / 365.25)
    return out


def correlation_filter(X_df: pd.DataFrame, y: np.ndarray, threshold: float = 0.80):
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


def nrmse(y_true, y_pred):
    return 100 * np.sqrt(np.mean((y_true - y_pred) ** 2)) / np.mean(y_true)


def build_splits():
    """Load train/val/test, add Fourier, scale CONTINUOUS, fit PCA on train."""
    train = add_fourier(pd.read_csv(ROOT / "data" / "train.csv", parse_dates=["date"]))
    val = add_fourier(pd.read_csv(ROOT / "data" / "val.csv", parse_dates=["date"]))
    test = add_fourier(pd.read_csv(ROOT / "data" / "test.csv", parse_dates=["date"]))

    sx = StandardScaler()
    X_tr_c = sx.fit_transform(train[CONTINUOUS])
    X_va_c = sx.transform(val[CONTINUOUS])
    X_te_c = sx.transform(test[CONTINUOUS])

    def stack(arr, df):
        cont = pd.DataFrame(arr, columns=CONTINUOUS, index=df.index)
        cal = df[CALENDAR].reset_index(drop=True)
        return pd.concat([cont.reset_index(drop=True), cal], axis=1)

    X_tr = stack(X_tr_c, train)
    X_va = stack(X_va_c, val)
    X_te = stack(X_te_c, test)

    y_tr = train["demand_MW"].values
    y_te = test["demand_MW"].values

    # Strategy column lists
    retained = correlation_filter(train[CONTINUOUS], y_tr, threshold=0.80)
    filter_cols = retained + CALENDAR

    with open(ROOT / "data" / "strategy_elasticnet_params.json") as f:
        ep = json.load(f)
    enet = ElasticNet(alpha=ep["alpha"], l1_ratio=ep["l1_ratio"],
                      max_iter=5000, random_state=SEED).fit(X_tr.values, y_tr)
    nonzero = [c for c, k in zip(X_tr.columns, enet.coef_) if k != 0]
    lasso_cols = sorted(set(nonzero) | set(CALENDAR),
                        key=lambda c: list(X_tr.columns).index(c))

    # PCA strategy — fit on train scaled CONTINUOUS, keep 9 components.
    pca = PCA(n_components=PCA_N_COMPONENTS, random_state=SEED).fit(X_tr[CONTINUOUS].values)
    pc_cols = [f"PC{i+1}" for i in range(PCA_N_COMPONENTS)]

    def to_pca(X_wide: pd.DataFrame) -> pd.DataFrame:
        pcs = pca.transform(X_wide[CONTINUOUS].values)
        pc_df = pd.DataFrame(pcs, columns=pc_cols, index=X_wide.index)
        return pd.concat([pc_df, X_wide[CALENDAR].reset_index(drop=True)], axis=1)

    return {
        "train": {"X": X_tr, "y": y_tr, "dates": train["date"]},
        "val":   {"X": X_va, "dates": val["date"]},
        "test":  {"X": X_te, "y": y_te, "dates": test["date"]},
        "filter_cols": filter_cols,
        "lasso_cols": lasso_cols,
        "to_pca": to_pca,
        "pc_cols": pc_cols,
    }


def compute_climatology_scaled(train_X: pd.DataFrame, train_dates: pd.Series,
                                cols: list) -> dict:
    """Day-of-year mean (scaled units) for each climate feature on training data."""
    doy = train_dates.dt.dayofyear.values
    out = {}
    for c in cols:
        s = pd.Series(train_X[c].values, index=doy)
        out[c] = s.groupby(level=0).mean().to_dict()
    return out


def make_base_realistic_builder(X_te_wide: pd.DataFrame,
                                 val_last_row: pd.Series,
                                 climatology: dict,
                                 test_dates: pd.Series):
    """Closure returning a DataFrame of realistic exog (scaled cont + true cal)."""
    test_dates = test_dates.reset_index(drop=True)
    X_te_wide = X_te_wide.reset_index(drop=True)

    def builder(origin: int) -> pd.DataFrame:
        # Last observed full row (scaled cont + cal); if origin == 0, use last val row.
        last = X_te_wide.iloc[origin - 1] if origin > 0 else val_last_row
        rows = []
        for h_step in range(1, MAX_H + 1):
            # Row h_step is the exog for forecast target day origin + h_step
            # (rolling_eval pairs exog_real[h-1] with y_te[origin + h]).
            t = origin + h_step
            # True calendar at the future day; start from the true wide row
            # and override climate + economic columns with the realistic forecast.
            row = X_te_wide.iloc[t].copy()
            doy_t = int(test_dates.iloc[t].dayofyear)
            for c in CLIMATE:
                persist = last[c]
                clim = climatology[c].get(doy_t, persist)
                if h_step <= PERSIST_H:
                    row[c] = persist
                elif h_step >= CLIM_H:
                    row[c] = clim
                else:
                    w = (h_step - PERSIST_H) / (CLIM_H - PERSIST_H)
                    row[c] = (1 - w) * persist + w * clim
            for c in STRUCTURAL:
                row[c] = last[c]
            rows.append(row)
        return pd.DataFrame(rows, columns=X_te_wide.columns).reset_index(drop=True)

    return builder


def fit_two_stage(X_tr, y_tr, stage1_model):
    s = stage1_model.fit(X_tr, y_tr)
    resid = y_tr - s.predict(X_tr)
    sx = SARIMAX(resid, order=BEST_ORDER, seasonal_order=SEASONAL_ORDER,
                 enforce_stationarity=False, enforce_invertibility=False)
    return s, sx.fit(disp=False), resid


def rolling_eval_realistic(stage1, sarima_r, X_te_strategy, y_te,
                            realistic_builder_strategy, eval_step=15):
    exog_oracle = stage1.predict(X_te_strategy)
    resid_te = y_te - exog_oracle
    results = {h: {"pred": [], "actual": []} for h in HORIZONS}
    for origin in range(0, len(resid_te) - MAX_H, eval_step):
        if origin == 0:
            res_ext = sarima_r
        else:
            res_ext = sarima_r.append(endog=resid_te[:origin], refit=False)
        sarima_fc = res_ext.forecast(steps=MAX_H)

        X_real = realistic_builder_strategy(origin)
        exog_real = stage1.predict(X_real)

        for h in HORIZONS:
            t = origin + h
            if t >= len(resid_te):
                continue
            results[h]["pred"].append(sarima_fc[h - 1] + exog_real[h - 1])
            results[h]["actual"].append(y_te[t])

    out = {}
    for h, r in results.items():
        if not r["actual"]:
            continue
        y_t = np.array(r["actual"])
        y_p = np.array(r["pred"])
        out[h] = {"mape": mape(y_t, y_p), "nrmse": nrmse(y_t, y_p),
                  "n": len(y_t)}
    return out


def run_strategies(splits):
    with open(ROOT / "data" / "strategy_ridge_params.json") as f:
        rp = json.load(f)
    with open(ROOT / "data" / "strategy_elasticnet_params.json") as f:
        ep = json.load(f)

    X_tr = splits["train"]["X"]
    y_tr = splits["train"]["y"]
    X_te = splits["test"]["X"]
    y_te = splits["test"]["y"]

    # Climatology in scaled space on training data for the climate columns.
    climatology = compute_climatology_scaled(X_tr, splits["train"]["dates"], CLIMATE)
    val_last_row = splits["val"]["X"].iloc[-1]
    base_builder = make_base_realistic_builder(
        X_te, val_last_row, climatology, splits["test"]["dates"]
    )

    filter_cols = splits["filter_cols"]
    lasso_cols = splits["lasso_cols"]
    to_pca = splits["to_pca"]
    pc_cols = splits["pc_cols"]

    # Strategy training matrices and per-strategy realistic builders.
    strategies = []

    strategies.append((
        "OLS-All", X_tr, X_te, lambda: LinearRegression(),
        lambda origin: base_builder(origin),
    ))
    strategies.append((
        "Filter", X_tr[filter_cols], X_te[filter_cols], lambda: LinearRegression(),
        lambda origin: base_builder(origin)[filter_cols],
    ))
    strategies.append((
        "PCA", to_pca(X_tr), to_pca(X_te), lambda: LinearRegression(),
        lambda origin: to_pca(base_builder(origin)),
    ))
    strategies.append((
        "Ridge", X_tr, X_te, lambda: Ridge(alpha=rp["lambda"]),
        lambda origin: base_builder(origin),
    ))
    strategies.append((
        "Lasso", X_tr[lasso_cols], X_te[lasso_cols],
        lambda: ElasticNet(alpha=ep["alpha"], l1_ratio=ep["l1_ratio"],
                            max_iter=5000, random_state=SEED),
        lambda origin: base_builder(origin)[lasso_cols],
    ))

    rows = []
    for name, X_tr_s, X_te_s, mfact, realistic_b in strategies:
        t0 = time.time()
        stage1, sarima_r, _ = fit_two_stage(X_tr_s, y_tr, mfact())
        metrics = rolling_eval_realistic(stage1, sarima_r, X_te_s, y_te, realistic_b)
        msg = "  ".join(f"h={h}: {metrics[h]['mape']:.2f}%" for h in HORIZONS if h in metrics)
        print(f"    {name:<8}  {time.time()-t0:5.1f}s  {msg}")
        for h, m in metrics.items():
            rows.append({"strategy": name, "horizon": h,
                          "mape": m["mape"], "nrmse": m["nrmse"], "n": m["n"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def load_oracle_results() -> pd.DataFrame:
    """Parse sarimax_results.csv into (strategy, horizon, mape, nrmse) form."""
    df = pd.read_csv(ROOT / "data" / "sarimax_results.csv")
    df["strategy"] = df["model"].str.extract(r"\((.*?)\)$")
    df["strategy"] = df["strategy"].str.replace(r"\s*\(.*\)$", "", regex=True)
    df = df.rename(columns={"horizon_days": "horizon",
                             "mape_pct": "mape",
                             "nrmse_pct": "nrmse"})
    return df[["strategy", "horizon", "mape", "nrmse"]]


def main():
    print("=" * 70)
    print("EXPERIMENT 2 — Realistic (non-oracle) forecast")
    print("=" * 70)
    print(f"Persistence h<= {PERSIST_H}; climatology h>= {CLIM_H}; "
          f"blend in between (linear weight).")
    print()

    print("Loading splits, scaler, PCA, strategy column sets...")
    splits = build_splits()
    print(f"  Filter cols ({len(splits['filter_cols'])}): {splits['filter_cols']}")
    print(f"  Lasso cols  ({len(splits['lasso_cols'])}):  {splits['lasso_cols']}")
    print(f"  PCA components: {splits['pc_cols']}")

    print("\nRunning rolling-origin SARIMAX with realistic exog...")
    realistic_df = run_strategies(splits)

    # Save in the same schema as sarimax_results.csv for parity.
    realistic_df_out = realistic_df.copy()
    realistic_df_out["model"] = realistic_df_out["strategy"].apply(
        lambda s: f"SARIMAX-realistic ({s})"
    )
    realistic_df_out = realistic_df_out.rename(columns={
        "horizon": "horizon_days",
        "mape": "mape_pct",
        "nrmse": "nrmse_pct",
        "n": "n_eval_points",
    })
    realistic_df_out = realistic_df_out[
        ["model", "horizon_days", "mape_pct", "nrmse_pct", "n_eval_points"]
    ]
    out_path = ROOT / "data" / "sarimax_results_realistic.csv"
    realistic_df_out.to_csv(out_path, index=False)

    # Compare against oracle.
    oracle = load_oracle_results()
    merged = realistic_df.merge(oracle, on=["strategy", "horizon"],
                                 suffixes=("_real", "_oracle"))
    merged["delta_mape"] = merged["mape_real"] - merged["mape_oracle"]
    merged["delta_nrmse"] = merged["nrmse_real"] - merged["nrmse_oracle"]
    delta_path = ROOT / "data" / "realistic_vs_oracle_delta.csv"
    merged[["strategy", "horizon", "mape_oracle", "mape_real",
            "delta_mape", "nrmse_oracle", "nrmse_real", "delta_nrmse"]
           ].to_csv(delta_path, index=False)

    print()
    print("=" * 70)
    print("MAPE — realistic vs oracle (Δ = realistic − oracle, pp)")
    print("=" * 70)
    piv = merged.pivot_table(index="strategy", columns="horizon",
                              values="delta_mape").round(2)
    print(piv)

    print()
    print("Headline MAPE under realistic exog:")
    head = merged.pivot_table(index="strategy", columns="horizon",
                               values="mape_real").round(2)
    print(head)

    # ---------- Plot: MAPE × horizon, oracle vs realistic ----------
    print("\nGenerating plot...")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"OLS-All": "#888888", "Filter": "#4477aa", "PCA": "#aa3377",
              "Ridge": "#117733", "Lasso": "#cc6677"}
    for strat in ["OLS-All", "Filter", "PCA", "Ridge", "Lasso"]:
        if strat not in colors:
            continue
        o = oracle[oracle["strategy"] == strat].sort_values("horizon")
        r = realistic_df[realistic_df["strategy"] == strat].sort_values("horizon")
        ax.plot(o["horizon"], o["mape"], marker="o", linestyle="-",
                color=colors[strat], alpha=0.9, label=f"{strat} (oracle)")
        ax.plot(r["horizon"], r["mape"], marker="s", linestyle="--",
                color=colors[strat], alpha=0.9, label=f"{strat} (realistic)")
    ax.set_xscale("log")
    ax.set_xticks(HORIZONS); ax.set_xticklabels(HORIZONS)
    ax.set_xlabel("Forecast horizon (days, log scale)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("MAPE × Horizon — Oracle vs Realistic Exogenous Construction")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = ROOT / "plots" / "realistic_vs_oracle_mape.png"
    pdf_path = plot_path.with_suffix(".pdf")
    plt.savefig(plot_path, bbox_inches="tight", dpi=120)
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print("\nSaved:")
    print(f"  {out_path.relative_to(ROOT)}")
    print(f"  {delta_path.relative_to(ROOT)}")
    print(f"  {plot_path.relative_to(ROOT)}")
    print(f"  {pdf_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
