"""
Experiment 6 — GDP / population interpolation sensitivity (Note #11).

Reruns Stage-1 fitting and H1 coefficient stability + rolling-origin SARIMAX
under two upsampling schemes for GDP (quarterly) and population (annual):
  - "linear"  — the current scheme (interpolate("linear")), already in train.csv
  - "ffill"   — forward-fill from the last observed quarterly / annual value

Outputs:
  - data/interpolation_sensitivity_cv.csv     — per-feature CV stability
  - data/interpolation_sensitivity_mape.csv   — MAPE × horizon × strategy × scheme
  - data/interpolation_sensitivity_summary.csv — economic-group median CV summary
"""
from __future__ import annotations

import json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
HORIZONS = [1, 3, 7, 30, 90, 180]
MAX_H = max(HORIZONS)

# ---------------------------------------------------------------------------
# 1. Build the alternative (ffill) GDP and population series
# ---------------------------------------------------------------------------
def build_ffill_columns() -> pd.DataFrame:
    """Daily-indexed GDP and population using forward-fill from raw anchors."""
    # GDP — quarterly (CPMNACSCAB1GQNL)
    gdp_raw = pd.read_csv(ROOT / "raw_data" / "eurostatGDP.csv",
                          parse_dates=["observation_date"])
    gdp_raw = gdp_raw.rename(columns={"observation_date": "date",
                                      "CPMNACSCAB1GQNL": "gdp_mln_eur"})
    gdp_raw = gdp_raw[(gdp_raw["date"] >= "2009-01-01") &
                      (gdp_raw["date"] <= "2025-12-31")].copy()

    # Population — annual (Year, Population in millions); rescale to match
    # the existing modelling_dataset_daily.csv population units (thousands)
    pop_raw = pd.read_csv(ROOT / "raw_data" / "Population.csv",
                          sep=";", decimal=",")
    pop_raw["date"] = pd.to_datetime(pop_raw["Year"].astype(str) + "-01-01")
    pop_raw = pop_raw[(pop_raw["date"] >= "2009-01-01") &
                      (pop_raw["date"] <= "2025-12-31")].copy()

    # Detect scaling by comparing first anchor with the current daily dataset
    daily = pd.read_csv(ROOT / "data" / "modelling_dataset_daily.csv",
                        parse_dates=["date"])
    anchor_2009 = daily.loc[daily["date"] == "2009-01-01",
                             ["gdp_mln_eur", "population"]].iloc[0]
    gdp_ratio = anchor_2009["gdp_mln_eur"] / gdp_raw.iloc[0]["gdp_mln_eur"]
    pop_ratio = anchor_2009["population"]  / pop_raw.iloc[0]["Population"]
    print(f"  GDP scale match : daily/raw = {gdp_ratio:.4f}")
    print(f"  Pop scale match : daily/raw = {pop_ratio:.4f}")
    gdp_raw["gdp_mln_eur"] = gdp_raw["gdp_mln_eur"] * gdp_ratio
    pop_raw["population"]  = pop_raw["Population"]  * pop_ratio

    # Forward-fill to daily resolution
    full_idx = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    gdp_daily = (gdp_raw.set_index("date")["gdp_mln_eur"]
                 .reindex(full_idx).ffill())
    pop_daily = (pop_raw.set_index("date")["population"]
                 .reindex(full_idx).ffill())

    out = pd.DataFrame({"date": full_idx,
                        "gdp_mln_eur_ffill": gdp_daily.values,
                        "population_ffill": pop_daily.values})
    return out


# ---------------------------------------------------------------------------
# 2. Replace GDP and population columns in train/val/test under each scheme
# ---------------------------------------------------------------------------
def apply_scheme(scheme: str, ffill_df: pd.DataFrame) -> dict:
    """Return scheme-specific train / val / test dataframes plus split sizes."""
    train = pd.read_csv(ROOT / "data" / "train.csv", parse_dates=["date"])
    val   = pd.read_csv(ROOT / "data" / "val.csv",   parse_dates=["date"])
    test  = pd.read_csv(ROOT / "data" / "test.csv",  parse_dates=["date"])

    if scheme == "ffill":
        merged = train[["date"]].merge(ffill_df, on="date", how="left")
        train["gdp_mln_eur"] = merged["gdp_mln_eur_ffill"].values
        train["population"]  = merged["population_ffill"].values

        merged_v = val[["date"]].merge(ffill_df, on="date", how="left")
        val["gdp_mln_eur"] = merged_v["gdp_mln_eur_ffill"].values
        val["population"]  = merged_v["population_ffill"].values

        merged_t = test[["date"]].merge(ffill_df, on="date", how="left")
        test["gdp_mln_eur"] = merged_t["gdp_mln_eur_ffill"].values
        test["population"]  = merged_t["population_ffill"].values

    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# 3. Build per-strategy feature matrices (scaled cont. + raw calendar)
# ---------------------------------------------------------------------------
CLIMATE = ["temp_c", "wind_ms", "precip_mm", "sunshine_h",
           "global_rad", "pressure_hpa", "humidity_pct", "cloudiness", "nao"]
STRUCTURAL = ["price_eur_kwh", "gdp_mln_eur", "population",
              "wind_energy_gwh", "solar_energy_gwh"]
CONTINUOUS = CLIMATE + STRUCTURAL
RAW_CAL    = ["day_of_week", "month", "is_weekend", "is_holiday"]
FOURIER    = ["sin1_ann", "cos1_ann", "sin2_ann", "cos2_ann"]
CALENDAR   = RAW_CAL + FOURIER

ECONOMIC_FEATS = ["price_eur_kwh", "gdp_mln_eur", "population",
                  "wind_energy_gwh", "solar_energy_gwh", "month"]


def add_fourier(df: pd.DataFrame) -> pd.DataFrame:
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


def build_features(splits: dict) -> dict:
    train = add_fourier(splits["train"])
    val   = add_fourier(splits["val"])
    test  = add_fourier(splits["test"])

    sx = StandardScaler()
    X_tr_c = sx.fit_transform(train[CONTINUOUS])
    X_va_c = sx.transform(val[CONTINUOUS])
    X_te_c = sx.transform(test[CONTINUOUS])

    def stack(cont_arr, df):
        cont = pd.DataFrame(cont_arr, columns=CONTINUOUS, index=df.index)
        cal  = df[CALENDAR].reset_index(drop=True)
        return pd.concat([cont.reset_index(drop=True), cal], axis=1)

    X_tr = stack(X_tr_c, train)
    X_va = stack(X_va_c, val)
    X_te = stack(X_te_c, test)

    y_tr = train["demand_MW"].values
    y_va = val["demand_MW"].values
    y_te = test["demand_MW"].values

    # Strategy feature sets
    retained = correlation_filter(train[CONTINUOUS], y_tr, threshold=0.80)
    filter_cols = retained + CALENDAR

    # Lasso selection — use saved hyperparams; refit to identify zero coefs
    with open(ROOT / "data" / "strategy_elasticnet_params.json") as f:
        enet_params = json.load(f)
    enet = ElasticNet(alpha=enet_params["alpha"],
                      l1_ratio=enet_params["l1_ratio"],
                      max_iter=5000, random_state=SEED).fit(X_tr.values, y_tr)
    enet_cols = [c for c, k in zip(X_tr.columns, enet.coef_) if k != 0]
    # Force calendar features in
    lasso_cols = sorted(set(enet_cols) | set(CALENDAR),
                        key=lambda c: list(X_tr.columns).index(c))

    return {
        "train": {"X": X_tr, "y": y_tr},
        "val":   {"X": X_va, "y": y_va},
        "test":  {"X": X_te, "y": y_te},
        "filter_cols": filter_cols,
        "lasso_cols":  lasso_cols,
    }


# ---------------------------------------------------------------------------
# 4. H1: 10-fold coefficient-of-variation stability for each strategy
# ---------------------------------------------------------------------------
def coef_cv(X: pd.DataFrame, y: np.ndarray, model_factory) -> pd.Series:
    """Return per-feature coefficient of variation across 10 rolling folds."""
    tscv = TimeSeriesSplit(n_splits=10)
    coefs = []
    for tr_idx, _ in tscv.split(X):
        m = model_factory()
        m.fit(X.iloc[tr_idx], y[tr_idx])
        coefs.append(pd.Series(m.coef_, index=X.columns))
    df = pd.DataFrame(coefs)
    mean = df.mean()
    std  = df.std()
    cv = (std / mean.abs()).clip(upper=5.0)  # cap at 5 (matches notebook 04)
    cv.name = None
    return cv


def h1_stability(features: dict) -> pd.DataFrame:
    X = features["train"]["X"]
    y = features["train"]["y"]

    # OLS-All
    ols_cv   = coef_cv(X, y, lambda: LinearRegression())
    # Filter
    Xf = X[features["filter_cols"]]
    flt_cv   = coef_cv(Xf, y, lambda: LinearRegression())
    # Ridge
    with open(ROOT / "data" / "strategy_ridge_params.json") as f:
        rp = json.load(f)
    ridge_cv = coef_cv(X, y, lambda: Ridge(alpha=rp["lambda"]))
    # Lasso (use saved hyperparams)
    with open(ROOT / "data" / "strategy_elasticnet_params.json") as f:
        ep = json.load(f)
    Xl = X[features["lasso_cols"]]
    lasso_cv = coef_cv(Xl, y, lambda: ElasticNet(alpha=ep["alpha"],
                                                  l1_ratio=ep["l1_ratio"],
                                                  max_iter=5000,
                                                  random_state=SEED))

    out = pd.DataFrame({"OLS-All": ols_cv, "Filter": flt_cv,
                         "Ridge": ridge_cv, "Lasso": lasso_cv})
    return out


# ---------------------------------------------------------------------------
# 5. SARIMAX rolling-origin MAPE per strategy (oracle exog)
# ---------------------------------------------------------------------------
def mape(y_true, y_pred):
    return 100 * np.mean(np.abs((y_true - y_pred) / y_true))


def fit_two_stage(X_tr, y_tr, sarima_order, seasonal_order, stage1):
    s = stage1.fit(X_tr, y_tr)
    resid = y_tr - s.predict(X_tr)
    sx = SARIMAX(resid, order=sarima_order, seasonal_order=seasonal_order,
                 enforce_stationarity=False, enforce_invertibility=False)
    return s, sx.fit(disp=False)


def rolling_eval(stage1, sarima_r, X_test_arr, y_test, eval_step=30):
    # Use Stage-1 model itself for exog contribution (oracle: true future X)
    exog_te = stage1.predict(X_test_arr)
    # SARIMA needs full training residuals already inside sarima_r
    # We'll compute test residuals on the fly: y_test - exog_te
    resid_te = y_test - exog_te

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
            results[h]["pred"].append(fc[h-1] + exog_te[t])
            results[h]["actual"].append(y_test[t])

    return {h: mape(np.array(r["actual"]), np.array(r["pred"]))
            for h, r in results.items() if r["actual"]}


def sarimax_mape(features: dict) -> dict:
    SEASONAL = (0, 1, 1, 7)
    BEST_ORDER = (2, 0, 1)  # current global choice

    out = {}
    strategies = [
        ("OLS-All", features["train"]["X"], LinearRegression()),
        ("Filter",  features["train"]["X"][features["filter_cols"]],
                    LinearRegression()),
        ("Ridge",   features["train"]["X"],
                    Ridge(alpha=json.loads((ROOT / "data" /
                                            "strategy_ridge_params.json").read_text())["lambda"])),
        ("Lasso",   features["train"]["X"][features["lasso_cols"]],
                    ElasticNet(alpha=json.loads((ROOT / "data" /
                                                 "strategy_elasticnet_params.json").read_text())["alpha"],
                               l1_ratio=json.loads((ROOT / "data" /
                                                    "strategy_elasticnet_params.json").read_text())["l1_ratio"],
                               max_iter=5000, random_state=SEED)),
    ]
    y_tr = features["train"]["y"]
    y_te = features["test"]["y"]
    X_te_full = features["test"]["X"]

    for name, X_tr, model in strategies:
        t0 = time.time()
        if name == "Filter":
            X_te = X_te_full[features["filter_cols"]]
        elif name == "Lasso":
            X_te = X_te_full[features["lasso_cols"]]
        else:
            X_te = X_te_full
        stage1, sarima_r = fit_two_stage(X_tr, y_tr, BEST_ORDER, SEASONAL, model)
        mape_per_h = rolling_eval(stage1, sarima_r, X_te, y_te)
        out[name] = mape_per_h
        print(f"    {name:<8}  {time.time()-t0:5.1f}s  "
              + "  ".join(f"h={h}: {v:.2f}%" for h, v in mape_per_h.items()))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Building ffill GDP / population series...")
    ffill_df = build_ffill_columns()

    cv_long_rows  = []
    mape_long_rows = []

    for scheme in ["linear", "ffill"]:
        print(f"\n=== Scheme: {scheme} ===")
        splits = apply_scheme(scheme, ffill_df)
        feats  = build_features(splits)
        print(f"  Filter cols ({len(feats['filter_cols'])}): {feats['filter_cols']}")
        print(f"  Lasso cols  ({len(feats['lasso_cols'])}):  {feats['lasso_cols']}")

        # H1: coefficient stability
        print("  Computing H1 CV stability...")
        cv_df = h1_stability(feats)
        for strat in cv_df.columns:
            for feat, v in cv_df[strat].dropna().items():
                cv_long_rows.append({"scheme": scheme, "strategy": strat,
                                     "feature": feat, "cv": float(v)})

        # SARIMAX MAPE
        print("  Running SARIMAX rolling-origin MAPE...")
        mape_dict = sarimax_mape(feats)
        for strat, h_map in mape_dict.items():
            for h, v in h_map.items():
                mape_long_rows.append({"scheme": scheme, "strategy": strat,
                                       "horizon": h, "mape": float(v)})

    cv_df_long   = pd.DataFrame(cv_long_rows)
    mape_df_long = pd.DataFrame(mape_long_rows)

    cv_df_long.to_csv(ROOT / "data" / "interpolation_sensitivity_cv.csv",
                       index=False)
    mape_df_long.to_csv(ROOT / "data" / "interpolation_sensitivity_mape.csv",
                        index=False)

    # Summary: economic-group median CV by scheme × strategy
    econ_cv = cv_df_long[cv_df_long["feature"].isin(ECONOMIC_FEATS)]
    summary = (econ_cv.groupby(["scheme", "strategy"])["cv"]
               .median().round(3)
               .unstack("scheme"))
    summary["delta_ffill_minus_linear"] = (summary["ffill"] - summary["linear"]).round(3)
    summary.to_csv(ROOT / "data" / "interpolation_sensitivity_summary.csv")

    print("\nEconomic-group MEDIAN CV by scheme × strategy:")
    print(summary)

    # MAPE summary
    mape_pivot = (mape_df_long
                  .pivot_table(index=["strategy", "horizon"],
                               columns="scheme", values="mape")
                  .round(3))
    mape_pivot["delta_ffill_minus_linear"] = (
        mape_pivot["ffill"] - mape_pivot["linear"]).round(3)
    print("\nMAPE by strategy × horizon × scheme:")
    print(mape_pivot)

    print("\nSaved:")
    print("  data/interpolation_sensitivity_cv.csv")
    print("  data/interpolation_sensitivity_mape.csv")
    print("  data/interpolation_sensitivity_summary.csv")


if __name__ == "__main__":
    main()
