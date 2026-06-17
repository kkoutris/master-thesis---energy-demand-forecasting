"""
Experiment 10 — Realistic-exog protocol sensitivity (Review points 4-7).

The headline realistic protocol (experiments/realistic_forecast.py) uses:
persistence for climate at h<=7, train day-of-year climatology at h>=30, a
linear blend in (7,30), and forward-fill for economic predictors. Four
refinements are tested against it, all strategies, identical rolling-origin
protocol (origins every 15 days, oracle SARIMA state advancement):

  V0  baseline      — current protocol verbatim. Must reproduce
                      data/sarimax_results_realistic.csv (sanity gate).
  V1  nwp_proxy     — oracle climate for h <= 5 (proxy for skillful NWP
                      weather forecasts), V0 rules beyond. Together with V0
                      this brackets operationally achievable accuracy:
                      V0 = NWP-denied lower bound, V1 ~ NWP-equipped.
  V2  ar_decay      — principled blend for ALL h: anomaly persistence decays
                      at the predictor's own train-period anomaly
                      autocorrelation: x(t0+h) = clim(doy) +
                      rho_c(h) * (x(t0-1) - clim(doy(t0-1))), rho clipped >=0.
  V3  trend_clim    — V0 but climatology corrected by a per-predictor linear
                      anomaly trend fit on TRAIN only and extrapolated to the
                      test date (addresses the +0.86 degC warm test period).
  V4  pub_lag       — V0 but economic predictors taken at publication-lag-
                      safe offsets: gdp/population t0-90d, wind/solar energy
                      t0-30d, price t0-1d (the daily GDP series is linearly
                      interpolated from quarterly data upstream, which embeds
                      ~1 quarter of look-ahead; this variant removes it).

Outputs:
  - data/realistic_protocol_sensitivity.csv   MAPE x variant x strategy x
                                              horizon, deltas vs V0 & oracle
  - plots/realistic_protocol_sensitivity.png
"""
from __future__ import annotations

import json, time, warnings
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
EVAL_STEP = 15
SEASONAL_ORDER = (0, 1, 1, 7)
BEST_ORDER = (2, 0, 1)
PCA_N_COMPONENTS = 9
PERSIST_H = 7
CLIM_H = 30
NWP_H = 5                      # V1: oracle climate up to this horizon
ECON_LAGS = {"gdp_mln_eur": 90, "population": 90,
             "wind_energy_gwh": 30, "solar_energy_gwh": 30,
             "price_eur_kwh": 1}   # V4 publication-lag offsets (days)

CLIMATE = ["temp_c", "wind_ms", "precip_mm", "sunshine_h",
           "global_rad", "pressure_hpa", "humidity_pct", "cloudiness", "nao"]
STRUCTURAL = ["price_eur_kwh", "gdp_mln_eur", "population",
              "wind_energy_gwh", "solar_energy_gwh"]
CONTINUOUS = CLIMATE + STRUCTURAL
RAW_CAL = ["day_of_week", "month", "is_weekend", "is_holiday"]
FOURIER = ["sin1_ann", "cos1_ann", "sin2_ann", "cos2_ann"]
CALENDAR = RAW_CAL + FOURIER
VARIANTS = ["V0_baseline", "V1_nwp_proxy", "V2_ar_decay",
            "V3_trend_clim", "V4_pub_lag"]


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


def build_splits():
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

    retained = correlation_filter(train[CONTINUOUS], y_tr, threshold=0.80)
    filter_cols = retained + CALENDAR

    with open(ROOT / "data" / "strategy_elasticnet_params.json") as f:
        ep = json.load(f)
    enet = ElasticNet(alpha=ep["alpha"], l1_ratio=ep["l1_ratio"],
                      max_iter=5000, random_state=SEED).fit(X_tr.values, y_tr)
    nonzero = [c for c, k in zip(X_tr.columns, enet.coef_) if k != 0]
    lasso_cols = sorted(set(nonzero) | set(CALENDAR),
                        key=lambda c: list(X_tr.columns).index(c))

    pca = PCA(n_components=PCA_N_COMPONENTS, random_state=SEED).fit(X_tr[CONTINUOUS].values)
    pc_cols = [f"PC{i+1}" for i in range(PCA_N_COMPONENTS)]

    def to_pca(X_wide):
        pcs = pca.transform(X_wide[CONTINUOUS].values)
        pc_df = pd.DataFrame(pcs, columns=pc_cols, index=X_wide.index)
        return pd.concat([pc_df, X_wide[CALENDAR].reset_index(drop=True)], axis=1)

    return {
        "train": {"X": X_tr, "y": y_tr, "dates": train["date"].reset_index(drop=True)},
        "val":   {"X": X_va, "dates": val["date"].reset_index(drop=True)},
        "test":  {"X": X_te, "y": y_te, "dates": test["date"].reset_index(drop=True)},
        "filter_cols": filter_cols, "lasso_cols": lasso_cols, "to_pca": to_pca,
    }


def climatology_tables(splits):
    """Per-climate-predictor (scaled space, train only):
       - day-of-year mean,
       - linear anomaly trend (per day) for V3,
       - anomaly autocorrelation rho(h), h=1..MAX_H, clipped at 0, for V2."""
    X_tr = splits["train"]["X"]
    dates = splits["train"]["dates"]
    doy = dates.dt.dayofyear.values
    t_ord = dates.map(pd.Timestamp.toordinal).values.astype(float)

    clim, trend, rho = {}, {}, {}
    for c in CLIMATE:
        x = X_tr[c].values
        s = pd.Series(x, index=doy)
        clim[c] = s.groupby(level=0).mean().to_dict()
        anom = x - np.array([clim[c][d] for d in doy])
        # V3: linear anomaly trend on train (slope per day, centred on train)
        A = np.vstack([t_ord - t_ord.mean(), np.ones_like(t_ord)]).T
        slope, _ = np.linalg.lstsq(A, anom, rcond=None)[0]
        trend[c] = {"slope_per_day": float(slope), "t_center": float(t_ord.mean())}
        # V2: anomaly autocorrelation out to MAX_H
        a = anom - anom.mean()
        denom = float(a @ a)
        r = np.empty(MAX_H + 1)
        r[0] = 1.0
        for k in range(1, MAX_H + 1):
            r[k] = float(a[k:] @ a[:-k]) / denom
        rho[c] = np.clip(r, 0.0, None)
    return clim, trend, rho


def precompute_test_grids(splits, clim, trend):
    """clim_test[t, c] and trend-corrected clim for every test day (+ val tail
    lookups for origin 0)."""
    te_dates = splits["test"]["dates"]
    doy_te = te_dates.dt.dayofyear.values
    t_ord = te_dates.map(pd.Timestamp.toordinal).values.astype(float)

    clim_te = np.empty((len(te_dates), len(CLIMATE)))
    clim_te_trend = np.empty_like(clim_te)
    for j, c in enumerate(CLIMATE):
        base = np.array([clim[c].get(d, np.nan) for d in doy_te])
        # doy 366 fallback: nearest available day
        if np.isnan(base).any():
            fill = np.nanmean(list(clim[c].values()))
            base = np.where(np.isnan(base), fill, base)
        clim_te[:, j] = base
        tr = trend[c]
        clim_te_trend[:, j] = base + tr["slope_per_day"] * (t_ord - tr["t_center"])
    return clim_te, clim_te_trend


def build_variant_exog(variant, origin, splits, clim_te, clim_te_trend, rho,
                       X_hist, val_len):
    """Wide (MAX_H x features) realistic exog matrix for one origin/variant.
    X_hist = concat(val, test) wide matrix for lookback; val_len = len(val)."""
    X_te = splits["test"]["X"]
    # Row h-1 is the exog for forecast target day origin + h (the rolling
    # eval pairs exog_real[h-1] with y_te[origin + h]).
    base = X_te.iloc[origin + 1:origin + 1 + MAX_H].copy().reset_index(drop=True)
    h_steps = np.arange(1, MAX_H + 1)

    last = X_hist.iloc[val_len + origin - 1]      # t0-1 (val tail if origin=0)
    clim_seg = clim_te[origin + 1:origin + 1 + MAX_H]
    clim_seg_v3 = clim_te_trend[origin + 1:origin + 1 + MAX_H]
    # climatology at t0-1 (for V2 anomaly): test day origin-1, else last val day
    if origin > 0:
        clim_t0 = clim_te[origin - 1]
    else:
        val_doy = int(splits["val"]["dates"].iloc[-1].dayofyear)
        clim_t0 = np.array([_CLIM_LOOKUP[c].get(val_doy, clim_te[0][j])
                            for j, c in enumerate(CLIMATE)])

    persist = np.array([last[c] for c in CLIMATE])

    if variant == "V2_ar_decay":
        anom = persist - clim_t0
        rho_mat = np.stack([rho[c][h_steps] for c in CLIMATE], axis=1)
        clim_block = clim_seg + rho_mat * anom
    else:
        clim_base = clim_seg_v3 if variant == "V3_trend_clim" else clim_seg
        w = np.clip((h_steps - PERSIST_H) / (CLIM_H - PERSIST_H), 0.0, 1.0)
        clim_block = (1 - w)[:, None] * persist[None, :] + w[:, None] * clim_base

    for j, c in enumerate(CLIMATE):
        if variant == "V1_nwp_proxy":
            # oracle climate for h <= NWP_H, protocol values beyond
            vals = base[c].values.copy()
            vals[h_steps > NWP_H] = clim_block[h_steps > NWP_H, j]
            base[c] = vals
        else:
            base[c] = clim_block[:, j]

    if variant == "V4_pub_lag":
        for c in STRUCTURAL:
            lag = ECON_LAGS[c]
            idx = max(val_len + origin - lag, 0)
            base[c] = X_hist.iloc[idx][c]
    else:
        for c in STRUCTURAL:
            base[c] = last[c]
    return base


_CLIM_LOOKUP = {}   # filled in main(); used only for the origin==0 edge case


def fit_two_stage(X_tr, y_tr, stage1_model):
    s = stage1_model.fit(X_tr, y_tr)
    resid = y_tr - s.predict(X_tr)
    sx = SARIMAX(resid, order=BEST_ORDER, seasonal_order=SEASONAL_ORDER,
                 enforce_stationarity=False, enforce_invertibility=False)
    return s, sx.fit(disp=False)


def main():
    print("=" * 70)
    print("EXPERIMENT 10 — Realistic protocol sensitivity (V0..V4)")
    print("=" * 70)

    splits = build_splits()
    clim, trend, rho = climatology_tables(splits)
    global _CLIM_LOOKUP
    _CLIM_LOOKUP = clim
    clim_te, clim_te_trend = precompute_test_grids(splits, clim, trend)

    print("Trend slopes (scaled units / decade):")
    for c in CLIMATE:
        print(f"  {c:<14} {trend[c]['slope_per_day'] * 3652.5:+.3f}")
    print("Anomaly autocorrelation rho(h) [temp_c]: "
          + "  ".join(f"h={h}: {rho['temp_c'][h]:.2f}" for h in [1, 3, 5, 7, 14, 30]))

    X_te = splits["test"]["X"]
    y_te = splits["test"]["y"]
    y_tr = splits["train"]["y"]
    X_hist = pd.concat([splits["val"]["X"], X_te], ignore_index=True)
    val_len = len(splits["val"]["X"])

    with open(ROOT / "data" / "strategy_ridge_params.json") as f:
        rp = json.load(f)
    with open(ROOT / "data" / "strategy_elasticnet_params.json") as f:
        ep = json.load(f)
    to_pca = splits["to_pca"]
    fc, lc = splits["filter_cols"], splits["lasso_cols"]

    strategies = {
        "OLS-All": (splits["train"]["X"], X_te,
                    lambda X: X, lambda: LinearRegression()),
        "Filter":  (splits["train"]["X"][fc], X_te[fc],
                    lambda X: X[fc], lambda: LinearRegression()),
        "PCA":     (to_pca(splits["train"]["X"]), to_pca(X_te),
                    lambda X: to_pca(X), lambda: LinearRegression()),
        "Ridge":   (splits["train"]["X"], X_te,
                    lambda X: X, lambda: Ridge(alpha=rp["lambda"])),
        "Lasso":   (splits["train"]["X"][lc], X_te[lc],
                    lambda X: X[lc],
                    lambda: ElasticNet(alpha=ep["alpha"], l1_ratio=ep["l1_ratio"],
                                       max_iter=5000, random_state=SEED)),
    }

    origins = list(range(0, len(y_te) - MAX_H, EVAL_STEP))
    print(f"\n{len(origins)} origins, {len(VARIANTS)} variants, "
          f"{len(strategies)} strategies")

    # Pre-build variant exog matrices once per (variant, origin); shared by
    # all strategies (each strategy then selects/transforms columns).
    print("Pre-building variant exog matrices...")
    t0 = time.time()
    exog_cache = {(v, o): build_variant_exog(v, o, splits, clim_te,
                                             clim_te_trend, rho, X_hist, val_len)
                  for v in VARIANTS for o in origins}
    print(f"  done [{time.time()-t0:.1f}s]")

    rows = []
    for name, (X_tr_s, X_te_s, select, mfact) in strategies.items():
        t0 = time.time()
        stage1, sarima_r = fit_two_stage(X_tr_s, y_tr, mfact())
        exog_oracle = stage1.predict(X_te_s)
        resid_te = y_te - exog_oracle

        results = {v: {h: {"p": [], "a": []} for h in HORIZONS} for v in VARIANTS}
        for origin in origins:
            res_ext = sarima_r if origin == 0 else sarima_r.append(
                endog=resid_te[:origin], refit=False)
            sarima_fc = res_ext.forecast(steps=MAX_H)
            for v in VARIANTS:
                exog_real = stage1.predict(select(exog_cache[(v, origin)]))
                for h in HORIZONS:
                    t = origin + h
                    if t >= len(resid_te):
                        continue
                    results[v][h]["p"].append(sarima_fc[h - 1] + exog_real[h - 1])
                    results[v][h]["a"].append(y_te[t])

        for v in VARIANTS:
            for h in HORIZONS:
                a = np.array(results[v][h]["a"])
                p = np.array(results[v][h]["p"])
                rows.append({"variant": v, "strategy": name, "horizon": h,
                             "mape": mape(a, p), "n": len(a)})
        v0 = {h: r["mape"] for r in rows
              if r["strategy"] == name and r["variant"] == "V0_baseline"
              for h in [r["horizon"]] }
        print(f"  {name:<8} [{time.time()-t0:5.1f}s]  V0: "
              + "  ".join(f"h={h}: {v0[h]:.2f}%" for h in HORIZONS))

    df = pd.DataFrame(rows)

    # Deltas vs V0 and vs oracle
    v0_map = df[df.variant == "V0_baseline"].set_index(["strategy", "horizon"])["mape"]
    df["delta_vs_v0"] = df.apply(
        lambda r: r["mape"] - v0_map.loc[(r["strategy"], r["horizon"])], axis=1)
    oracle = pd.read_csv(ROOT / "data" / "sarimax_results.csv")
    oracle["strategy"] = oracle["model"].str.extract(r"\((.*?)\)$")[0] \
        .str.replace(r"\s*\(.*\)$", "", regex=True)
    o_map = oracle.set_index(["strategy", "horizon_days"])["mape_pct"]
    df["delta_vs_oracle"] = df.apply(
        lambda r: r["mape"] - o_map.loc[(r["strategy"], r["horizon"])], axis=1)
    df.to_csv(ROOT / "data" / "realistic_protocol_sensitivity.csv", index=False)

    # Sanity gate: V0 vs saved realistic results
    saved = pd.read_csv(ROOT / "data" / "sarimax_results_realistic.csv")
    saved["strategy"] = saved["model"].str.extract(r"\((.*?)\)$")[0]
    s_map = saved.set_index(["strategy", "horizon_days"])["mape_pct"]
    v0_df = df[df.variant == "V0_baseline"]
    worst = max(abs(r["mape"] - s_map.loc[(r["strategy"], r["horizon"])])
                for _, r in v0_df.iterrows())
    print(f"\nSanity: V0 vs sarimax_results_realistic.csv — "
          f"max |Δ MAPE| = {worst:.4f} pp "
          f"({'OK' if worst < 0.05 else 'MISMATCH — investigate'})")

    print("\nΔ MAPE vs V0 (pp; negative = refinement helps):")
    for v in VARIANTS[1:]:
        piv = df[df.variant == v].pivot_table(index="strategy",
                                              columns="horizon",
                                              values="delta_vs_v0").round(3)
        print(f"\n--- {v} ---")
        print(piv.to_string())

    # Plot: Filter + Ridge, all variants + oracle
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    styles = {"V0_baseline": ("-", "o"), "V1_nwp_proxy": ("--", "s"),
              "V2_ar_decay": (":", "^"), "V3_trend_clim": ("-.", "v"),
              "V4_pub_lag": ("--", "x")}
    colors = {"Filter": "#4477aa", "Ridge": "#117733"}
    for strat in ["Filter", "Ridge"]:
        for v, (ls, mk) in styles.items():
            d = df[(df.strategy == strat) & (df.variant == v)].sort_values("horizon")
            ax.plot(d["horizon"], d["mape"], linestyle=ls, marker=mk, ms=4,
                    color=colors[strat], alpha=0.85,
                    label=f"{strat} {v.split('_', 1)[1]}")
        o = oracle[oracle.strategy == strat].sort_values("horizon_days")
        ax.plot(o["horizon_days"], o["mape_pct"], lw=2.2, color=colors[strat],
                alpha=0.35, label=f"{strat} oracle")
    ax.set_xscale("log"); ax.set_xticks(HORIZONS); ax.set_xticklabels(HORIZONS)
    ax.set_xlabel("Forecast horizon (days, log scale)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Realistic-protocol refinements — MAPE × horizon (Filter, Ridge)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROOT / "plots" / "realistic_protocol_sensitivity.png",
                bbox_inches="tight", dpi=120)
    plt.close()

    print("\nSaved:")
    print("  data/realistic_protocol_sensitivity.csv")
    print("  plots/realistic_protocol_sensitivity.png")


if __name__ == "__main__":
    main()
