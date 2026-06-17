"""
Experiment 8 — Residual-diagnostics depth (Ljung-Box effect sizes + ARCH-LM).

Addresses two review points on the Stage-2 residual diagnostics:
  1. Ljung-Box with n ≈ 4,280 flags even tiny autocorrelations as significant.
     Reporting only p-values is potentially misleading — this script adds the
     Box statistic, degrees of freedom, and effect-size views (residual ACF
     magnitudes vs the white-noise band ±1.96/sqrt(n)).
  2. Ljung-Box ignores heteroskedasticity — adds Engle's ARCH-LM test.

For each of the five strategies, fits the universal SARIMA(2,0,1)x(0,1,1,7)
on Stage-1 train residuals (identical to notebook 03) and reports:
  - Ljung-Box Q, df and p at lags 7/14/21/28/365, both unadjusted (model_df=0,
    as in notebook 03 cell 30) and adjusted for the 4 estimated ARMA
    parameters (model_df = p+q+P+Q = 4).
  - Residual ACF at lags 1..28 and 365; max/mean |r_k| vs the 95% white-noise
    band; sum of squared autocorrelations (the quantity LB scales by ~n).
  - ARCH-LM (het_arch) at 7 and 28 lags on the SARIMA residuals.

Outputs:
  - data/residual_diagnostics_lb.csv    Ljung-Box detail per strategy × lag
  - data/residual_diagnostics_acf.csv   residual ACF per strategy × lag
  - data/residual_diagnostics_arch.csv  ARCH-LM per strategy
"""
from __future__ import annotations

import json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
UNIVERSAL_ORDER = (2, 0, 1)
UNIVERSAL_SEASONAL = (0, 1, 1, 7)
MODEL_DF = sum(UNIVERSAL_ORDER[::2]) + UNIVERSAL_SEASONAL[0] + UNIVERSAL_SEASONAL[2]  # p+q+P+Q = 4
LB_LAGS = [7, 14, 21, 28, 365]


def load_strategy(name):
    return {s: pd.read_csv(ROOT / "data" / f"strategy_{name}_{s}.csv",
                           parse_dates=["date"]).drop(columns=["date"]).values
            for s in ("train", "test")}


def build_strategies():
    train = pd.read_csv(ROOT / "data" / "train.csv", parse_dates=["date"])
    y_tr = train["demand_MW"].values

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
        stage1 = model.fit(X["train"], y_tr)
        out[name] = y_tr - stage1.predict(X["train"])
    return out


def main():
    print("=" * 70)
    print("EXPERIMENT 8 — Residual diagnostics depth (LB effect sizes, ARCH-LM)")
    print("=" * 70)

    resids = build_strategies()
    lb_rows, acf_rows, arch_rows = [], [], []

    for name, resid_tr in resids.items():
        t0 = time.time()
        r = SARIMAX(resid_tr, order=UNIVERSAL_ORDER,
                    seasonal_order=UNIVERSAL_SEASONAL,
                    enforce_stationarity=False,
                    enforce_invertibility=False).fit(disp=False)
        e = np.asarray(r.resid)
        n = len(e)
        band = 1.96 / np.sqrt(n)

        # Ljung-Box: notebook protocol (model_df=0) and ARMA-adjusted df
        lb0 = acorr_ljungbox(e, lags=LB_LAGS, return_df=True)
        lba = acorr_ljungbox(e, lags=LB_LAGS, model_df=MODEL_DF, return_df=True)
        for lag in LB_LAGS:
            lb_rows.append({
                "strategy": name, "lag": lag, "n": n,
                "lb_stat": float(lb0.loc[lag, "lb_stat"]),
                "df_unadjusted": lag,
                "p_unadjusted": float(lb0.loc[lag, "lb_pvalue"]),
                "df_adjusted": lag - MODEL_DF,
                "p_adjusted": float(lba.loc[lag, "lb_pvalue"]),
            })

        # Effect sizes: residual ACF magnitudes
        r_k = acf(e, nlags=365, fft=True)
        for lag in list(range(1, 29)) + [365]:
            acf_rows.append({"strategy": name, "lag": lag,
                             "acf": float(r_k[lag]),
                             "wn_band_95": band,
                             "exceeds_band": bool(abs(r_k[lag]) > band)})
        a28 = np.abs(r_k[1:29])
        print(f"  {name:<8} n={n}  band=±{band:.4f}  "
              f"max|r|₁..₂₈={a28.max():.4f} (lag {int(np.argmax(a28))+1})  "
              f"mean|r|₁..₂₈={a28.mean():.4f}  Σr²₁..₂₈={float((a28**2).sum()):.5f}  "
              f"|r₃₆₅|={abs(r_k[365]):.4f}  [{time.time()-t0:.1f}s]")

        # ARCH-LM on SARIMA residuals
        for nlags in (7, 28):
            lm_stat, lm_p, f_stat, f_p = het_arch(e, nlags=nlags)
            arch_rows.append({"strategy": name, "nlags": nlags,
                              "lm_stat": lm_stat, "lm_pvalue": lm_p,
                              "f_stat": f_stat, "f_pvalue": f_p})

    lb_df = pd.DataFrame(lb_rows)
    acf_df = pd.DataFrame(acf_rows)
    arch_df = pd.DataFrame(arch_rows)
    lb_df.to_csv(ROOT / "data" / "residual_diagnostics_lb.csv", index=False)
    acf_df.to_csv(ROOT / "data" / "residual_diagnostics_acf.csv", index=False)
    arch_df.to_csv(ROOT / "data" / "residual_diagnostics_arch.csv", index=False)

    print("\nLjung-Box detail (Q, df, p — unadjusted and ARMA-df-adjusted):")
    print(lb_df.round(4).to_string(index=False))

    print("\nResidual ACF — lags exceeding the 95% white-noise band, per strategy:")
    exc = acf_df[(acf_df["exceeds_band"]) & (acf_df["lag"] <= 28)]
    for name, sub in exc.groupby("strategy"):
        lags = ", ".join(f"{int(l)}({v:+.3f})"
                         for l, v in zip(sub["lag"], sub["acf"]))
        print(f"  {name:<8} {lags}")

    print("\nARCH-LM (Engle) on SARIMA residuals:")
    print(arch_df.round(4).to_string(index=False))

    print("\nSaved:")
    for f in ["residual_diagnostics_lb.csv", "residual_diagnostics_acf.csv",
              "residual_diagnostics_arch.csv"]:
        print(f"  data/{f}")


if __name__ == "__main__":
    main()
