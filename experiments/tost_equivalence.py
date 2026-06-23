"""
Experiment — TOST equivalence test for H1 (MAPE non-inferiority).

WHY. A non-significant Diebold-Mariano (DM) result does NOT establish that two
strategies are practically equivalent in accuracy: absence of evidence is not
evidence of absence. To support the H1 non-inferiority claim we add a formal
two-one-sided-tests (TOST) equivalence test on the per-horizon MAPE difference,
REUSING the exact Newey-West(Bartlett) long-run variance + Harvey-Leybourne-
Newbold (HLN) small-sample correction and the same t degrees of freedom as the
corrected DM test (`dm_correction.nw_hln_se`). The two tests therefore share one
variance estimator and are mutually consistent — the TOST is NOT computed with a
naive iid variance.

TESTED QUANTITY (per strategy pair A,B; per horizon h), in percentage points:
    dbar = MAPE_A - MAPE_B
built as the mean of the per-origin absolute-percentage-error differential
(100*|y-yhat|/|y|, from `dm_correction.pct_errors_per_origin`), so that dbar
equals the difference of the two strategies' MAPE values already in
data/sarimax_results.csv / sarimax_mape_table.csv. The sanity check below
asserts this for every pair x horizon; if it failed it would mean the loss
differential is on a non-MAPE scale (e.g. MW absolute error) and the test would
be wrong, so the script STOPS in that case.

EQUIVALENCE MARGIN: delta = 0.5 pp, FIXED A PRIORI (see DELTA). Rationale: 0.5 pp
is an order of magnitude below the accuracy gaps that distinguish modelling
approaches in this literature — the SARIMAX models beat the seasonal-naive /
OLS-temperature baselines by several pp — so a MAPE difference under 0.5 pp is
not operationally meaningful for Dutch daily demand. delta is NOT tuned to obtain
a significant result and is NOT derived from the observed spread of strategies.

TOST PROCEDURE (alpha = 0.05), per cell:
    se      = nw_hln_se(d)            # overlap-aware NW + HLN, origin units
    df      = n - 1                   # identical to the DM test for this horizon
    t_lower = (dbar + delta) / se     # H0_lower: true diff <= -delta
    t_upper = (dbar - delta) / se     # H0_upper: true diff >= +delta
    p_lower = 1 - t.cdf(t_lower, df)  # P(T >= t_lower)
    p_upper =     t.cdf(t_upper, df)  # P(T <= t_upper)
    p_tost  = max(p_lower, p_upper)
    equivalent = p_tost < alpha       # both one-sided nulls rejected
The 90% CI (= 1 - 2*alpha) for dbar is [dbar -/+ t.ppf(0.95, df)*se]; equivalence
holds iff this CI lies entirely inside (-delta, +delta). The two verdicts are
mathematically identical and are asserted to agree in every cell.

GUARDRAIL / HONESTY. With overlapping windows (h > 15 days, m = ceil(h/15) > 1)
the effective sample is small and power is limited, so some or all cells may come
back "inconclusive" (the one-sided nulls cannot be rejected => equivalence cannot
be concluded). That is the correct, acceptable outcome and is reported as found;
it is NOT to be "fixed" by widening delta or swapping in an iid variance.

PAIRS. Convention A = regularised strategy, B = Filter, so a positive dbar means
the regularised strategy is worse.
  Required (carry the H1 claim): Ridge vs Filter, Lasso vs Filter.
  Completeness:                  Ridge vs Lasso, PCA vs Filter.

PROTOCOL. Oracle exog by default (matches the main MAPE table and the H1
verdict). Pass --realistic to ALSO run the realistic (non-oracle) protocol of
experiments/realistic_forecast.py and save it separately.

OUTPUTS
  - data/tost_equivalence_oracle.csv      tidy results, one row per pair x horizon
  - data/tost_equivalence_realistic.csv   (only with --realistic)
  console: per-horizon df + NW truncation lag (cross-check vs the DM test),
           per-pair summary, an appendix-ready LaTeX booktabs tabular, and one
           plain-language verdict line per required pair.
The test is analytic (no resampling); SEED is set only to honour the repo's
determinism convention (it affects the ElasticNet refit, not the TOST itself).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import dm_correction as dm   # shared NW+HLN variance helper + builders/errors

HORIZONS = dm.HORIZONS
MAX_H = dm.MAX_H
EVAL_STEP = dm.EVAL_STEP
ALPHA = 0.05
SEED = 42

# delta: equivalence margin in percentage points, FIXED A PRIORI (see docstring).
# 0.5 pp is an order of magnitude below the multi-pp gaps that separate modelling
# approaches in this literature. Do NOT tune this to the data.
DELTA = 0.5

# Convention A = regularised, B = Filter  => positive dbar = regularised is worse.
REQUIRED_PAIRS = [("Ridge", "Filter"), ("Lasso", "Filter")]
COMPLETENESS_PAIRS = [("Ridge", "Lasso"), ("PCA", "Filter")]
TOST_PAIRS = REQUIRED_PAIRS + COMPLETENESS_PAIRS
NEEDED_STRATEGIES = sorted({s for pair in TOST_PAIRS for s in pair})

# Map saved-table model labels -> strategy keys.
_TABLE_KEYWORDS = [("OLS-All", "OLS-All"), ("Filter", "Filter"), ("PCA", "PCA"),
                   ("Ridge", "Ridge"), ("Lasso", "Lasso")]


# ---------------------------------------------------------------------------
# Core TOST on one per-origin loss differential
# ---------------------------------------------------------------------------
def tost_cell(d, h, delta=DELTA, alpha=ALPHA, step=EVAL_STEP):
    """TOST equivalence test on a per-origin MAPE-scale loss differential d (pp).

    Reuses dm_correction.nw_hln_se => identical NW+HLN se and df as the DM test.
    Returns a dict with every reported quantity for one (pair, horizon) cell and
    asserts the p-value verdict and the 90% CI verdict agree.
    """
    d = np.asarray(d, dtype=float)
    dbar, se, df, m, L = dm.nw_hln_se(d, h, step)

    t_lower = (dbar + delta) / se          # H0_lower: true diff <= -delta
    t_upper = (dbar - delta) / se          # H0_upper: true diff >= +delta
    p_lower = 1.0 - scipy_stats.t.cdf(t_lower, df)   # P(T >= t_lower)
    p_upper = scipy_stats.t.cdf(t_upper, df)         # P(T <= t_upper)
    p_tost = max(p_lower, p_upper)
    equiv_p = bool(p_tost < alpha)

    # 90% CI (= 1 - 2*alpha) and the equivalent "CI inside (-delta, delta)" rule.
    tcrit = scipy_stats.t.ppf(1.0 - alpha, df)       # = t.ppf(0.95, df)
    ci_low = dbar - tcrit * se
    ci_high = dbar + tcrit * se
    equiv_ci = bool((ci_low > -delta) and (ci_high < delta))

    if equiv_p != equiv_ci:
        raise AssertionError(
            f"TOST verdict mismatch at h={h}: p_tost={p_tost:.6g} "
            f"(equivalent={equiv_p}) vs 90% CI [{ci_low:.4f}, {ci_high:.4f}] "
            f"inside (-{delta}, {delta}) = {equiv_ci}")

    return {
        "horizon": h, "n_origins": len(d), "df": df,
        "m_origin_units": m, "nw_truncation": L,
        "MAPE_A": np.nan, "MAPE_B": np.nan,   # filled by caller
        "dbar": dbar, "se": se, "delta": delta,
        "ci90_low": ci_low, "ci90_high": ci_high,
        "t_lower": t_lower, "t_upper": t_upper,
        "p_lower": p_lower, "p_upper": p_upper, "p_tost": p_tost,
        "verdict": "equivalent" if equiv_p else "inconclusive",
    }


# ---------------------------------------------------------------------------
# Per-origin percentage errors — oracle and realistic protocols
# ---------------------------------------------------------------------------
def oracle_pct_errors():
    """Per-origin absolute percentage errors per strategy (oracle exog).

    Reuses dm_correction.build_strategies + pct_errors_per_origin so the per-origin
    loss series is constructed identically to the corrected DM test.
    """
    strategies = dm.build_strategies()
    out = {}
    for name in NEEDED_STRATEGIES:
        out[name] = dm.pct_errors_per_origin(strategies[name])
    return out


def _rolling_pct_realistic(stage1, sarima_r, X_te_strategy, y_te, builder,
                           eval_step=EVAL_STEP):
    """Per-origin absolute percentage errors under the realistic exog builder.

    Mirrors realistic_forecast.rolling_eval_realistic, but accumulates the
    per-origin APE (100*|y-yhat|/|y|) instead of only the aggregate MAPE.
    """
    exog_oracle = stage1.predict(X_te_strategy)
    resid_te = y_te - exog_oracle
    pe = {h: [] for h in HORIZONS}
    for origin in range(0, len(resid_te) - MAX_H, eval_step):
        res_ext = sarima_r if origin == 0 else sarima_r.append(
            endog=resid_te[:origin], refit=False)
        sarima_fc = res_ext.forecast(steps=MAX_H)
        exog_real = stage1.predict(builder(origin))
        for h in HORIZONS:
            t = origin + h
            if t >= len(resid_te):
                continue
            pred = sarima_fc[h - 1] + exog_real[h - 1]
            pe[h].append(100.0 * abs(y_te[t] - pred) / abs(y_te[t]))
    return {h: np.array(v) for h, v in pe.items()}


def realistic_pct_errors():
    """Per-origin absolute percentage errors per strategy (non-oracle exog).

    Reuses realistic_forecast's splits, climatology and exog builder; the
    strategy spec list mirrors realistic_forecast.run_strategies (kept in sync;
    the regenerated MAPE is cross-checked against sarimax_results_realistic.csv).
    """
    import realistic_forecast as rf

    splits = rf.build_splits()
    with open(ROOT / "data" / "strategy_ridge_params.json") as f:
        rp = json.load(f)
    with open(ROOT / "data" / "strategy_elasticnet_params.json") as f:
        ep = json.load(f)

    X_tr = splits["train"]["X"]
    y_tr = splits["train"]["y"]
    X_te = splits["test"]["X"]
    y_te = splits["test"]["y"]

    climatology = rf.compute_climatology_scaled(
        X_tr, splits["train"]["dates"], rf.CLIMATE)
    val_last_row = splits["val"]["X"].iloc[-1]
    base_builder = rf.make_base_realistic_builder(
        X_te, val_last_row, climatology, splits["test"]["dates"])

    fcols, lcols, to_pca = splits["filter_cols"], splits["lasso_cols"], splits["to_pca"]

    specs = {
        "Filter": (X_tr[fcols], X_te[fcols], LinearRegression(),
                   lambda o: base_builder(o)[fcols]),
        "PCA":    (to_pca(X_tr), to_pca(X_te), LinearRegression(),
                   lambda o: to_pca(base_builder(o))),
        "Ridge":  (X_tr, X_te, Ridge(alpha=rp["lambda"]),
                   lambda o: base_builder(o)),
        "Lasso":  (X_tr[lcols], X_te[lcols],
                   ElasticNet(alpha=ep["alpha"], l1_ratio=ep["l1_ratio"],
                              max_iter=5000, random_state=rf.SEED),
                   lambda o: base_builder(o)[lcols]),
    }
    out = {}
    for name in NEEDED_STRATEGIES:
        Xtr_s, Xte_s, model, builder = specs[name]
        stage1, sarima_r, _ = rf.fit_two_stage(Xtr_s, y_tr, model)
        out[name] = _rolling_pct_realistic(stage1, sarima_r, Xte_s, y_te, builder)
    return out


# ---------------------------------------------------------------------------
# Sanity check: dbar must equal the saved MAPE difference (MAPE scale)
# ---------------------------------------------------------------------------
def load_table_mape(csv_name):
    """{strategy: {h: MAPE}} from a results CSV (model, horizon_days, mape_pct)."""
    df = pd.read_csv(ROOT / "data" / csv_name)
    out = {}
    for _, r in df.iterrows():
        for kw, name in _TABLE_KEYWORDS:
            if kw in str(r["model"]):
                out.setdefault(name, {})[int(r["horizon_days"])] = float(r["mape_pct"])
                break
    return out


def sanity_check_scale(pct, table_csv, label, tol=0.05):
    """Confirm dbar == MAPE_A - MAPE_B (MAPE scale), cross-checked vs the table.

    Two checks per pair x horizon:
      (1) definitional: dbar (mean of the differential) == mean(APE_A) - mean(APE_B)
          to ~1e-9. This is what proves the loss is on the MAPE (pp) scale — were
          it the MW absolute-error differential, dbar would be ~hundreds, not ~0.1.
      (2) reproduction: the regenerated per-strategy MAPE matches the saved table
          within `tol` pp.
    Raises (STOPS) on any failure.
    """
    table = load_table_mape(table_csv)
    print(f"\nSANITY CHECK — dbar == MAPE_A - MAPE_B on the MAPE scale ({label}):")
    print(f"  reference table: data/{table_csv}")
    header = (f"  {'pair':<16}{'h':>4}{'MAPE_A':>9}{'tblA':>8}{'MAPE_B':>9}"
              f"{'tblB':>8}{'dbar':>9}{'tbl_diff':>10}{'|Δdef|':>10}")
    print(header)
    max_def, max_tbl = 0.0, 0.0
    for (a, b) in TOST_PAIRS:
        for h in HORIZONS:
            da, db = pct[a][h], pct[b][h]
            n = min(len(da), len(db))
            mape_a, mape_b = float(da[:n].mean()), float(db[:n].mean())
            dbar = float((da[:n] - db[:n]).mean())
            def_err = abs(dbar - (mape_a - mape_b))            # check (1)
            tbl_a, tbl_b = table[a][h], table[b][h]
            tbl_diff = tbl_a - tbl_b
            max_def = max(max_def, def_err)
            max_tbl = max(max_tbl, abs(mape_a - tbl_a), abs(mape_b - tbl_b))
            print(f"  {a+' vs '+b:<16}{h:>4}{mape_a:>9.4f}{tbl_a:>8.4f}"
                  f"{mape_b:>9.4f}{tbl_b:>8.4f}{dbar:>9.4f}{tbl_diff:>10.4f}"
                  f"{def_err:>10.2e}")
    print(f"  max |dbar - (MAPE_A-MAPE_B)| (definitional) = {max_def:.2e}")
    print(f"  max |regenerated MAPE - table MAPE|         = {max_tbl:.4f} pp")
    if max_def > 1e-9:
        raise SystemExit("STOP: dbar != MAPE_A - MAPE_B — loss differential is "
                         "NOT on the MAPE scale; the TOST would be wrong.")
    if max_tbl > tol:
        raise SystemExit(f"STOP: regenerated MAPE deviates from data/{table_csv} "
                         f"by > {tol} pp — protocol mismatch; investigate before "
                         "trusting the TOST.")
    print(f"  PASS: loss differential is on the MAPE scale and reproduces the "
          f"saved table (within {tol} pp).")


# ---------------------------------------------------------------------------
# Build the tidy results table
# ---------------------------------------------------------------------------
def build_results(pct):
    rows = []
    for (a, b) in TOST_PAIRS:
        for h in HORIZONS:
            da, db = pct[a][h], pct[b][h]
            n = min(len(da), len(db))
            if len(da) != len(db):
                # origins are shared across strategies, so this should not happen
                print(f"  WARNING {a} vs {b} h={h}: unequal n "
                      f"({len(da)} vs {len(db)}); truncating to {n}.")
            da, db = da[:n], db[:n]
            cell = tost_cell(da - db, h)
            cell["MAPE_A"] = float(da.mean())
            cell["MAPE_B"] = float(db.mean())
            cell["pair"] = f"{a} vs {b}"
            cell["strategy_A"] = a
            cell["strategy_B"] = b
            rows.append(cell)
    cols = ["pair", "strategy_A", "strategy_B", "horizon", "n_origins", "df",
            "m_origin_units", "nw_truncation", "MAPE_A", "MAPE_B", "dbar", "se",
            "delta", "ci90_low", "ci90_high", "t_lower", "t_upper",
            "p_lower", "p_upper", "p_tost", "verdict"]
    return pd.DataFrame(rows)[cols]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_design_table(res):
    """Per-horizon df and NW truncation lag, to cross-check against the DM test."""
    print("\nPer-horizon design (same for every pair; cross-check vs DM test):")
    print(f"  {'h(days)':>8}{'n_origins':>11}{'df':>5}{'m(origin)':>11}"
          f"{'NW_lag_L':>10}")
    sub = res[res["pair"] == res["pair"].iloc[0]].sort_values("horizon")
    for _, r in sub.iterrows():
        print(f"  {int(r.horizon):>8}{int(r.n_origins):>11}{int(r.df):>5}"
              f"{int(r.m_origin_units):>11}{int(r.nw_truncation):>10}")
    print("  (DM test uses the same df = n-1 and NW Bartlett truncation L = m-1.)")


def print_pair_summaries(res):
    for (a, b) in TOST_PAIRS:
        tag = "[required]" if (a, b) in REQUIRED_PAIRS else "[completeness]"
        sub = res[res["pair"] == f"{a} vs {b}"].sort_values("horizon")
        print(f"\n{a} vs {b}  {tag}  (A={a}, B={b}; +dbar => {a} worse)")
        print(f"  {'h':>5}{'dbar(pp)':>10}{'se':>8}{'90% CI':>20}"
              f"{'p_tost':>9}  verdict")
        for _, r in sub.iterrows():
            ci = f"[{r.ci90_low:+.3f},{r.ci90_high:+.3f}]"
            print(f"  {int(r.horizon):>5}{r.dbar:>10.3f}{r.se:>8.3f}{ci:>20}"
                  f"{r.p_tost:>9.3f}  {r.verdict}")


def print_plain_language(res):
    print("\nPlain-language verdicts (required pairs, margin +/-0.5 pp):")
    for (a, b) in REQUIRED_PAIRS:
        sub = res[res["pair"] == f"{a} vs {b}"]
        eq = sorted(int(h) for h in sub.loc[sub.verdict == "equivalent", "horizon"])
        inc = sorted(int(h) for h in sub.loc[sub.verdict == "inconclusive", "horizon"])
        eq_s = "{" + ", ".join(map(str, eq)) + "}" if eq else "{none}"
        inc_s = "{" + ", ".join(map(str, inc)) + "}" if inc else "{none}"
        print(f"  {a} vs {b}: equivalent within +/-0.5 pp at h in {eq_s}; "
              f"inconclusive at h in {inc_s}.")


def print_latex(res, protocol):
    print(f"\nLaTeX booktabs tabular ({protocol} protocol) — appendix-ready:")
    print(r"\begin{tabular}{llrrrrl}")
    print(r"\toprule")
    print(r"Pair & $h$ & $\bar d$ (pp) & SE & 90\% CI & $p_{\mathrm{TOST}}$ "
          r"& Verdict \\")
    print(r"\midrule")
    for pi, (a, b) in enumerate(TOST_PAIRS):
        if pi > 0:
            print(r"\midrule")
        sub = res[res["pair"] == f"{a} vs {b}"].sort_values("horizon")
        for ri, (_, r) in enumerate(sub.iterrows()):
            pair_cell = f"{a} vs {b}" if ri == 0 else ""
            ci = f"$[{r.ci90_low:+.2f}, {r.ci90_high:+.2f}]$"
            verdict = ("equiv." if r.verdict == "equivalent" else "inconcl.")
            ptost = f"{r.p_tost:.3f}"
            print(f"{pair_cell} & {int(r.horizon)} & ${r.dbar:+.3f}$ & "
                  f"{r.se:.3f} & {ci} & {ptost} & {verdict} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"% Equivalence margin delta = 0.5 pp (fixed a priori). "
          r"Variance: Newey-West(Bartlett, L=m-1) + HLN, t with df=n-1 "
          r"(shared with the DM test).")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_protocol(pct, table_csv, out_csv, protocol_label):
    sanity_check_scale(pct, table_csv, protocol_label)
    res = build_results(pct)
    res.to_csv(ROOT / "data" / out_csv, index=False)

    print_design_table(res)
    print_pair_summaries(res)
    print_plain_language(res)
    print_latex(res, protocol_label)

    n_equiv = int((res.verdict == "equivalent").sum())
    print(f"\nSaved: data/{out_csv}  "
          f"({n_equiv}/{len(res)} cells equivalent, "
          f"{len(res) - n_equiv} inconclusive).")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--realistic", action="store_true",
                    help="also run the realistic (non-oracle) protocol")
    args = ap.parse_args()

    np.random.seed(SEED)   # determinism convention (analysis itself is analytic)

    print("=" * 74)
    print("TOST equivalence test for H1 — MAPE non-inferiority (delta = 0.5 pp)")
    print("=" * 74)
    print(f"Pairs (A vs B, +dbar => A worse): "
          f"{', '.join(a+' vs '+b for a, b in TOST_PAIRS)}")
    print(f"Required: {', '.join(a+' vs '+b for a, b in REQUIRED_PAIRS)}")

    print("\n[ORACLE] Regenerating per-origin percentage errors "
          "(notebook-03 protocol)...")
    pct_oracle = oracle_pct_errors()
    run_protocol(pct_oracle, "sarimax_results.csv",
                 "tost_equivalence_oracle.csv", "oracle")

    if args.realistic:
        print("\n" + "=" * 74)
        print("[REALISTIC] Regenerating per-origin percentage errors "
              "(non-oracle exog)...")
        pct_real = realistic_pct_errors()
        run_protocol(pct_real, "sarimax_results_realistic.csv",
                     "tost_equivalence_realistic.csv", "realistic")

    print("\nDone.")


if __name__ == "__main__":
    main()
