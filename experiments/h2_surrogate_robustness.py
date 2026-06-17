"""
Experiment 15 — H2 OOS-SHAP surrogate robustness (4 review points on
dimension iv + resolving the H2 sign).

Trace finding: the canonical OOS H2 gap on disk is NEGATIVE
(data/interpretability_h2_bootstrap_oos.csv: -2.98 pp [-3.16,-2.81]),
opposite the documented +8.59 pp (which is the in-sample number). This
script establishes what H2 actually is under the OOS protocol and stress-
tests it on the four methodology points.

Parts:
  0  Sign / Fourier decomposition — recompute the OOS gap and the per-feature
     (population, sunshine_h) attribution WITH and WITHOUT the annual Fourier
     terms, to test whether Fourier absorption flipped the gap.
  1  XGB CV-tuning (point i) — randomized TimeSeriesSplit search on TRAIN
     residuals (no test leakage), select by mean CV R²; report tuned vs
     off-the-shelf OOS R² and recompute the H2 gap + CI under the tuned
     surrogate. Decides whether the negative gap is a weak-surrogate artifact.
  2  Unique vs shared signal (point iii) — (a) TreeSHAP interaction values:
     fraction of population/sunshine_h attribution that is interaction (shared)
     vs main-effect; (b) partner ablation: drop each filtered feature's most-
     correlated retained feature from the surrogate input and measure the
     change in the filtered feature's |SHAP|.
  3  Block bootstrap (point iv) — circular block bootstrap (block 7 and 30)
     on the per-row SHAP arrays vs the iid row bootstrap CI.
  H3 sanity — SHAP rank-consistency (Spearman) across Filter/Ridge/Lasso under
     baseline vs tuned surrogate.
  Point ii (version pin / de-hack) is handled in requirements.txt + a probe of
     whether shap 0.48 still needs the builtins.float patch (printed here).

Outputs:
  - data/h2_robustness_gap.csv          gap under each protocol/condition
  - data/h2_robustness_xgb_tuning.csv   baseline vs tuned params + OOS R²
  - data/h2_robustness_unique_signal.csv interaction frac + ablation per feature
  - data/h2_robustness_h3.csv           Spearman ranks baseline vs tuned
"""
from __future__ import annotations

import builtins, json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N_BOOT = 2000
FILTERED_OUT = ["population", "sunshine_h"]

CLIMATE = ["temp_c", "wind_ms", "precip_mm", "sunshine_h",
           "global_rad", "pressure_hpa", "humidity_pct", "cloudiness", "nao"]
STRUCTURAL = ["price_eur_kwh", "gdp_mln_eur", "population",
              "wind_energy_gwh", "solar_energy_gwh"]
CONTINUOUS = CLIMATE + STRUCTURAL
RAW_CAL = ["day_of_week", "month", "is_weekend", "is_holiday"]
FOURIER = ["sin1_ann", "cos1_ann", "sin2_ann", "cos2_ann"]

XGB_BASELINE = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=SEED, verbosity=0, n_jobs=-1, tree_method="hist")


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
    tcorr = {c: abs(np.corrcoef(X_df[c].values, y)[0, 1]) for c in cols}
    drop = set()
    cm = X_df.corr().abs()
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if j <= i or c1 in drop or c2 in drop:
                continue
            if cm.loc[c1, c2] > threshold:
                drop.add(c1 if tcorr[c1] < tcorr[c2] else c2)
    return [c for c in cols if c not in drop]


def explainer_for(model):
    """shap TreeExplainer; try the clean path first, fall back to the
    builtins.float patch. Returns (explainer, patch_needed)."""
    try:
        return shap.TreeExplainer(model), False
    except Exception:
        _orig = builtins.float
        def _f(x):
            if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
                return _orig(x[1:-1])
            return _orig(x)
        builtins.float = _f
        try:
            ex = shap.TreeExplainer(model.get_booster())
        finally:
            builtins.float = _orig
        return ex, True


def build_splits(include_fourier):
    train = add_fourier(pd.read_csv(ROOT / "data" / "train.csv", parse_dates=["date"]))
    test = add_fourier(pd.read_csv(ROOT / "data" / "test.csv", parse_dates=["date"]))
    cal = RAW_CAL + (FOURIER if include_fourier else [])
    sx = StandardScaler()
    Xtr_c = sx.fit_transform(train[CONTINUOUS]); Xte_c = sx.transform(test[CONTINUOUS])

    def stack(arr, df):
        cont = pd.DataFrame(arr, columns=CONTINUOUS, index=df.index)
        return pd.concat([cont.reset_index(drop=True),
                          df[cal].reset_index(drop=True)], axis=1)

    X_tr, X_te = stack(Xtr_c, train), stack(Xte_c, test)
    y_tr, y_te = train["demand_MW"].values, test["demand_MW"].values
    filter_cols = correlation_filter(train[CONTINUOUS], y_tr) + cal
    with open(ROOT / "data" / "strategy_elasticnet_params.json") as f:
        ep = json.load(f)
    enet = ElasticNet(alpha=ep["alpha"], l1_ratio=ep["l1_ratio"],
                      max_iter=5000, random_state=SEED).fit(X_tr.values, y_tr)
    lasso_cols = sorted({c for c, k in zip(X_tr.columns, enet.coef_) if k != 0} | set(cal),
                        key=lambda c: list(X_tr.columns).index(c))
    return dict(X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
                filter_cols=filter_cols, lasso_cols=lasso_cols)


def stage1_resid(X_tr_s, X_te_s, y_tr, y_te, model):
    model.fit(X_tr_s, y_tr)
    return y_tr - model.predict(X_tr_s), y_te - model.predict(X_te_s)


def r2_var(yt, yp):
    return 1.0 - np.var(yt - yp) / np.var(yt)


def oos_surrogate(resid_tr, resid_te, X_shap_tr, X_shap_te, params):
    m = xgb.XGBRegressor(**params).fit(X_shap_tr, resid_tr)
    r2 = r2_var(resid_te, m.predict(X_shap_te))
    ex, patched = explainer_for(m)
    sv = ex.shap_values(X_shap_te)
    return m, ex, sv, r2, patched


def pct_filtered(sv, idx):
    ma = np.abs(sv).mean(axis=0)
    return 100 * ma[idx].sum() / ma.sum()


def iid_ci(sv_f, sv_r, idx, rng):
    n = sv_f.shape[0]
    g = [pct_filtered(sv_f[b], idx) - pct_filtered(sv_r[b], idx)
         for b in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    return float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))


def block_ci(sv_f, sv_r, idx, block, rng):
    n = sv_f.shape[0]
    nb = int(np.ceil(n / block))
    g = []
    for _ in range(N_BOOT):
        starts = rng.integers(0, n, nb)
        rows = np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]
        g.append(pct_filtered(sv_f[rows], idx) - pct_filtered(sv_r[rows], idx))
    return float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))


def main():
    print("=" * 70)
    print("EXPERIMENT 15 — H2 OOS-SHAP surrogate robustness")
    print("=" * 70)
    rng = np.random.default_rng(SEED)
    with open(ROOT / "data" / "strategy_ridge_params.json") as f:
        rp = json.load(f)

    gap_rows = []

    # ---- Part 0: sign / Fourier decomposition ---------------------------
    print("\n[0] OOS gap — with vs without annual Fourier terms")
    svs = {}
    for fourier in (True, False):
        s = build_splits(fourier)
        Xshap_tr, Xshap_te = s["X_tr"], s["X_te"]
        idx = [list(Xshap_te.columns).index(f) for f in FILTERED_OUT]
        rt_f, re_f = stage1_resid(s["X_tr"][s["filter_cols"]], s["X_te"][s["filter_cols"]],
                                  s["y_tr"], s["y_te"], LinearRegression())
        rt_r, re_r = stage1_resid(s["X_tr"], s["X_te"], s["y_tr"], s["y_te"],
                                  Ridge(alpha=rp["lambda"]))
        _, _, sv_f, r2f, patched = oos_surrogate(rt_f, re_f, Xshap_tr, Xshap_te, XGB_BASELINE)
        _, _, sv_r, r2r, _ = oos_surrogate(rt_r, re_r, Xshap_tr, Xshap_te, XGB_BASELINE)
        pf, pr = pct_filtered(sv_f, idx), pct_filtered(sv_r, idx)
        gap = pf - pr
        lo, hi = iid_ci(sv_f, sv_r, idx, np.random.default_rng(SEED))
        tag = "fourier" if fourier else "no_fourier"
        if fourier:
            svs.update(sv_f=sv_f, sv_r=sv_r, idx=idx, splits=s, patched=patched)
        # per-feature
        maf, mar = np.abs(sv_f).mean(0), np.abs(sv_r).mean(0)
        per = {}
        for f in FILTERED_OUT:
            j = list(Xshap_te.columns).index(f)
            per[f] = (100 * maf[j] / maf.sum(), 100 * mar[j] / mar.sum())
        gap_rows.append(dict(protocol=f"oos_baseline_{tag}", pct_filter=round(pf, 3),
                             pct_ridge=round(pr, 3), gap_pp=round(gap, 3),
                             ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                             r2_filter=round(r2f, 3), r2_ridge=round(r2r, 3)))
        print(f"  {tag:<10} gap={gap:+.3f} pp  CI=[{lo:+.3f},{hi:+.3f}]  "
              f"pctF={pf:.2f} pctR={pr:.2f}  R2 f/r={r2f:.2f}/{r2r:.2f}")
        for f in FILTERED_OUT:
            print(f"      {f:<12} Filter%={per[f][0]:.2f}  Ridge%={per[f][1]:.2f}")
    print(f"  [shap patch needed under installed versions? {svs['patched']}]")

    # sanity vs disk OOS number
    try:
        disk = pd.read_csv(ROOT / "data" / "interpretability_h2_bootstrap_oos.csv")
        g0 = next(r for r in gap_rows if r["protocol"] == "oos_baseline_fourier")
        print(f"  [sanity] disk OOS gap={disk['observed_gap_pp'].iloc[0]:+.3f}  "
              f"recomputed={g0['gap_pp']:+.3f}  "
              f"(|Δ|={abs(disk['observed_gap_pp'].iloc[0]-g0['gap_pp']):.3f})")
    except Exception as e:
        print(f"  [sanity] skipped ({e})")

    # ---- Part 1: XGB CV-tuning (no test leakage) ------------------------
    print("\n[1] XGB CV-tuning — randomized TimeSeriesSplit search on TRAIN residuals")
    s = svs["splits"]
    Xshap_tr, Xshap_te = s["X_tr"], s["X_te"]
    idx = svs["idx"]
    # tuning target: Ridge train residuals (full-feature)
    rt_r, re_r = stage1_resid(s["X_tr"], s["X_te"], s["y_tr"], s["y_te"], Ridge(alpha=rp["lambda"]))
    grid = dict(n_estimators=[100, 200, 300, 400, 600], max_depth=[2, 3, 4, 5, 6],
                learning_rate=[0.02, 0.05, 0.1], min_child_weight=[1, 3, 5, 10],
                subsample=[0.6, 0.8, 1.0], colsample_bytree=[0.6, 0.8, 1.0],
                reg_lambda=[0.5, 1, 2, 5])
    tscv = TimeSeriesSplit(n_splits=5)
    rsr = np.random.default_rng(SEED)
    best, best_score = None, -np.inf
    t0 = time.time()
    for _ in range(40):
        params = {k: rsr.choice(v).item() for k, v in grid.items()}
        params.update(random_state=SEED, verbosity=0, n_jobs=-1, tree_method="hist")
        fold = []
        for tr, va in tscv.split(Xshap_tr):
            m = xgb.XGBRegressor(**params).fit(Xshap_tr.iloc[tr], rt_r[tr])
            fold.append(r2_var(rt_r[va], m.predict(Xshap_tr.iloc[va])))
        sc = float(np.mean(fold))
        if sc > best_score:
            best_score, best = sc, params
    print(f"  searched 40 configs in {time.time()-t0:.1f}s; best CV R²={best_score:.4f}")
    print(f"  tuned params: { {k:best[k] for k in grid} }")

    tuning_rows = []
    sv_tuned = {}
    for name, X_tr_s, X_te_s, model in [
        ("Filter", s["X_tr"][s["filter_cols"]], s["X_te"][s["filter_cols"]], LinearRegression()),
        ("Ridge", s["X_tr"], s["X_te"], Ridge(alpha=rp["lambda"])),
        ("Lasso", s["X_tr"][s["lasso_cols"]], s["X_te"][s["lasso_cols"]],
         ElasticNet(alpha=json.load(open(ROOT/'data'/'strategy_elasticnet_params.json'))['alpha'],
                    l1_ratio=1.0, max_iter=5000, random_state=SEED))]:
        rt, re = stage1_resid(X_tr_s, X_te_s, s["y_tr"], s["y_te"], model)
        _, _, sv_b, r2_b, _ = oos_surrogate(rt, re, Xshap_tr, Xshap_te, XGB_BASELINE)
        _, _, sv_t, r2_t, _ = oos_surrogate(rt, re, Xshap_tr, Xshap_te, best)
        sv_tuned[name] = dict(baseline=sv_b, tuned=sv_t)
        tuning_rows.append(dict(strategy=name, r2_baseline=round(r2_b, 4), r2_tuned=round(r2_t, 4)))
        print(f"  {name:<8} OOS R²: baseline={r2_b:.3f} -> tuned={r2_t:.3f}")

    # H2 gap under tuned surrogate
    gpf = pct_filtered(sv_tuned["Filter"]["tuned"], idx)
    gpr = pct_filtered(sv_tuned["Ridge"]["tuned"], idx)
    lo_t, hi_t = iid_ci(sv_tuned["Filter"]["tuned"], sv_tuned["Ridge"]["tuned"], idx,
                        np.random.default_rng(SEED))
    gap_rows.append(dict(protocol="oos_tuned_fourier", pct_filter=round(gpf, 3),
                         pct_ridge=round(gpr, 3), gap_pp=round(gpf - gpr, 3),
                         ci_lo=round(lo_t, 3), ci_hi=round(hi_t, 3),
                         r2_filter=tuning_rows[0]["r2_tuned"], r2_ridge=tuning_rows[1]["r2_tuned"]))
    print(f"  H2 gap (tuned surrogate): {gpf-gpr:+.3f} pp  CI=[{lo_t:+.3f},{hi_t:+.3f}]")

    # ---- Part 3: block bootstrap (point iv) -----------------------------
    print("\n[3] Block bootstrap on the H2 gap (baseline surrogate, Fourier)")
    sv_f, sv_r = svs["sv_f"], svs["sv_r"]
    for blk in (7, 30):
        lo_b, hi_b = block_ci(sv_f, sv_r, idx, blk, np.random.default_rng(SEED))
        gap_rows.append(dict(protocol=f"block_bootstrap_b{blk}",
                             pct_filter=round(pct_filtered(sv_f, idx), 3),
                             pct_ridge=round(pct_filtered(sv_r, idx), 3),
                             gap_pp=round(pct_filtered(sv_f, idx) - pct_filtered(sv_r, idx), 3),
                             ci_lo=round(lo_b, 3), ci_hi=round(hi_b, 3),
                             r2_filter=np.nan, r2_ridge=np.nan))
        print(f"  block={blk:<3} CI=[{lo_b:+.3f},{hi_b:+.3f}]")
    pd.DataFrame(gap_rows).to_csv(ROOT / "data" / "h2_robustness_gap.csv", index=False)
    pd.DataFrame(tuning_rows + [{"strategy": "_best_params", **{k: best[k] for k in grid}}]
                 ).to_csv(ROOT / "data" / "h2_robustness_xgb_tuning.csv", index=False)

    # ---- Part 2: unique vs shared signal (point iii) --------------------
    print("\n[2] Unique vs shared signal — interaction fraction + partner ablation")
    # (a) interaction values on the Ridge baseline surrogate
    rt_r2, re_r2 = stage1_resid(s["X_tr"], s["X_te"], s["y_tr"], s["y_te"], Ridge(alpha=rp["lambda"]))
    m_r = xgb.XGBRegressor(**XGB_BASELINE).fit(Xshap_tr, rt_r2)
    ex_r, _ = explainer_for(m_r)
    inter = ex_r.shap_interaction_values(Xshap_te)   # (n, F, F)
    cols = list(Xshap_te.columns)
    uniq_rows = []
    # correlations on train continuous block
    corr = s["X_tr"][CONTINUOUS].corr().abs()
    for f in FILTERED_OUT:
        j = cols.index(f)
        total = np.abs(inter[:, j, :]).mean(0).sum()          # all attribution touching f
        main = np.abs(inter[:, j, j]).mean()                  # pure main effect
        inter_frac = 1 - main / total if total > 0 else np.nan
        # nearest retained partner
        retained_cont = [c for c in CONTINUOUS if c in s["filter_cols"]]
        partner = corr.loc[f, retained_cont].drop(f, errors="ignore").idxmax()
        partner_corr = float(corr.loc[f, partner])
        # ablation: drop partner from surrogate input, recompute |SHAP| of f
        keep = [c for c in cols if c != partner]
        m_ab = xgb.XGBRegressor(**XGB_BASELINE).fit(Xshap_tr[keep], rt_r2)
        ex_ab, _ = explainer_for(m_ab)
        sv_ab = ex_ab.shap_values(Xshap_te[keep])
        shap_f_base = np.abs(sv_r[:, j]).mean()
        shap_f_abl = np.abs(sv_ab[:, keep.index(f)]).mean()
        uniq_rows.append(dict(feature=f, interaction_fraction=round(float(inter_frac), 3),
                              nearest_retained=partner, partner_corr=round(partner_corr, 3),
                              shap_baseline=round(float(shap_f_base), 2),
                              shap_partner_ablated=round(float(shap_f_abl), 2),
                              pct_change=round(100 * (shap_f_abl - shap_f_base) / shap_f_base, 1)))
        print(f"  {f:<12} interaction frac={inter_frac:.2f}  partner={partner}"
              f"(r={partner_corr:.2f})  |SHAP| {shap_f_base:.1f}->{shap_f_abl:.1f} "
              f"({uniq_rows[-1]['pct_change']:+.0f}%)")
    pd.DataFrame(uniq_rows).to_csv(ROOT / "data" / "h2_robustness_unique_signal.csv", index=False)

    # ---- H3 sanity: rank consistency baseline vs tuned ------------------
    print("\n[H3] SHAP rank consistency (Spearman) — baseline vs tuned surrogate")
    h3_rows = []
    for proto in ("baseline", "tuned"):
        ranks = {n: pd.Series(np.abs(sv_tuned[n][proto]).mean(0), index=cols).rank(ascending=False)
                 for n in ("Filter", "Ridge", "Lasso")}
        for a, b in [("Filter", "Ridge"), ("Filter", "Lasso"), ("Ridge", "Lasso")]:
            rho = spearmanr(ranks[a], ranks[b]).correlation
            h3_rows.append(dict(protocol=proto, pair=f"{a} vs {b}", spearman_rho=round(float(rho), 3)))
            print(f"  {proto:<8} {a} vs {b}: rho={rho:.3f}")
    pd.DataFrame(h3_rows).to_csv(ROOT / "data" / "h2_robustness_h3.csv", index=False)

    print("\nSaved:")
    for f in ["h2_robustness_gap", "h2_robustness_xgb_tuning",
              "h2_robustness_unique_signal", "h2_robustness_h3"]:
        print(f"  data/{f}.csv")


if __name__ == "__main__":
    main()
