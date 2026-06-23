"""
Regenerate thesis figures after the hypothesis renumbering (H1=accuracy,
H2=stability, H3=ranking) and the out-of-sample (OOS) reframing of H3.

Writes ONLY to plots/ (PNG + PDF). Reads data/ artifacts; never writes data/,
so the numbers locked into the LaTeX are untouched. Reuses the OOS-surrogate
machinery from experiments/xgb_surrogate_cv.py (same splits, seed, XGB params).

Regenerated figures (content/title corrected):
  - interpretability_h1_stratified            Fig 1  (title H1 -> H2)
  - interpretability_h3_shap_consistency       Fig 2  (OOS rho matrix 0.94/0.95/0.98)
  - interpretability_shap_beeswarm_{ols-all,filter,pca,ridge,lasso}  (App O, OOS surrogate)
  - interpretability_h2_bootstrap_ci           Fig 20 (title H2 -> H1, in-sample gap)

Plus PNG->PDF export (same basenames) for every other figure used in the thesis.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from PIL import Image

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from xgb_surrogate_cv import (build_splits, stage1_resid_pair, fit_xgb,
                              _shap_tree_explainer, SEED)
import shap

PLOTS = ROOT / "plots"
DATA = ROOT / "data"
S4_LABEL = "Lasso"
FILTERED_OUT_S1 = ["population", "sunshine_h"]


def savefig(name: str) -> None:
    plt.savefig(PLOTS / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.savefig(PLOTS / f"{name}.pdf", bbox_inches="tight")
    plt.close()
    print(f"  regenerated  plots/{name}.png + .pdf")


# ---------------------------------------------------------------------------
# Fig 1 — coefficient stability by feature group (now H2). Redraw from CSV.
# ---------------------------------------------------------------------------
def fig1_stratified() -> None:
    df = pd.read_csv(DATA / "interpretability_h1_stratified.csv").set_index("group")
    groups = ["Behavioral", "Climate", "Economic"]
    cols = {"OLS-All": "ols_all_median_cv", "Filter": "filter_median_cv",
            "Ridge": "ridge_median_cv", S4_LABEL: "lasso_median_cv"}
    colors = {"OLS-All": "#7f7f7f", "Filter": "#e74c3c",
              "Ridge": "#2980b9", S4_LABEL: "#27ae60"}

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(groups))
    width = 0.20
    for i, (strat, col) in enumerate(cols.items()):
        med = [float(df.loc[g, col]) for g in groups]
        bars = ax.bar(x + (i - 1.5) * width, med, width, label=strat,
                      color=colors[strat], alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, med):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel("Median CV (lower = more stable)", fontsize=10)
    ax.set_title("H2: Coefficient Stability by Feature Group\n"
                 "(10-fold rolling CV; lower bar = more stable coefficients)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    plt.tight_layout()
    savefig("interpretability_h1_stratified")


# ---------------------------------------------------------------------------
# Surrogates — OOS (beeswarms + Fig 2) and in-sample (Fig 20), matching
# notebook 04 cell 14 / xgb_surrogate_cv.py exactly.
# ---------------------------------------------------------------------------
def compute_surrogates():
    s = build_splits()
    X_tr, X_te = s["X_tr_wide"], s["X_te_wide"]
    y_tr, y_te = s["y_tr"], s["y_te"]
    with open(DATA / "strategy_ridge_params.json") as f:
        rp = json.load(f)
    with open(DATA / "strategy_elasticnet_params.json") as f:
        ep = json.load(f)

    strategies = [
        ("OLS-All", X_tr, X_te, LinearRegression()),
        ("Filter", X_tr[s["filter_cols"]], X_te[s["filter_cols"]], LinearRegression()),
        ("PCA", s["to_pca"](X_tr), s["to_pca"](X_te), LinearRegression()),
        ("Ridge", X_tr, X_te, Ridge(alpha=rp["lambda"])),
        ("Lasso", X_tr[s["lasso_cols"]], X_te[s["lasso_cols"]],
         ElasticNet(alpha=ep["alpha"], l1_ratio=ep["l1_ratio"],
                    max_iter=5000, random_state=SEED)),
    ]

    sv_oos, sv_in = {}, {}
    for name, Xtr_s, Xte_s, model in strategies:
        resid_tr, resid_te = stage1_resid_pair(Xtr_s, Xte_s, y_tr, y_te, model)
        # OOS: fit XGB on TRAIN residuals, explain TEST  (for beeswarms + Fig 2)
        m_oos = fit_xgb(X_tr, resid_tr)
        sv_oos[name] = _shap_tree_explainer(m_oos).shap_values(X_te)
        # in-sample: fit XGB on TEST residuals, explain TEST  (Fig 20 only)
        if name in ("Filter", "Ridge"):
            m_in = fit_xgb(X_te, resid_te)
            sv_in[name] = _shap_tree_explainer(m_in).shap_values(X_te)
    return X_te, sv_oos, sv_in


# ---------------------------------------------------------------------------
# Fig 2 — H3 ranking-consistency matrix from the OOS surrogate.
# ---------------------------------------------------------------------------
def fig2_h3(X_te, sv_oos) -> None:
    strategies_h3 = ["Filter", "Ridge", S4_LABEL]
    means = {n: np.abs(sv_oos[n]).mean(axis=0) for n in strategies_h3}
    rho = pd.DataFrame(np.nan, index=strategies_h3, columns=strategies_h3)
    for a in strategies_h3:
        rho.loc[a, a] = 1.0
    for a, b in [("Filter", "Ridge"), ("Filter", S4_LABEL), ("Ridge", S4_LABEL)]:
        r = spearmanr(means[a], means[b]).statistic
        rho.loc[a, b] = rho.loc[b, a] = r
    print("  OOS Spearman: " + ", ".join(
        f"{a}-{b}={rho.loc[a, b]:.4f}"
        for a, b in [("Filter", "Ridge"), ("Filter", S4_LABEL), ("Ridge", S4_LABEL)]))

    fig, ax = plt.subplots(figsize=(5, 4))
    rv = rho.astype(float)
    mask = np.eye(len(strategies_h3), dtype=bool)
    sns.heatmap(rv, ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
                annot=rv.round(2).astype(str), fmt="", linewidths=0.5,
                cbar_kws={"label": "Spearman ρ", "shrink": 0.8}, mask=mask)
    for i in range(len(strategies_h3)):
        ax.text(i + 0.5, i + 0.5, "1.00", ha="center", va="center",
                fontsize=10, fontweight="bold", color="#555555")
    ax.set_title("H3: SHAP-Based Predictor Importance\n"
                 "Ranking Consistency (ρ > 0.70 = consistent)",
                 fontsize=11, fontweight="bold")
    ax.set_xticklabels(strategies_h3, rotation=20)
    ax.set_yticklabels(strategies_h3, rotation=0)
    plt.tight_layout()
    savefig("interpretability_h3_shap_consistency")


# ---------------------------------------------------------------------------
# Appendix O beeswarms — OOS surrogate.
# ---------------------------------------------------------------------------
def beeswarms(X_te, sv_oos) -> None:
    names = ["OLS-All", "Filter", "PCA", "Ridge", "Lasso"]
    labels = ["Strategy 0: OLS-All (unregularised)", "Strategy 1: Filter (OLS)",
              "Strategy 2: PCA", "Strategy 3: Ridge", f"Strategy 4: {S4_LABEL}"]
    for name, label in zip(names, labels):
        shap.summary_plot(sv_oos[name], X_te, plot_type="dot",
                          max_display=18, show=False)
        plt.title(f"SHAP Beeswarm — {label}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        savefig(f"interpretability_shap_beeswarm_{name.lower()}")


# ---------------------------------------------------------------------------
# Fig 20 — in-sample information-loss bootstrap (now H1). Matches nb04 cell 19.
# ---------------------------------------------------------------------------
def fig20_bootstrap(X_te, sv_in) -> None:
    s3_cols = list(X_te.columns)
    idx = [s3_cols.index(f) for f in FILTERED_OUT_S1 if f in s3_cols]

    def pct(sv):
        m = np.abs(sv).mean(axis=0)
        return 100 * m[idx].sum() / m.sum()

    def gap(svf, svr):
        return pct(svf) - pct(svr)

    np.random.seed(SEED)
    n_boot, n_test = 2000, sv_in["Filter"].shape[0]
    boot = np.empty(n_boot)
    for k in range(n_boot):
        ix = np.random.choice(n_test, n_test, replace=True)
        boot[k] = gap(sv_in["Filter"][ix], sv_in["Ridge"][ix])
    obs = gap(sv_in["Filter"], sv_in["Ridge"])
    lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
    print(f"  in-sample gap = {obs:+.2f} pp   CI [{lo:+.2f}, {hi:+.2f}]")

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(boot, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(obs, color="crimson", lw=2, label=f"Observed gap ({obs:.2f}pp)")
    ax.axvline(lo, color="grey", lw=1.5, ls="--", label=f"95% CI [{lo:.2f}, {hi:.2f}]")
    ax.axvline(hi, color="grey", lw=1.5, ls="--")
    ax.axvline(0, color="black", lw=1, ls=":")
    ax.set_xlabel("Information loss gap (pp): Filter % − Ridge %")
    ax.set_ylabel("Bootstrap frequency")
    ax.set_title("H1: Bootstrap Distribution of SHAP Information Loss Gap")
    ax.legend(fontsize=9)
    plt.tight_layout()
    savefig("interpretability_h2_bootstrap_ci")


# ---------------------------------------------------------------------------
# Fig 21 — OOS information-loss bootstrap (now H1). Mirrors shap_oos_refresh.py
# but writes ONLY the figure (no data/ CSV), reusing the OOS surrogate.
# ---------------------------------------------------------------------------
def fig21_oos_bootstrap(X_te, sv_oos, sv_in) -> None:
    s3_cols = list(X_te.columns)
    idx = [s3_cols.index(f) for f in FILTERED_OUT_S1 if f in s3_cols]

    def pct(sv):
        m = np.abs(sv).mean(axis=0)
        return 100 * m[idx].sum() / m.sum()

    def gap(svf, svr):
        return pct(svf) - pct(svr)

    rng = np.random.default_rng(SEED)
    n_boot, n_test = 2000, sv_oos["Filter"].shape[0]
    boot = np.empty(n_boot)
    for i in range(n_boot):
        ix = rng.integers(0, n_test, n_test)
        boot[i] = gap(sv_oos["Filter"][ix], sv_oos["Ridge"][ix])
    obs = gap(sv_oos["Filter"], sv_oos["Ridge"])
    lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
    in_obs = gap(sv_in["Filter"], sv_in["Ridge"])
    print(f"  OOS gap = {obs:+.2f} pp   CI [{lo:+.2f}, {hi:+.2f}]   "
          f"(in-sample overlay {in_obs:+.2f})")

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.hist(boot, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(obs, color="crimson", lw=2, label=f"Observed gap ({obs:+.2f} pp)")
    ax.axvline(lo, color="grey", lw=1.5, ls="--", label=f"95% CI [{lo:+.2f}, {hi:+.2f}]")
    ax.axvline(hi, color="grey", lw=1.5, ls="--")
    ax.axvline(0, color="black", lw=1, ls=":")
    ax.axvline(in_obs, color="darkorange", lw=1.5, ls="-.",
               label=f"In-sample observed ({in_obs:+.2f} pp)")
    ax.set_xlabel("Information-loss gap (pp): %|SHAP|_Filter − %|SHAP|_Ridge in filtered features")
    ax.set_ylabel("Bootstrap frequency")
    ax.set_title("H1 bootstrap distribution — OOS surrogate (Filter vs Ridge)")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    savefig("h2_bootstrap_oos_ci")


# ---------------------------------------------------------------------------
# PNG -> PDF export for every other thesis figure (unchanged content).
# ---------------------------------------------------------------------------
UNCHANGED = [
    "vif_daily_training_data", "vif_all_predictors", "correlation_matrix_training",
    "pca_explained_variance", "pca_loadings", "ridge_cv_curve", "elasticnet_cv_heatmap",
    "sarima_acf_pacf", "sarimax_residual_acf", "stage1_residuals_comparison",
    "sarimax_forecast_vs_actual", "ridge_coefficients", "elasticnet_coefficients",
    "ridge_vs_elasticnet_coefs", "interpretability_coef_stability",
    "interpretability_h2_linear_rmse", "interpretability_h1_stability_accuracy",
]


def export_unchanged() -> None:
    for name in UNCHANGED:
        png = PLOTS / f"{name}.png"
        if not png.exists():
            print(f"  WARNING missing {png.name} — skipped")
            continue
        Image.open(png).convert("RGB").save(PLOTS / f"{name}.pdf",
                                             "PDF", resolution=150.0)
        print(f"  exported    plots/{name}.pdf  (from PNG)")


def main() -> None:
    print("=" * 70)
    print("REGENERATE THESIS FIGURES (OOS H3 + renumbered titles) -> PNG + PDF")
    print("=" * 70)
    print("[Fig 1] coefficient stability by group (title -> H2)")
    fig1_stratified()
    print("[surrogates] fitting OOS (beeswarms/Fig2) + in-sample (Fig20)...")
    X_te, sv_oos, sv_in = compute_surrogates()
    print("[Fig 2] H3 ranking-consistency matrix (OOS)")
    fig2_h3(X_te, sv_oos)
    print("[App O] beeswarms (OOS surrogate)")
    beeswarms(X_te, sv_oos)
    print("[Fig 20] information-loss bootstrap (title -> H1)")
    fig20_bootstrap(X_te, sv_in)
    print("[Fig 21] OOS information-loss bootstrap (title -> H1)")
    fig21_oos_bootstrap(X_te, sv_oos, sv_in)
    print("[export] PNG -> PDF for unchanged thesis figures")
    export_unchanged()
    print("\nDone. All outputs in plots/ (PNG + PDF). No data/ files written.")


if __name__ == "__main__":
    main()
