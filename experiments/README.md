# Revision experiments

Standalone scripts that re-implement the notebook protocols (Stage-1 refit + rolling-origin
SARIMAX, SHAP surrogates, etc.) to answer specific reviewer points. Each writes only to the
repo's `data/` and `plots/` directories; the notebook pipeline (`01`–`04`) is the source of the
main-text results, and these scripts feed the **appendices and robustness claims**.

## Running

Each script is portable — it derives the repo root from its own location
(`ROOT = Path(__file__).resolve().parents[1]`), so the working directory does not matter and no
path edits are needed if the repo moves:

```bash
~/anaconda3/envs/machine_learning/bin/python experiments/<script>.py
```

Three scripts import a sibling module (they add `ROOT/"experiments"` to `sys.path` automatically),
so the imported module must be present but need not be run first:

| Script | imports |
|---|---|
| `dm_loss_and_mtc_sensitivity.py` | `dm_correction` |
| `tost_equivalence.py` | `dm_correction` (+ `realistic_forecast` with `--realistic`) |
| `h1_power_mde_refinement.py` | `coef_stability_robustness` |
| `h3_rank_consistency_robustness.py` | `h2_surrogate_robustness` |

The XGBoost/SHAP `TreeExplainer` incompatibility is handled by a `_shap_tree_explainer`
workaround in `fourier_ablation.py` (xgboost pinned to `3.2.0`) — reuse it for any SHAP-on-XGBoost code.

## Script → thesis map

### Forecast accuracy (H1): Diebold–Mariano + information analysis

| Script | Thesis | Key outputs |
|---|---|---|
| `dm_correction.py` | §H1, Fig. (DM grid) | `data/sarimax_dm_corrected.csv`, `plots/dm_holm_grid_corrected.png` — corrected DM (Newey–West LRV + HLN, origin units). Exposes `nw_hln_se` (shared variance helper) and `pct_errors_per_origin` (MAPE-scale per-origin errors) reused by `tost_equivalence.py` |
| `dm_loss_and_mtc_sensitivity.py` | §H1 | `data/dm_loss_mtc_sensitivity.csv` — squared vs abs loss, Holm/Hochberg/BH, block-bootstrap |
| `tost_equivalence.py` | §H1 (non-inferiority) | `data/tost_equivalence_oracle.csv` (+ `data/tost_equivalence_realistic.csv` with `--realistic`) — TOST equivalence test on per-horizon MAPE diff (δ=0.5 pp), reusing the DM `nw_hln_se` variance + df |
| `xgb_surrogate_cv.py` | §3.5 / §H1 info analysis | `data/xgb_surrogate_r2.csv`, `data/xgb_surrogate_shap.csv`, `plots/xgb_surrogate_{r2,shap_compare}.png` — in-sample vs train→test vs CV surrogate |
| `shap_oos_refresh.py` | Fig. 2, App. F (OOS) | `data/fourier_ablation_shap_oos.csv`, `data/interpretability_h2_bootstrap_oos.csv`, `plots/fourier_ablation_shap_oos.png`, `plots/h2_bootstrap_oos_ci.{png,pdf}` |
| `statistical_depth.py` | §3.5 effect sizes | `data/h1_effect_sizes.csv` **[current]**; DM/power outputs **[superseded]** by `dm_correction.py` and `h1_power_mde_refinement.py` (see script header) |

### Coefficient stability (H2)

| Script | Thesis | Key outputs |
|---|---|---|
| `coef_stability_robustness.py` | §H2 | `data/h1_cv_scheme_robustness.csv`, `data/h1_dimension_correlation.csv`, `data/h1_composite_scorecard.csv`, `data/h1_lasso_feature_loss.csv` — expanding/blocked/fixed-roll CV schemes, composite scorecard |
| `h1_power_mde_refinement.py` | §3.5 / §H2 | `data/h1_mde_refined.csv`, `data/h1_global_holm_check.csv`, `plots/h1_power_curve_refined.png` — refined MDE band, union Holm |
| `h2_surrogate_robustness.py` | §H2 | `data/h2_robustness_{gap,xgb_tuning,unique_signal,h3}.csv` — OOS-SHAP gap sign, Fourier decomposition, XGB tuning, interactions, block bootstrap |

### Ranking consistency (H3)

| Script | Thesis | Key outputs |
|---|---|---|
| `h3_rank_consistency_robustness.py` | §H3 | `data/h3_rank_consistency_robustness.csv`, `data/h3_concordance.csv` — Spearman ρ / Kendall τ_b thresholds, Holm, Kendall's W permutation test |

### Data, protocol & order-selection robustness

| Script | Thesis | Key outputs |
|---|---|---|
| `fourier_ablation.py` | App. F | `data/fourier_ablation_{mape,cv,stage1,shap}.csv`, `plots/fourier_ablation_{mape,shap}.png` — with/without annual Fourier (k=2) |
| `realistic_forecast.py` | App. E, §3.4 (Tables 4–5) | `data/sarimax_results_realistic.csv`, `data/realistic_vs_oracle_delta.csv`, `plots/realistic_vs_oracle_mape.{png,pdf}` — non-oracle exogenous protocol |
| `realistic_protocol_sensitivity.py` | §3.4 / notebook 03 | `data/realistic_protocol_sensitivity.csv`, `plots/realistic_protocol_sensitivity.png` — realistic-protocol variants V0–V4 |
| `stage1_refit_sensitivity.py` | notebook 03 | `data/stage1_refit_sensitivity.csv`, `data/stage1_coef_drift.csv` — Stage-1 refit (expanding, every 90d) vs static; coefficient drift |
| `interpolation_sensitivity.py` | App. G | `data/interpolation_sensitivity_{cv,mape,summary}.csv` — linear vs forward-fill GDP/population upsampling |
| `order_selection_robustness.py` | §3.4 / §5.3 | `data/order_grid_expanded.csv`, `data/order_per_strategy{,_grid}.csv`, `data/order_sensitivity_mape.csv` — per-strategy SARIMA order search |
| `residual_diagnostics_depth.py` | §3.4 / §5.3 | `data/residual_diagnostics_{lb,acf,arch}.csv` — Ljung–Box effect sizes, white-noise ACF band, ARCH-LM |
