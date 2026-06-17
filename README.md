# Energy Demand Forecasting — Netherlands

**MSc Data Science Thesis · University of Amsterdam**

> *Investigating Climate and Economic Predictors for Energy Demand Forecasting*

This repository contains the analysis pipeline for the thesis, which studies how multicollinearity mitigation strategies (correlation filtering, PCA, Ridge, ElasticNet) affect both the forecast accuracy and interpretability of SARIMAX models for Dutch electricity demand. A two-stage SARIMAX approach is used to isolate the contribution of climate and economic predictors from the underlying temporal dynamics, evaluated across six forecast horizons (H ∈ {1, 3, 7, 30, 90, 180} days).

---

## Repository Layout

```
.
├── EDA.ipynb                          # Exploratory data analysis
├── 01_feature_engineering.ipynb       # Build daily modelling dataset (2009–2025)
├── 02_predictor_strategies.ipynb      # Five predictor strategies (OLS-All, Filter, PCA, Ridge, ElasticNet)
├── 03_sarimax_models.ipynb            # Two-stage SARIMAX fitting & rolling-origin forecasting
├── 04_interpretability.ipynb          # H1 / H2 / H3 hypothesis tests (thesis core)
│
├── data/                              # Intermediate pipeline artifacts (CSV/JSON)
├── plots/                             # Generated figures
├── raw_data/                          # Raw climate and economic source files
├── loadConsumption/                   # Raw load data (TenneT/ENTSO-E)
│   └── preprocess_ENTSO-E.ipynb       #   + load preprocessing notebook
└── experiments/                       # Revision robustness scripts (see experiments/README.md)
```

The four numbered notebooks run sequentially — each saves outputs to `data/` consumed by the next; any single notebook can be re-run in isolation as long as its upstream `data/` files are present. The standalone scripts in `experiments/` re-implement the notebook protocols to answer specific reviewer points (Fourier ablation, order-selection robustness, realistic exogenous forecasting, statistical depth, etc.); `experiments/README.md` maps each script to its thesis section and outputs.

---

## Setup

**Python 3.10+ recommended** (developed on the `machine_learning` conda env, Python 3.10.18).

```bash
pip install -r requirements.txt
jupyter lab
```

The `experiments/` scripts run standalone from the repo root, e.g.:

```bash
python experiments/order_selection_robustness.py
```
