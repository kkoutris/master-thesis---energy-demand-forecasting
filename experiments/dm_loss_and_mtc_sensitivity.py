"""
Experiment 12 — DM-test loss function, multiple-testing and dependence
sensitivity (three review points on the DM layer).

Builds on the corrected DM machinery in experiments/dm_correction.py
(per-origin absolute errors, Newey-West LRV in origin units + HLN(m)).

  1. Loss function: the headline DM uses d_t = |e_A| - |e_B| (matches MAPE,
     the accuracy metric the thesis makes claims about). Sensitivity: rerun
     all 42 pair x horizon cells with SQUARED loss d_t = e_A^2 - e_B^2 and
     report verdict flips. Squared loss emphasises large (winter) errors —
     relevant given the strong ARCH found in the residuals.
  2. Multiple-testing correction: Holm controls FWER under arbitrary
     dependence but is conservative when tests are positively correlated
     (within a pair, all 6 horizons share the same 50 origins). Sensitivity:
     apply Hochberg (valid under non-negative dependence) and BH-FDR, both
     within-pair (x6) and jointly (x42), and count survivors per method.
  3. Weak-dependence/stationarity of d_t at long horizons: report the
     origin-unit autocorrelations of d_t (the NW truncation should cover
     them), a first-half vs second-half mean shift check, and a circular
     block-bootstrap p-value (block length = overlap depth m = ceil(h/15),
     B = 5000) as an inference route that does not rely on the NW/HLN
     asymptotics.

Outputs:
  - data/dm_loss_mtc_sensitivity.csv   per pair x horizon: p_abs, p_sq,
        p_bootstrap, d_t autocorrelations, half-sample means, and
        Holm/Hochberg/BH survival flags
"""
from __future__ import annotations

import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import dm_correction as dmc  # noqa: E402  (build_strategies, abs_errors_per_origin, dm_corrected)

from statsmodels.stats.multitest import multipletests  # noqa: E402

HORIZONS = dmc.HORIZONS
PAIRS = dmc.PAIRS
ALPHA = 0.05
B_BOOT = 5000
SEED = 42


def block_bootstrap_p(d, m, B=B_BOOT, rng=None):
    """Two-sided circular block-bootstrap test of H0: E[d]=0.
    Block length = overlap depth m (>=1)."""
    rng = rng or np.random.default_rng(SEED)
    n = len(d)
    d_bar = d.mean()
    dc = d - d_bar                       # centred: imposes H0
    L = max(m, 1)
    n_blocks = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=(B, n_blocks))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]) % n
    samples = dc[idx.reshape(B, -1)[:, :n]]
    means = samples.mean(axis=1)
    return float((np.abs(means) >= abs(d_bar)).mean())


def autocorrs(d, max_lag):
    dc = d - d.mean()
    denom = float(dc @ dc)
    if denom == 0:
        return [np.nan] * max_lag
    return [float(dc[k:] @ dc[:-k]) / denom for k in range(1, max_lag + 1)]


def main():
    print("=" * 70)
    print("EXPERIMENT 12 — DM loss / multiple-testing / dependence sensitivity")
    print("=" * 70)

    print("\nRegenerating per-origin absolute errors (dm_correction protocol)...")
    strategies = dmc.build_strategies()
    errors = {}
    for name, s in strategies.items():
        t0 = time.time()
        errors[name] = dmc.abs_errors_per_origin(s)
        print(f"  {name:<8} [{time.time()-t0:.1f}s]")

    rng = np.random.default_rng(SEED)
    rows = []
    for (a, b) in PAIRS:
        for h in HORIZONS:
            n = min(len(errors[a][h]), len(errors[b][h]))
            e1, e2 = errors[a][h][:n], errors[b][h][:n]

            dm_abs, p_abs, m, L = dmc.dm_corrected(e1, e2, h)
            dm_sq, p_sq, _, _ = dmc.dm_corrected(e1 ** 2, e2 ** 2, h)

            d = e1 - e2
            ac = autocorrs(d, max_lag=12)
            p_boot = block_bootstrap_p(d, m, rng=rng)
            half = n // 2
            rows.append({
                "pair": f"{a} vs {b}", "horizon": h, "n": n,
                "m_overlap": m, "nw_truncation": L,
                "dm_abs": round(dm_abs, 4), "p_abs": round(p_abs, 4),
                "dm_sq": round(dm_sq, 4), "p_sq": round(p_sq, 4),
                "p_bootstrap": round(p_boot, 4),
                "d_ac1": round(ac[0], 3),
                "d_ac_max_to_m": round(max(np.abs(ac[:max(m - 1, 1)])), 3),
                "d_mean_half1": round(float(d[:half].mean()), 1),
                "d_mean_half2": round(float(d[half:].mean()), 1),
            })
    df = pd.DataFrame(rows)

    # Sanity: p_abs must match the saved corrected DM table
    ref = pd.read_csv(ROOT / "data" / "sarimax_dm_corrected.csv")
    mrg = df.merge(ref, left_on=["pair", "horizon"],
                   right_on=["pair", "horizon_days"])
    worst = (mrg["p_abs"] - mrg["p_corrected"]).abs().max()
    print(f"\nSanity: p_abs vs sarimax_dm_corrected.csv — max |Δ| = {worst:.4f} "
          f"({'OK' if worst < 1e-6 else 'MISMATCH'})")

    # Multiple-testing corrections on each loss's p-values
    for loss in ("abs", "sq"):
        pcol = f"p_{loss}"
        for method, tag in (("holm", "holm"), ("simes-hochberg", "hochberg"),
                            ("fdr_bh", "bh")):
            within = np.full(len(df), np.nan)
            for pair, sub in df.groupby("pair"):
                rej, p_adj, _, _ = multipletests(sub[pcol].values, alpha=ALPHA,
                                                 method=method)
                within[sub.index.values] = p_adj
            df[f"p_{loss}_{tag}_within"] = np.round(within, 4)
            _, p_joint, _, _ = multipletests(df[pcol].values, alpha=ALPHA,
                                             method=method)
            df[f"p_{loss}_{tag}_joint"] = np.round(p_joint, 4)

    df.to_csv(ROOT / "data" / "dm_loss_mtc_sensitivity.csv", index=False)

    # ---------------- Summary ----------------
    print("\n[1] Loss function: absolute vs squared — verdicts at raw 5%:")
    flips = df[(df.p_abs < ALPHA) != (df.p_sq < ALPHA)]
    print(f"  raw-significant (abs): {(df.p_abs < ALPHA).sum()} of 42; "
          f"(squared): {(df.p_sq < ALPHA).sum()} of 42; verdict flips: {len(flips)}")
    if len(flips):
        print(flips[["pair", "horizon", "p_abs", "p_sq"]].to_string(index=False))

    print("\n[2] Survivors per multiple-testing method (alpha=0.05):")
    print(f"  {'method':<28}{'abs loss':>10}{'sq loss':>10}")
    for tag, label in (("holm", "Holm"), ("hochberg", "Hochberg"), ("bh", "BH-FDR")):
        for fam in ("within", "joint"):
            na = (df[f"p_abs_{tag}_{fam}"] < ALPHA).sum()
            ns = (df[f"p_sq_{tag}_{fam}"] < ALPHA).sum()
            print(f"  {label + ' (' + fam + ')':<28}{na:>10}{ns:>10}")

    print("\n[3] d_t dependence/stationarity diagnostics (h >= 30):")
    long_h = df[df.horizon >= 30]
    print(long_h[["pair", "horizon", "m_overlap", "d_ac1", "d_ac_max_to_m",
                  "p_abs", "p_bootstrap", "d_mean_half1", "d_mean_half2"]]
          .to_string(index=False))
    agree = ((df.p_abs < ALPHA) == (df.p_bootstrap < ALPHA)).mean()
    print(f"\n  NW/HLN vs block-bootstrap verdict agreement: {agree:.0%} of 42 cells")
    boot_sig = df[df.p_bootstrap < ALPHA]
    print(f"  bootstrap-significant cells: {len(boot_sig)}")
    if len(boot_sig):
        print(boot_sig[["pair", "horizon", "p_abs", "p_bootstrap"]].to_string(index=False))

    print("\nSaved: data/dm_loss_mtc_sensitivity.csv")


if __name__ == "__main__":
    main()
