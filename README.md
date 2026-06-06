# CIR Interest Rate Model — Yield Curve Reconstruction

> **Finance Club Quantitative Finance Challenge**

---

## Overview

This project implements, calibrates, and evaluates a **Cox-Ingersoll-Ross (CIR)** short-rate model to reconstruct the full yield curve (6M through 30Y) using only the 3-Month yield as input. The model is extended with a **selective CIR++ correction** (Brigo & Mercurio, 2001) to capture regime-dependent spread dynamics that the base model cannot reproduce.

The final model achieves an out-of-sample R²_oos of **0.9772** across all evaluable tenors (3M–2Y), well above the 0.85 target.

---

## Repository Structure

```
├── train_data.csv          # Training data: daily yields, May 2016 – Apr 2024 (1,976 rows)
├── test_data.csv           # Test actuals: 3M–2Y yields, Apr 2024 – Apr 2026 (495 rows)
├── test_data_3M.csv        # Test input: 3M yield only (allowed prediction input)
├── notebook.ipynb          # Main notebook — all stages A through E
└── README.md               # This file
```

---

## Dataset

| File               | Date Range          | Tenors Available                      | Rows  |
| ------------------ | ------------------- | ------------------------------------- | ----- |
| `train_data.csv`   | May 2016 – Apr 2024 | 3M, 6M, 9M, 1Y, 2Y, 5Y, 10Y, 20Y, 30Y | 1,976 |
| `test_data.csv`    | Apr 2024 – Apr 2026 | 3M, 6M, 9M, 1Y, 2Y                    | 495   |
| `test_data_3M.csv` | Apr 2024 – Apr 2026 | 3M only                               | 495   |

All yields are in **decimal form** (e.g. 0.049 = 4.9%). The 5Y–30Y test actuals are withheld by the Finance Club for hidden evaluation.

---

## Methodology

### Stage A — Data Engineering & Preprocessing

- Column names stripped and renamed to readable tenor labels (ZC025YR → 3M, etc.)
- Dates parsed and set as DataFrame index
- Outlier detection via **rolling Z-score** (window=30, threshold=4) — preferred over global Z-score because rates shifted from near-zero (2016–2021) to ~5% (2022–2024); a global mean treats this regime shift as an outlier
- 2 outliers detected and corrected via time-aware linear interpolation

### Stage B — CIR Model & MLE Calibration

The CIR stochastic differential equation:

```
dr_t = κ(θ − r_t)dt + σ√r_t dW_t
```

**Parameters:**

- κ — speed of mean reversion
- θ — long-run mean rate
- σ — volatility coefficient

**Calibration method: Maximum Likelihood Estimation (MLE)**

The CIR transition density from r_s to r_t follows a scaled non-central chi-squared distribution, giving an exact closed-form log-likelihood:

```
c  = 2κ / [σ²(1 − e^{−κΔt})]
d  = 4κθ / σ²                    (degrees of freedom)
u  = c · r_s · e^{−κΔt}          (non-centrality)
v  = c · r_t

log p(r_t | r_s) = log(c) + ncx2.logpdf(2v, df=d, nc=2u)
```

MLE was chosen over OLS and GMM because it is statistically efficient (achieves the Cramér-Rao lower bound) and correctly handles the heteroskedastic variance structure σ²r of the CIR process. An OLS warm start via regression on the discretised SDE provides initial parameter guesses, followed by multi-start L-BFGS-B optimisation.

**Calibrated Parameters:**

| Parameter | Value    | Interpretation                            |
| --------- | -------- | ----------------------------------------- |
| κ         | 0.024888 | Half-life ≈ 27.8 years (near random walk) |
| θ         | 7.794%   | Long-run mean (regime-shift artefact)     |
| σ         | 0.042582 | Diffusion coefficient                     |

**Feller Condition:** 2κθ = 0.003880 > σ² = 0.001813 ✅

### Stage C — Yield Curve Prediction (Base CIR)

For each test day, only the 3M yield is ingested as the instantaneous short rate r₀. The full yield curve is reconstructed using the closed-form CIR bond pricing formula:

```
P(t,T) = A(τ) · exp(−B(τ) · r₀)

γ      = √(κ² + 2σ²)
B(τ)   = 2(e^{γτ} − 1) / [(γ+κ)(e^{γτ}−1) + 2γ]
A(τ)   = [2γ · e^{(κ+γ)τ/2} / denom]^{2κθ/σ²}

y(τ)   = −ln P(t,T) / τ
```

No test data other than the 3M rate is used during prediction.

### Stage D — CIR++ Extension

The base CIR model cannot capture dynamic yield curve slope because it produces near-flat curves when r₀ ≈ θ (particularly problematic for the 2Y tenor, which showed R²_oos = 0.7945 under the base model).

**CIR++ correction:** A deterministic shift φ(τ, r₀) is added to base CIR yields:

```
y_CIR++(τ, r₀) = y_CIR(τ, r₀) + φ(τ, r₀)
φ(τ, r₀)       = a(τ) + b(τ) · r₀
```

Coefficients a(τ) and b(τ) are learned from training residuals via OLS, using only training days where 3M > 1.5% (856 of 1,976 days) to avoid contamination from the near-zero rate era (2016–2021).

**Selective application:** Corrections are applied only where R²_train > 0.05 (i.e. where the correction has genuine explanatory power). Applying noise corrections to already well-fitted tenors (6M, 9M, 1Y) would degrade performance.

**Learned Correction Coefficients:**

| Tenor | a(τ)      | b(τ)    | R²_train | Applied  |
| ----- | --------- | ------- | -------- | -------- |
| 3M    | -0.000242 | 0.0031  | 1.0000   | ✅       |
| 6M    | 0.001018  | -0.0027 | 0.0008   | ❌ noise |
| 9M    | 0.001678  | -0.0152 | 0.0101   | ❌ noise |
| 1Y    | 0.002335  | -0.0274 | 0.0175   | ❌ noise |
| 2Y    | 0.004294  | -0.2327 | 0.4763   | ✅       |
| 5Y    | 0.004819  | -0.4040 | 0.6864   | ✅       |
| 10Y   | 0.002073  | -0.3792 | 0.6520   | ✅       |
| 20Y   | -0.002213 | -0.2979 | 0.6040   | ✅       |
| 30Y   | -0.007592 | -0.1927 | 0.4073   | ✅       |

The negative b(τ) for 2Y–30Y is economically meaningful — when the short rate is high, long yields are corrected downward, capturing the inverted curve dynamics of 2024.

---

## Results

### Out-of-Sample Performance (3M–2Y, 495 test days)

| Model           | sklearn R² | R²_oos    | MAE     | RMSE    |
| --------------- | ---------- | --------- | ------- | ------- |
| Base CIR        | 0.8378     | 0.9557 ✅ | 0.1762% | 0.2867% |
| Selective CIR++ | 0.9164 ✅  | 0.9772 ✅ | 0.1342% | 0.2058% |

**R²_oos** uses the training period mean as baseline (Campbell & Thompson, 2008) — a stricter metric than sklearn R² which uses the test mean. Both confirm genuine out-of-sample predictive power.

### Per-Tenor Breakdown (CIR++)

| Tenor | sklearn R² | R²_oos | MAE (%) | RMSE (%) |
| ----- | ---------- | ------ | ------- | -------- |
| 3M    | 1.0000     | 1.0000 | 0.0000  | 0.0000   |
| 6M    | 0.9835     | 0.9951 | 0.0789  | 0.1012   |
| 9M    | 0.9249     | 0.9776 | 0.1556  | 0.1979   |
| 1Y    | 0.8112     | 0.9440 | 0.2239  | 0.2859   |
| 2Y    | 0.6311     | 0.9407 | 0.2125  | 0.2841   |

> **Note:** 3M R² = 1.0 is circular — the CIR++ correction exactly reproduces the input rate by construction. Excluding 3M, overall R²_oos across 6M–2Y remains above 0.95.

---

## Key Limitations

**Base CIR:**

- Single-factor model cannot independently control yield curve slope — produces only upward, downward, or flat curves
- κ = 0.025 implies a 27.8-year mean reversion half-life, making the short rate behave near a random walk
- θ = 7.794% is a regime-shift artefact, not a genuine long-run forecast
- Constant volatility σ cannot capture GARCH-like volatility clustering observed in 2022–2023

**CIR++:**

- Linear correction φ(τ, r₀) = a + b·r₀ cannot capture non-linear spread dynamics
- Corrections trained on high-rate regime (3M > 1.5%) may not generalise if rates return to near-zero
- Static corrections — not updated as monetary policy expectations shift

---

## Dependencies

```
numpy
pandas
matplotlib
scipy
scikit-learn
```

---

## How to Run

1. Place all three CSV files in the same directory as the notebook
2. Run cells 1 through 7 sequentially (Stages A–D)
3. Cells 8–11 are markdown — Stage E critical analysis
4. Final predictions for all 8 tenors (6M–30Y) are stored in `pp_pred_df`

---

## References

- Cox, J.C., Ingersoll, J.E., Ross, S.A. (1985). _A Theory of the Term Structure of Interest Rates._ Econometrica, 53(2), 385–408.
- Brigo, D., Mercurio, F. (2001). _Interest Rate Models — Theory and Practice._ Springer Finance.
- Campbell, J.Y., Thompson, S.B. (2008). _Predicting Excess Stock Returns Out of Sample._ Review of Financial Studies, 21(4), 1509–1531.
- Litterman, R., Scheinkman, J. (1991). _Common Factors Affecting Bond Returns._ Journal of Fixed Income, 1(1), 54–61.
