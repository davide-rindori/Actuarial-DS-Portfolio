# Model Passport: Hierarchical LSTM Mortality Framework
**Version:** 1.0.0 (Pre-print Edition)  
**Status:** Validated & Audited  
**Owner:** Davide Rindori

## 1. Model Identity
- **Model Type:** Hybrid Neural-Actuarial (Hierarchical LSTM + Mean-Bias Correction).
- **Primary Use:** Longevity risk assessment, SCR calculation (SST/Solvency II), and Longevity Swap pricing.
- **Target Population:** High-longevity frontier cluster (CHE, SWE, NOR, DEUTW, NLD, JPN).

## 2. Methodology Snapshot
- **Architecture:** Stacked LSTM (32, 16 units) with Bayesian Uncertainty (MC Dropout).
- **Stationarity Strategy:** First Differences ($\Delta K_t$) to neutralize linear drift.
- **Anchor Mechanism:** Mean-Bias Correction (MBC) to align neural variations with long-term actuarial trends.

## 3. Governance & Validation Verdicts
### A. Biological Consistency
- **Monotonicity Test:** PASS (Confirmed Gompertzian compliance for ages 40-90).
- **Monotonicity FAIL (Youth):** Identified in CHE/NOR for ages 20-30; deemed non-material for longevity risk.

### B. Statistical Robustness
- **Out-of-Sample Performance:** +21.88% RMSE improvement over Li-Lee in Japan.
- **Residual Analysis:** Lexis Map confirmed zero cohort-effect leakage.
- **XAI Audit:** SHAP analysis verified West Germany as the primary leading indicator for Swiss mortality.

## 4. Risk & Capital Metrics (2050 Forecast)
- **Switzerland (CHE) SCR (ES 99.0%):** +0.089 years.
- **Japan (JPN) SCR (ES 99.0%):** +0.088 years.
- **Observation:** Risk convergence identified at the longevity frontier.

## 5. Model Robustness & Sensitivity (Clean Run Audit)
- **Lookback Optimization**: Sensitivity analysis (Fig 19) revealed that RMSE decreases as the lookback window increases (5y: 7.14, 10y: 7.05, 15y: 6.91). 
- **Design Decision**: A 10-year lookback was selected as the standard. While a 15-year window offers marginal precision gains (+2% over 10y), it significantly reduces the available training sample size. The 10y window provides the optimal balance between capturing deep temporal dependencies and maintaining statistical volume for the 6-country cluster.
- **Inertia Analysis**: The model effectively filters short-term noise by leveraging a bimodal memory profile (t-1 and t-8), as confirmed by SHAP temporal saliency.