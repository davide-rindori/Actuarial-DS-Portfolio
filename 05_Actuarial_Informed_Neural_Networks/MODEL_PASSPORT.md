# Model Passport: Actuarial-Informed Neural Network (AINN)
**Version:** 1.0.0  
**Status:** Validated  
**Owner:** Dr. Davide Rindori

## 1. Model Identity
- **Model Type:** Constrained Neural-Actuarial (Stacked LSTM + Actuarial Loss Penalties).
- **Primary Use:** Sex-specific longevity risk assessment, SCR calculation (SST/Solvency II).
- **Target Population:** High-longevity frontier cluster (CHE, SWE, NOR, DEUTW, NLD, JPN), Male and Female separately.
- **Predecessor:** Project 04 (Unconstrained LSTM+MBC, Total only).

## 2. Architecture
- **Type:** Stacked LSTM (48, 32 units) + Dropout(0.2) per layer.
- **Input:** 15 years of first-difference mortality factors + binary sex indicator (shape: 15 × 8).
- **Output:** 8-dimensional vector (1 common + 6 country-specific + 1 sex indicator).
- **Training:** Joint Male/Female (90 samples), batch_size=8, lr=0.001, early stopping (patience=20).
- **Optimisation:** Optuna TPE, 100 trials, 6-dimensional joint search (lookback, units, lr, λ).

## 3. Constrained Loss Function
$$\mathcal{L} = \mathcal{L}_{MSE} + 0.001 \cdot \mathcal{L}_{coherence} + 0.001 \cdot \mathcal{L}_{monotonicity}$$

- **Coherence:** penalises divergence of country-specific factors from the common trend.
- **Monotonicity (temporal):** penalises positive $\Delta K_t$ (mortality worsening over time).
- **Stationarity:** tested and excluded (hurts performance, inconsistent with data).

## 4. Governance & Validation Verdicts

### A. Predictive Performance
- **Validation RMSE (2012-2020):** 6.1725 (original scale).
- **Male RMSE:** 5.7246. **Female RMSE:** 6.5900.
- **Multi-seed CV:** 8.90% (PASS, threshold < 10%).
- **Rolling-window CV:** 5.78% (PASS, threshold < 20%).

### B. Biological Consistency
- **Gompertz Monotonicity (ages 40-90, 2050):** STRUCTURALLY COMPLIANT.
  - Small violations (< 0.001) inherited from HMD data granularity, not model-generated.
  - The observation-anchored approach preserves data-level irregularities; the model cannot introduce new violations ($B_x \cdot \Delta K_t$ is a uniform shift).

### C. Explainability (XAI)
- **Temporal Saliency:** Distributed importance across the 15-year window (peak at t-3/t-4, ~10%). No pathological concentration.
- **SHAP Influence Mapping:** Netherlands and Japan are top predictors for Swiss male mortality. Distributed cross-country influence hierarchy — no single dominant predictor.

### D. Model Stability
- **Model-Based Stress Test (shock in $\Delta K_t$ domain):** STABLE.
  - Amplification ratio: 1.02× — no explosive feedback.
  - Shock absorbed smoothly; effect stabilises within 10 years.

## 5. Risk & Capital Metrics (2050 Forecast)

### SCR — Full Cluster

| Country | Male SCR (ES 99%) | Female SCR (ES 99%) |
|:---|:---|:---|
| Switzerland | +3.760 yrs | +2.919 yrs |
| Sweden | +3.854 yrs | +3.223 yrs |
| Norway | +3.664 yrs | +3.076 yrs |
| West Germany | +4.474 yrs | +3.526 yrs |
| Netherlands | +4.107 yrs | +3.632 yrs |
| Japan | +3.752 yrs | +2.355 yrs |

### Reverse Stress Test (SST Compliance)

| Country | Male δ* | Female δ* |
|:---|:---|:---|
| Switzerland | 45.3% | 48.5% |
| West Germany | 46.7% | 50.0% |
| Japan | 45.1% | 46.1% |

- **Linearity:** PASS (R² > 0.999 for all countries).
- **10% Shock:** consumes 20-22% of the SCR buffer.

## 6. Key Design Decisions & Rationale

| Decision | Rationale |
|:---|:---|
| Joint M/F training | Doubles sample size (+2.4% RMSE improvement), production-grade sex-specific output |
| Optuna 6D joint tuning | Avoids sequential tuning circularity; finds lookback-architecture interaction |
| Lookback = 15 | Captures full deceleration transition (1997-2011); +12.7% over lb=10 |
| λ = 0.001 | Governance instrument; neutral on RMSE (-0.009%); formally constrained |
| Observation-anchored e₀ | Eliminates rank-1 reconstruction bias; anchors to HMD reality |
| Specific factors fixed at 2020 | Li-Lee stationarity assumption; prevents recursive drift |

## 7. Known Limitations

1. **Constraint effect on RMSE is neutral (-0.009%):** constraints serve governance, not accuracy.
2. **Multi-seed CV = 8.90% (borderline):** Seed 123 outlier. Model deployed with seed 42 only.
3. **Female RMSE > Male RMSE:** Female mortality is intrinsically harder to predict in this cluster.
4. **SCR larger than Project 04:** wider CI from sex-specific decomposition and lb=15 amplification.
5. **Monotonicity surrogate is temporal, not age-based:** true Gompertz in loss deferred to future work.
