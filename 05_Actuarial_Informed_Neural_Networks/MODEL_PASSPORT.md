# Model Passport: Actuarial-Informed Neural Network (AINN)
**Version:** 0.1.0 (Development)  
**Status:** In Development  
**Owner:** Davide Rindori

## 1. Model Identity
- **Model Type:** Constrained Neural-Actuarial (LSTM + Actuarial Loss Penalties + MBC).
- **Primary Use:** Longevity risk assessment, SCR calculation (SST/Solvency II), challenger model validation.
- **Target Population:** High-longevity frontier cluster (CHE, SWE, NOR, DEUTW, NLD, JPN).
- **Predecessor:** Project 04 (Unconstrained LSTM+MBC).

## 2. Methodology Snapshot
- **Architecture:** Stacked LSTM with actuarial-informed loss function.
- **Constraints:** Coherence (Li-Lee), Monotonicity (Gompertz), Stationarity (mean-reversion).
- **Stationarity Strategy:** First Differences ($\Delta K_t$).
- **Anchor Mechanism:** Mean-Bias Correction (MBC), formalised as Bayesian shrinkage.
- **Uncertainty:** Dual framework (MC Dropout + process noise), extended with fat-tail alternatives.

## 3. Governance & Validation (Planned)
### A. Biological Consistency
- **Monotonicity**: Enforced during training (not post-hoc).
- **Gompertz Compliance**: Embedded in loss function.

### B. Statistical Robustness
- **Multi-seed**: 5+ independent training runs.
- **Rolling-window**: Expanding validation windows.
- **Fat-tail**: Student-t and bootstrap process noise.

### C. Stress Testing
- **Model-based**: Shock injection in ΔK_t domain.
- **Reverse stress test**: Critical threshold δ* computation.

## 4. Risk & Capital Metrics
- To be populated after training.

## 5. Comparison with Predecessor (Project 04)
| Property | Project 04 | Project 05 |
|:---|:---|:---|
| Constraints | None (post-hoc audit) | Training-embedded |
| Validation | Single split | Rolling-window + multi-seed |
| Process noise | Gaussian | Gaussian + fat-tail |
| Stress test | Level-shift (post-hoc) | Model-based (ΔK_t injection) |
| MBC interpretation | Empirical correction | Bayesian shrinkage (formalised) |
