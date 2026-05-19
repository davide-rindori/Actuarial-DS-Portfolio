# Project 05: Actuarial-Informed Neural Networks
## *Constrained Deep Learning for Multi-Population Mortality Forecasting*

### Project Objective
To develop an alternative internal model for multi-population longevity forecasting that embeds actuarial domain knowledge directly into the neural network training process. This model serves as a **challenger** to both the Li-Lee benchmark and the unconstrained LSTM framework (Project 04), while satisfying the governance, robustness, and explainability requirements expected by regulators (FINMA/EIOPA) for internal model validation.

---

### Research Question
*Can physics-informed actuarial constraints embedded in the training loss improve neural mortality forecasting — particularly for near-linear populations where unconstrained LSTMs show no advantage — while preserving the model's ability to capture non-linear regime shifts?*

---

### The 5 Pillars of Research

#### 1. Constrained Loss Function (AINN Architecture)
Embed actuarial domain knowledge as soft constraints in the training loss:
$$\mathcal{L} = \mathcal{L}_{MSE} + \lambda_1 \mathcal{L}_{coherence} + \lambda_2 \mathcal{L}_{monotonicity} + \lambda_3 \mathcal{L}_{stationarity}$$

- **Coherence penalty**: penalise divergence of country-specific factors from zero (Li-Lee assumption as a regulariser, not a hard constraint).
- **Monotonicity penalty**: penalise $m_{x+1} < m_x$ in reconstructed mortality curves (Gompertzian compliance during training, not post-hoc).
- **Stationarity penalty**: penalise unit-root behaviour in predicted specific factors.
- **Key question**: does constraining the loss improve generalisation, or does it fight the data?

#### 2. Credibility-Weighted Blending & MBC as Bayesian Shrinkage
- Formalise MBC as a regularisation technique in learning-theoretic terms.
- The bias vector $\mathcal{B}$ introduces bias to reduce variance (integration drift).
- Connect to credibility theory: MBC weight = 1 is full credibility to historical drift.
- Frame as Bayesian shrinkage toward the Li-Lee drift prior.
- Explore credibility-weighted blending of neural and classical forecasts.

#### 3. Robustness & Validation (Regulatory Grade)
- **Multi-seed robustness**: run the full pipeline with 5+ seeds, report mean ± std of key metrics.
- **Rolling-window validation**: expanding windows (e.g., train 1956-2005/val 2006-2014, train 1956-2008/val 2009-2017) for more robust performance estimates.
- **Fat-tail process noise**: replace Gaussian $\eta_t$ with Student-t or bootstrap residuals; compare SCR under different distributional assumptions.

#### 4. True Model-Based Stress Testing
- Translate a mortality shock (e.g., -10% on $m_x$) into the $\Delta K_t$ domain.
- Inject into the sliding window at the shock year and re-execute the recursive forecast with MC Dropout.
- Quantify the transient response: does the LSTM amplify, dampen, or neutralise the shock?
- Demonstrates model stability for regulatory model validation (FINMA/EIOPA).

#### 5. Regulatory Governance & Auditability
- Full Model Passport with validation verdicts.
- Biological consistency audit (monotonicity, Gompertz compliance).
- Lexis residual maps for cohort-effect leakage detection.
- SHAP influence mapping for cross-country explainability.
- Reverse stress test and SCR calibration (SST/Solvency II).

---

### Positioning: Challenger Model Framework

| Criterion | Li-Lee (Benchmark) | Project 04 (LSTM+MBC) | Project 05 (AINN) |
|:---|:---|:---|:---|
| Non-linear dynamics | ✗ | ✓ | ✓ |
| Coherence guarantee | ✓ (by construction) | ✗ (implicit) | ✓ (soft constraint) |
| Monotonicity | Not enforced | Post-hoc audit | Training-embedded |
| Uncertainty framework | Parametric (RWD) | Dual (MCD + process) | Dual + fat-tail |
| Robustness evidence | N/A | Single split | Rolling-window + multi-seed |
| Stress test | Level-shift | Level-shift | Model-based (ΔK_t injection) |

---

### Why This Belongs on arXiv
Three original contributions:
1. **Architecture**: First application of PINN-style actuarial constraints to multi-population mortality forecasting.
2. **Theory**: Formalisation of Mean-Bias Correction as Bayesian shrinkage / credibility weighting.
3. **Validation**: Rolling-window + multi-seed robustness protocol for neural mortality models, with fat-tail SCR comparison.

---

### Execution Phases
- [ ] **Phase A**: Data reproduction and Li-Lee baseline (reuse from Project 04).
- [ ] **Phase B**: Constrained loss design and λ-sweep ablation.
- [ ] **Phase C**: Credibility blending and MBC formalisation.
- [ ] **Phase D**: Rolling-window validation and multi-seed robustness.
- [ ] **Phase E**: Model-based stress test and fat-tail uncertainty.
- [ ] **Phase F**: Governance (Model Passport, SHAP, Lexis, Reverse Stress Test).
- [ ] **Phase G**: Paper drafting and arXiv submission.
