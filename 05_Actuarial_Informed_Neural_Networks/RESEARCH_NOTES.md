# Research Notes: Actuarial-Informed Neural Networks

## 1. Motivation & Context

### 1.1 From Project 04: What We Learned
- The unconstrained LSTM+MBC outperforms Li-Lee on 4/6 countries (+17.40% SWE, +12.57% DEUTW, +4.43% NLD, +3.52% NOR).
- It marginally underperforms on Switzerland (-2.08%) and Japan (-1.31%) — the two most linear populations.
- The ablation studies showed that First Differences (+48.6%) and MBC (+18.6%) are the two critical design choices.
- MC Dropout contributes ~1% of total uncertainty; process noise dominates (~99%).
- The model passes all biological consistency tests (Gompertz monotonicity) but only post-hoc — it is not structurally guaranteed.

### 1.2 The Gap This Project Fills
- **Structural guarantees**: Can we embed biological and actuarial constraints *during training* rather than verifying them after the fact?
- **Near-linear populations**: Can constrained training improve performance on CHE and JPN where the unconstrained model slightly loses?
- **Regulatory preference**: Regulators prefer models with built-in safeguards over models that happen to pass audits. A constrained architecture is inherently more defensible.

### 1.3 Connection to Wüthrich's Research Agenda
- Wüthrich's "actuarial learning" framework advocates for combining statistical learning with actuarial structure.
- The AINN approach is the natural multi-population extension of this philosophy.
- The credibility-weighted blending (Notebook 04) directly connects to credibility theory — a core actuarial concept.

---

## 2. Constrained Loss Function Design

### 2.1 General Framework
$$\mathcal{L} = \mathcal{L}_{MSE} + \lambda_1 \mathcal{L}_{coherence} + \lambda_2 \mathcal{L}_{monotonicity} + \lambda_3 \mathcal{L}_{stationarity}$$

### 2.2 Coherence Penalty ($\mathcal{L}_{coherence}$)
- **Rationale**: Li-Lee assumes country-specific factors $k_{t,i}$ are stationary (mean-reverting to zero). The unconstrained LSTM ignores this. We introduce it as a soft penalty.
- **Formulation**: $\mathcal{L}_{coherence} = \frac{1}{N} \sum_i |\hat{k}_{t,i}|^2$ (penalise magnitude of predicted specific factors).
- **Expected effect**: Regularises the model toward cluster coherence without forcing it. If the data genuinely supports divergence, the MSE term will dominate.

### 2.3 Monotonicity Penalty ($\mathcal{L}_{monotonicity}$)
- **Rationale**: Mortality must increase with age (Gompertz law). Project 04 verifies this post-hoc. Here we enforce it during training.
- **Formulation**: After back-transforming predicted $K_t$ and $k_{t,i}$ into $m_x$, penalise any violation: $\mathcal{L}_{monotonicity} = \frac{1}{A} \sum_x \max(0, m_x - m_{x+1})^2$ for ages 40-90.
- **Challenge**: Requires differentiable back-transformation within the training loop. May need to approximate or use a surrogate.

### 2.4 Stationarity Penalty ($\mathcal{L}_{stationarity}$)
- **Rationale**: Penalise unit-root behaviour in predicted specific factors.
- **Formulation**: $\mathcal{L}_{stationarity} = \frac{1}{N} \sum_i (\hat{k}_{t,i} - \hat{k}_{t-1,i})^2$ (penalise large changes, encouraging smooth mean-reversion).
- **Note**: This is the most controversial constraint — Project 04 showed that stationarity is violated in the data for 4/6 countries. The λ sweep will reveal whether this helps or hurts.

### 2.5 Lambda Sweep Strategy
- Test λ values on a logarithmic grid: [0, 0.001, 0.01, 0.1, 1.0].
- λ = 0 recovers the unconstrained LSTM (Project 04 baseline).
- Report RMSE, biological compliance rate, and stationarity metrics for each configuration.
- Identify the Pareto frontier: accuracy vs. constraint satisfaction.

---

## 3. MBC as Bayesian Shrinkage (Theoretical Contribution)

### 3.1 The Observation
MBC adds a constant bias $\mathcal{B}$ to each predicted variation:
$$Level_t = Level_{t-1} + (\Delta_{LSTM} + \mathcal{B})$$

This is equivalent to shrinking the neural prediction toward the Li-Lee drift:
$$\hat{\Delta}_t^{MBC} = \hat{\Delta}_t^{LSTM} + \underbrace{(\mu_{Li-Lee} - \mu_{LSTM})}_{\mathcal{B}}$$

### 3.2 Credibility Theory Interpretation
- Let $Z$ be the credibility weight assigned to the LSTM prediction.
- The blended forecast is: $\hat{\Delta}_t = Z \cdot \hat{\Delta}_t^{LSTM} + (1-Z) \cdot \mu_{Li-Lee}$
- MBC corresponds to $Z = 1$ with an additive correction — full credibility to the neural signal, but anchored to the actuarial prior.
- A generalised version would tune $Z \in [0, 1]$ based on out-of-sample performance.

### 3.3 Bayesian Shrinkage Framing
- The Li-Lee drift $\mu_{Li-Lee}$ acts as the prior mean.
- The LSTM prediction is the likelihood.
- MBC is a point estimate under a specific prior-likelihood weighting.
- This connects directly to Wüthrich's framework of "actuarial learning" as regularised statistical estimation.

---

## 4. Robustness Protocol

### 4.1 Multi-Seed Table
- Train with seeds: [42, 123, 256, 512, 1024].
- Report: RMSE (per country), SCR (ES 99.0%), δ* (reverse stress test).
- Acceptance criterion: CV < 10% across seeds for all key metrics.

### 4.2 Rolling-Window Validation
- Window 1: Train 1956-2005, Validate 2006-2014.
- Window 2: Train 1956-2008, Validate 2009-2017.
- Window 3: Train 1956-2011, Validate 2012-2020 (current standard).
- Report stability of RMSE improvements across windows.

### 4.3 Fat-Tail Process Noise
- Baseline: $\eta_t \sim \mathcal{N}(0, \sigma^2_{Li-Lee})$ (current Project 04 approach).
- Alternative 1: $\eta_t \sim t_\nu(0, \sigma^2)$ with $\nu$ estimated from residuals.
- Alternative 2: Bootstrap resampling of historical residuals.
- Compare: 95% CI width, SCR (ES 99.0%), and tail shape (kurtosis).

---

## 5. True Model-Based Stress Test

### 5.1 Concept
Instead of applying a post-hoc level-shift to $e_0$, translate a mortality shock into the $\Delta K_t$ domain:
1. Define shock: $m_x^{shocked} = (1 - \delta) \cdot m_x$ for all ages.
2. Compute the implied $\Delta K_t^{shock}$ via inverse Li-Lee mapping.
3. Inject $\Delta K_t^{shock}$ into the sliding window at the shock year (e.g., 2026).
4. Re-execute the recursive forecast with MC Dropout.

### 5.2 Expected Behaviour
- The shock appears as a single anomalous $\Delta K_t$ in the 10-year window.
- After ~10 years, the shock exits the window and the model "forgets" it.
- The result should converge to the deterministic level-shift, with a transient dampening/amplification effect.

### 5.3 Regulatory Value
- Demonstrates model stability (no explosive amplification).
- Quantifies the transient response function.
- Shows whether the LSTM introduces non-linear feedback effects.
- Directly addresses FINMA/EIOPA model validation requirements.

---

## 6. Open Questions & Risks

1. **Differentiability of monotonicity penalty**: Back-transforming $K_t \to m_x$ within the training loop may be computationally expensive or numerically unstable. May need a surrogate loss.
2. **Stationarity vs. data**: The ADF/KPSS analysis in Project 04 showed that stationarity is violated for 4/6 countries. The stationarity penalty may hurt performance. The λ sweep is critical.
3. **Overfitting to constraints**: If λ values are too high, the model may satisfy constraints perfectly but lose predictive power. The Pareto frontier analysis will reveal this.
4. **Computational cost**: Rolling-window validation × multi-seed × λ sweep = many training runs. Need to plan compute budget.
