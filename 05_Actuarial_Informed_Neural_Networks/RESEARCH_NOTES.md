# Research Notes: Actuarial-Informed Neural Networks

## 1. Motivation & Context

### 1.1 From Project 04: What We Learned
- The unconstrained LSTM+MBC outperforms Li-Lee on 4/6 countries (+17.40% SWE, +12.57% DEUTW, +4.43% NLD, +3.52% NOR).
- It marginally underperforms on Switzerland (-2.08%) and Japan (-1.31%) — the two most linear populations in the cluster.
- The ablation studies showed that First Differences (+48.6%) and MBC (+18.6%) are the two critical design choices.
- MC Dropout contributes ~1% of total uncertainty; process noise dominates (~99%).
- The model passes all biological consistency tests (Gompertz monotonicity) but only post-hoc — it is not structurally guaranteed.

### 1.2 The Gap This Project Fills
- **Structural guarantees**: Can we embed biological and actuarial constraints *during training* rather than verifying them after the fact?
- **Near-linear populations**: Can constrained training improve performance on CHE and JPN where the unconstrained model slightly loses?
- **Regulatory preference**: Regulators prefer models with built-in safeguards over models that happen to pass audits. A constrained architecture is inherently more defensible for internal model validation (FINMA/EIOPA).
- **Sex-specific modelling**: Project 04 uses both-sexes-combined data. L&H products (annuities, pension buy-outs, longevity swaps) require M/F separation. This project addresses that gap.

### 1.3 Connection to Wüthrich's Research Agenda
- Wüthrich's "actuarial learning" framework advocates for combining statistical learning with actuarial structure.
- The AINN approach is the natural multi-population extension of this philosophy.
- The credibility-weighted blending (Notebook 04) directly connects to credibility theory — a core actuarial concept.
- The formalisation of MBC as Bayesian shrinkage speaks the language of regularised statistical estimation.

---

## 2. Data & Preprocessing (Notebook 01)

### 2.1 Design Decisions
- **Same cluster as Project 04**: CHE, SWE, NOR, DEUTW, NLD, JPN (1956-2020, ages 0-90). This ensures direct comparability of results.
- **Sex-specific**: Unlike Project 04 (Total only), we load Female, Male, and Total columns from HMD. This doubles the modelling pipeline but enables production-grade applicability.
- **Same epsilon** ($10^{-10}$) for log-stability, same age cap at 90, same time window truncation rationale (DEUTW availability).

### 2.2 Notation Consistency with Project 04
We maintain identical notation throughout:
- $m_{x,t}$: Central death rate at age $x$, year $t$.
- $\ln(m_{x,t})$: Log-mortality (input to SVD).
- $a_x$: Mean log-mortality over time (age profile / biological baseline).
- $b_x$: Age sensitivity (first left singular vector, normalised: $\sum_x b_x = 1$).
- $k_t$: Mortality index / time trend (first right singular vector, scaled).
- $B_x$, $K_t$: Li-Lee Common Factor components.
- $b_{x,i}$, $k_{t,i}$: Country-specific components.
- $\Delta K_t$, $\Delta k_{t,i}$: First differences (annual changes).

The only extension is the sex subscript $s \in \{M, F\}$, which we add where needed: $m_{x,t,i,s}$, $K_t^{(s)}$, $k_{t,i}^{(s)}$.

### 2.3 Results: Data Validation
- All 6 countries loaded successfully: 65 years × 91 ages = 5,915 rows each.
- Zero NaN values after cleaning (epsilon applied to rare zero-death cells).
- Matrix dimensions: (91, 65) for each country × sex combination.
- Log-mortality range (CHE Male): [-23.03, -1.10] — consistent with Project 04.

---

## 3. Actuarial Benchmarking: Li-Lee Sex-Specific (Notebook 02)

### 3.1 Methodology
The implementation follows exactly the same procedure as Project 04 (Notebook 02), applied independently to Male and Female:
1. Lee-Carter independent fit via SVD for each country × sex.
2. Li-Lee Common Factor from the cluster average matrix (per sex).
3. Country-Specific Factors from the residual after removing the common component.
4. Stationarity tests (ADF + KPSS) on specific factors.
5. First Differences for AINN input.

### 3.2 Results: Common Factor $K_t$

| Sex | $K_t$ Range | Observation |
|:---|:---|:---|
| Male | [-73.04, +51.75] | Steeper decline, especially post-1970 |
| Female | [-69.78, +61.71] | Shallower decline, higher starting point |

**Interpretation**: The male common trend declines more steeply than the female one. This is consistent with the well-documented "male mortality convergence" phenomenon: male mortality improved faster than female mortality over the second half of the 20th century, driven primarily by reductions in cardiovascular disease and smoking-related deaths. The gap between M and F common trends narrows post-2000, consistent with the literature on the closing of the sex-mortality differential.

**Comparison with Project 04**: Project 04 reported a single $K_t$ (Total) ranging approximately [-60, +50]. The sex-specific decomposition reveals that the "Total" trend is a weighted average of a steeper male decline and a shallower female decline — as expected.

### 3.3 Results: Country-Specific Factors $k_{t,i}$

**Male:**
| Country | $k_{t,i}$ Range | Pattern |
|:---|:---|:---|
| CHE | [-4.21, +2.44] | Stable, near zero |
| SWE | [-0.18, +0.02] | Extremely stable |
| NOR | [-31.53, +7.09] | Highly volatile |
| DEUTW | [-1.21, +3.80] | Moderate, stable |
| NLD | [-14.12, +17.01] | Volatile |
| JPN | [-14.05, +31.23] | Large initial trend (catch-up) |

**Female:**
| Country | $k_{t,i}$ Range | Pattern |
|:---|:---|:---|
| CHE | [-7.37, +2.54] | Slightly more volatile than Male |
| SWE | [-7.26, +1.66] | More volatile than Male (unexpected) |
| NOR | [-27.82, +8.89] | Volatile (less than Male) |
| DEUTW | [-1.54, +1.89] | Stable |
| NLD | [-16.14, +24.78] | Volatile |
| JPN | [-17.65, +44.97] | Largest range in cluster |

**Key observations:**
1. **Japan's catch-up is more pronounced for females** (range 62.6 vs 45.3 for males). This reflects the extraordinary gains in female life expectancy in Japan from the 1960s onwards — Japan went from below-average female mortality to the world's highest female life expectancy.
2. **Sweden is more volatile for females than males** (range 8.9 vs 0.2). This is unexpected and warrants investigation. One hypothesis: Swedish female mortality improvements decelerated earlier than male ones, creating a specific-factor drift that the common trend does not capture.
3. **Norway remains the most volatile** for both sexes, consistent with Project 04's finding that small populations exhibit higher variance in rank-1 SVD models.
4. **Switzerland and West Germany remain stable** for both sexes — these are the "core" populations where the common trend captures most of the signal.

**Comparison with Project 04**: The patterns are qualitatively identical to Project 04 (Total). Japan's catch-up, Norway's volatility, and European stability are all reproduced. The sex split reveals additional structure (Sweden Female, Japan Female) that was masked in the combined data.

### 3.4 Stationarity Analysis

#### Results: Male

| Country | ADF p-value | ADF | KPSS p-value | KPSS | Status |
|:---|:---|:---|:---|:---|:---|
| **Switzerland** | 0.0002 | **PASS** | 0.1000 | **PASS** | **Stationary** |
| **Sweden** | 0.0000 | **PASS** | 0.1000 | **PASS** | **Stationary** |
| **Norway** | 0.0013 | **PASS** | 0.0288 | **FAIL** | Conflict |
| **West Germany** | 0.9987 | **FAIL** | 0.0100 | **FAIL** | **Unit Root** |
| **Netherlands** | 0.9158 | **FAIL** | 0.0100 | **FAIL** | **Unit Root** |
| **Japan** | 0.1704 | **FAIL** | 0.0654 | **PASS** | Conflict |

#### Results: Female

| Country | ADF p-value | ADF | KPSS p-value | KPSS | Status |
|:---|:---|:---|:---|:---|:---|
| **Switzerland** | 0.9587 | **FAIL** | 0.1000 | **PASS** | Conflict |
| **Sweden** | 0.0007 | **PASS** | 0.1000 | **PASS** | **Stationary** |
| **Norway** | 0.0576 | **FAIL** | 0.0581 | **PASS** | Conflict |
| **West Germany** | 0.8411 | **FAIL** | 0.0100 | **FAIL** | **Unit Root** |
| **Netherlands** | 0.8369 | **FAIL** | 0.0100 | **FAIL** | **Unit Root** |
| **Japan** | 0.0221 | **PASS** | 0.0100 | **FAIL** | Conflict |

#### Comparison with Project 04 (Total, both sexes combined)

| Country | Project 04 (Total) | Project 05 (Male) | Project 05 (Female) |
|:---|:---|:---|:---|
| **Switzerland** | Conflict (ADF FAIL, KPSS PASS) | **Stationary** | Conflict |
| **Sweden** | Unit Root (ADF FAIL, KPSS FAIL) | **Stationary** | **Stationary** |
| **Norway** | Stationary (ADF PASS, KPSS PASS) | Conflict | Conflict |
| **West Germany** | Unit Root | **Unit Root** | **Unit Root** |
| **Netherlands** | Unit Root | **Unit Root** | **Unit Root** |
| **Japan** | Conflict (ADF PASS, KPSS FAIL) | Conflict | Conflict |

#### Discussion

This is a significant finding. The sex-specific decomposition reveals a **fundamentally different stationarity landscape** compared to Project 04:

1. **Switzerland Male is stationary** (both tests agree, p < 0.001). In Project 04 (Total), Switzerland was in the "Conflict" zone. This suggests that the non-stationarity in the combined data was driven primarily by the female component. For the AINN, this means the coherence penalty is *appropriate* for Swiss males but may fight the data for Swiss females.

2. **Sweden Male is stationary** (ADF p = 0.0000). In Project 04, Sweden was classified as "Unit Root" — the strongest non-stationarity verdict in the cluster. The sex split reveals that this was driven entirely by the female component (or by the aggregation artefact of combining two differently-behaved series). Swedish males revert to the common trend; Swedish females do not diverge either (also stationary). This contradicts the Project 04 finding and suggests that the "both sexes combined" aggregation introduced a spurious unit root.

3. **Norway flips from Stationary to Conflict.** In Project 04, Norway was the cleanest stationary case. With sex-specific data, both Male and Female show conflict (ADF and KPSS disagree). The high volatility of Norwegian specific factors (small population effect) makes the tests less decisive when applied to sex-specific subsamples with even smaller effective sample sizes.

4. **West Germany and Netherlands remain Unit Root** for both sexes. This is the most robust finding — consistent across Project 04 and both sexes here. These countries have persistent structural drifts that no aggregation can mask.

5. **Japan remains in the Conflict zone** for both sexes, consistent with Project 04. The "Trend-Stationary" interpretation holds: Japan's mean is drifting away from the cluster, but with enough momentum to occasionally fool the ADF test.

#### Implications for the AINN

- The coherence penalty ($\mathcal{L}_{coherence}$) should be **sex-specific in its expected effect**: it is well-justified for Male (where 2/6 countries are genuinely stationary) but more controversial for Female (where only 1/6 is clearly stationary).
- The λ sweep should be run independently for Male and Female to identify potentially different optimal configurations.
- The finding that sex-specific decomposition changes the stationarity landscape is itself a publishable observation — it demonstrates that "both sexes combined" analysis can mask important structural differences.

### 3.5 First Differences and Feature Matrix
- Feature matrix shape: (64, 7) for each sex — 64 annual changes (from 65 years), 7 columns (1 common + 6 specific).
- This is the direct input for the AINN sliding-window training, identical in structure to Project 04.
- All parameters persisted to `li_lee_params.pkl` for downstream use.

---

## 4. Constrained Loss Function Design (Planned: Notebook 03)

### 4.1 General Framework
$$\mathcal{L} = \mathcal{L}_{MSE} + \lambda_1 \mathcal{L}_{coherence} + \lambda_2 \mathcal{L}_{monotonicity} + \lambda_3 \mathcal{L}_{stationarity}$$

### 4.2 Coherence Penalty ($\mathcal{L}_{coherence}$)
- **Rationale**: Li-Lee assumes country-specific factors $k_{t,i}$ are stationary (mean-reverting to zero). The unconstrained LSTM ignores this. We introduce it as a soft penalty.
- **Formulation**: $\mathcal{L}_{coherence} = \frac{1}{N} \sum_i |\hat{k}_{t,i}|^2$ (penalise magnitude of predicted specific factors).
- **Expected effect**: Regularises the model toward cluster coherence without forcing it. If the data genuinely supports divergence, the MSE term will dominate.
- **Connection to results**: Given that 4/6 countries show non-stationary specific factors, this penalty is expected to *hurt* raw RMSE but *improve* long-term forecast stability and biological plausibility.

### 4.3 Monotonicity Penalty ($\mathcal{L}_{monotonicity}$)
- **Rationale**: Mortality must increase with age (Gompertz law). Project 04 verifies this post-hoc. Here we enforce it during training.
- **Formulation**: After back-transforming predicted $K_t$ and $k_{t,i}$ into $m_x$, penalise any violation: $\mathcal{L}_{monotonicity} = \frac{1}{A} \sum_x \max(0, \ln m_x - \ln m_{x+1})^2$ for ages 40-90.
- **Challenge**: Requires differentiable back-transformation within the training loop. May need to approximate or use a surrogate.
- **Note on notation**: We operate in log-space, so the penalty is on $\ln(m_x) > \ln(m_{x+1})$, which is equivalent to $m_x > m_{x+1}$ given the monotonicity of the logarithm.

### 4.4 Stationarity Penalty ($\mathcal{L}_{stationarity}$)
- **Rationale**: Penalise unit-root behaviour in predicted specific factors.
- **Formulation**: $\mathcal{L}_{stationarity} = \frac{1}{N} \sum_i (\hat{k}_{t,i} - \hat{k}_{t-1,i})^2$ (penalise large changes, encouraging smooth mean-reversion).
- **Note**: This is the most controversial constraint. Project 04 showed that stationarity is violated in the data for 4/6 countries. The λ sweep will reveal whether this helps or hurts.
- **Hypothesis**: For near-linear populations (CHE, JPN), the stationarity penalty may improve performance by preventing the LSTM from overfitting to noise. For non-linear populations (SWE, DEUTW), it may hurt by suppressing genuine structural signals.

### 4.5 Lambda Sweep Strategy
- Test λ values on a logarithmic grid: [0, 0.001, 0.01, 0.1, 1.0].
- λ = 0 recovers the unconstrained LSTM (Project 04 baseline).
- Report RMSE, biological compliance rate, and stationarity metrics for each configuration.
- Identify the Pareto frontier: accuracy vs. constraint satisfaction.

---

## 5. MBC as Bayesian Shrinkage (Theoretical Contribution)

### 5.1 The Observation
MBC adds a constant bias $\mathcal{B}$ to each predicted variation:
$$Level_t = Level_{t-1} + (\Delta_{LSTM} + \mathcal{B})$$

This is equivalent to shrinking the neural prediction toward the Li-Lee drift:
$$\hat{\Delta}_t^{MBC} = \hat{\Delta}_t^{LSTM} + \underbrace{(\mu_{Li-Lee} - \mu_{LSTM})}_{\mathcal{B}}$$

### 5.2 Credibility Theory Interpretation
- Let $Z$ be the credibility weight assigned to the LSTM prediction.
- The blended forecast is: $\hat{\Delta}_t = Z \cdot \hat{\Delta}_t^{LSTM} + (1-Z) \cdot \mu_{Li-Lee}$
- MBC corresponds to $Z = 1$ with an additive correction — full credibility to the neural signal, but anchored to the actuarial prior.
- A generalised version would tune $Z \in [0, 1]$ based on out-of-sample performance.

### 5.3 Bayesian Shrinkage Framing
- The Li-Lee drift $\mu_{Li-Lee}$ acts as the prior mean.
- The LSTM prediction is the likelihood.
- MBC is a point estimate under a specific prior-likelihood weighting.
- This connects directly to Wüthrich's framework of "actuarial learning" as regularised statistical estimation.

---

## 6. Robustness Protocol (Planned: Notebook 05)

### 6.1 Multi-Seed Table
- Train with seeds: [42, 123, 256, 512, 1024].
- Report: RMSE (per country), SCR (ES 99.0%), δ* (reverse stress test).
- Acceptance criterion: CV < 10% across seeds for all key metrics.

### 6.2 Rolling-Window Validation
- Window 1: Train 1956-2005, Validate 2006-2014.
- Window 2: Train 1956-2008, Validate 2009-2017.
- Window 3: Train 1956-2011, Validate 2012-2020 (current standard).
- Report stability of RMSE improvements across windows.

### 6.3 Fat-Tail Process Noise
- Baseline: $\eta_t \sim \mathcal{N}(0, \sigma^2_{Li-Lee})$ (current Project 04 approach).
- Alternative 1: $\eta_t \sim t_\nu(0, \sigma^2)$ with $\nu$ estimated from residuals.
- Alternative 2: Bootstrap resampling of historical residuals.
- Compare: 95% CI width, SCR (ES 99.0%), and tail shape (kurtosis).

---

## 7. True Model-Based Stress Test (Planned: Notebook 06)

### 7.1 Concept
Instead of applying a post-hoc level-shift to $e_0$, translate a mortality shock into the $\Delta K_t$ domain:
1. Define shock: $m_x^{shocked} = (1 - \delta) \cdot m_x$ for all ages.
2. Compute the implied $\Delta K_t^{shock}$ via inverse Li-Lee mapping.
3. Inject $\Delta K_t^{shock}$ into the sliding window at the shock year (e.g., 2026).
4. Re-execute the recursive forecast with MC Dropout.

### 7.2 Expected Behaviour
- The shock appears as a single anomalous $\Delta K_t$ in the 10-year window.
- After ~10 years, the shock exits the window and the model "forgets" it.
- The result should converge to the deterministic level-shift, with a transient dampening/amplification effect.

### 7.3 Regulatory Value
- Demonstrates model stability (no explosive amplification).
- Quantifies the transient response function.
- Shows whether the LSTM introduces non-linear feedback effects.
- Directly addresses FINMA/EIOPA model validation requirements.

---

## 8. Open Questions & Risks

1. **Differentiability of monotonicity penalty**: Back-transforming $K_t \to m_x$ within the training loop may be computationally expensive or numerically unstable. May need a surrogate loss.
2. **Stationarity vs. data**: The ADF/KPSS analysis (both in Project 04 and replicated here) shows that stationarity is violated for 4/6 countries. The stationarity penalty may hurt performance. The λ sweep is critical.
3. **Overfitting to constraints**: If λ values are too high, the model may satisfy constraints perfectly but lose predictive power. The Pareto frontier analysis will reveal this.
4. **Computational cost**: Rolling-window validation × multi-seed × λ sweep = many training runs. Need to plan compute budget.
5. **Sex-specific dynamics**: The Sweden Female anomaly (higher volatility than Male) needs investigation. If sex-specific factors behave differently, the optimal λ configuration may differ by sex.
6. **Japan Female catch-up**: The larger range of $k_{t,JPN}^{(F)}$ compared to $k_{t,JPN}^{(M)}$ suggests that the coherence penalty may need sex-specific calibration.

---

## 9. Limitations (Current State)

- **No CBD benchmark**: Unlike Project 04, we have not implemented the Cairns-Blake-Dowd model for ages 65-90. This is a deliberate scope reduction — the AINN's contribution is in the constrained loss, not in additional actuarial baselines. CBD can be added later if needed for the paper.
- **No exposure data**: We work with death rates ($m_x$) only, not exposures ($E_x$). This is sufficient for the Li-Lee framework but limits the ability to compute weighted averages or credibility-weighted blends at the population level.
- **Single train/val split so far**: The rolling-window validation (Notebook 05) will address this. Current results are based on the standard 1956-2011 / 2012-2020 split.
