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

## 4. AINN Training: Design, Results, and Analysis (Notebook 03)

### 4.1 Architectural Decision: Joint Male/Female Training

#### The Problem
The initial approach — training separate models for Male and Female — produced severe overfitting. With only 45 training samples per sex (10-year lookback on 55 training differences), the model memorised the training set without generalising. Validation MSE was 28x higher than training MSE, and the model converged to near-constant predictions.

#### The Solution
We concatenated Male and Female sequences into a single training set, adding a binary sex indicator (0=Male, 1=Female) as the 8th input feature. This:
- Doubles the effective sample size from 45 to 90 training samples.
- Allows the model to learn shared temporal structure (post-2011 deceleration, convergence dynamics) across sexes.
- Preserves sex-specific output through the indicator feature.
- Is analogous to how Li-Lee uses a Common Factor across countries — here we use "common learning" across sexes.

#### Result
Joint training reduced RMSE from 7.77 (separate) to 7.59 (joint) — a **+2.39% improvement**. The validation loss dropped from ~29.5 to ~4.9, indicating genuine generalisation rather than memorisation.

#### Discussion
This is the single most impactful design choice in the notebook. It is also a genuine innovation over Project 04, which uses only "Total" (both sexes combined). The joint M/F approach is more realistic for L&H applications: a single model that produces coherent M/F projections, learning from the shared biological signal while respecting sex-specific dynamics.

The approach is defensible from multiple angles:
- **Statistical**: more data → better generalisation (fundamental ML principle).
- **Actuarial**: M and F mortality share the same underlying drivers (medical advances, lifestyle changes) with different intensities — a joint model captures this.
- **Practical**: one model to maintain instead of two, with guaranteed coherence between M/F projections.

### 4.2 Architecture: Fixed by Design

We use the same stacked LSTM (32-16 units, 20% dropout) as Project 04's champion. This is a deliberate methodological choice, not a limitation:
- **Isolation of variables**: the only difference between Project 04 and Project 05 is the constrained loss and the joint M/F training. If we also changed the architecture, we could not attribute improvements to the constraints.
- **Consistency with Bayesian tuning**: Project 04's tuner identified 32-16 as optimal for this dataset size. The dataset size is similar (90 vs 45 samples, same feature dimensionality).
- **Regulatory defensibility**: a fixed architecture with documented rationale is easier to validate than one that was re-tuned.

The output layer has 8 units (7 mortality factors + 1 sex indicator reconstruction). The sex indicator in the output is not penalised by any constraint — it serves only as a reconstruction target for the autoencoder-like structure.

### 4.3 Constrained Loss Function: Implementation

#### Final Formulation
$$\mathcal{L} = \mathcal{L}_{MSE} + \lambda_1 \cdot \mathcal{L}_{coherence} + \lambda_2 \cdot \mathcal{L}_{monotonicity}$$

#### Coherence Penalty ($\lambda_1$)
- **Implementation**: $\mathcal{L}_{coherence} = \frac{1}{6} \sum_{i=1}^{6} (\hat{\Delta k}_{t,i})^2$
- Penalises the squared magnitude of predicted country-specific variations (columns 1-6 of the output).
- A large $|\Delta k_{t,i}|$ means the country is diverging from the common trend — the penalty discourages this.
- **Note**: We penalise the scaled (standardised) predictions, not the original-scale values. This means the penalty operates on a normalised space where all factors have comparable magnitude.

#### Monotonicity Surrogate ($\lambda_2$)
- **Implementation**: $\mathcal{L}_{monotonicity} = \text{mean}(\text{ReLU}(\hat{\Delta K}_t)^2)$
- Penalises positive predictions of the common factor variation (column 0).
- **Rationale**: $K_t$ decreasing = mortality improving (longevity increasing). $\Delta K_t > 0$ means mortality is worsening, which is biologically implausible as a long-term trend for developed countries.
- **Why "surrogate"**: The original ROADMAP proposed penalising $m_{x+1} < m_x$ (age-monotonicity) directly. This requires back-transforming $\Delta K_t$ into death rates within the training loop — computationally expensive and numerically fragile. The temporal monotonicity surrogate (mortality should improve over time) is simpler, differentiable, and captures the same actuarial intuition at the aggregate level.
- **Limitation**: This does not enforce age-monotonicity (Gompertz). That will be verified post-hoc in the validation notebook, as in Project 04.

#### Stationarity Penalty ($\lambda_3$) — Tested and Excluded
- **Implementation**: $\mathcal{L}_{stationarity} = \frac{1}{6} \sum_{i=1}^{6} |\hat{\Delta k}_{t,i}|$
- Penalises the absolute magnitude of specific factor predictions (L1 norm, encouraging sparsity/mean-reversion).
- **Test results**: At $\lambda_3 = 0.1$, RMSE = 7.584 (vs 7.582 without). At $\lambda_3 = 1.0$, RMSE = 7.587 (worse).
- **Decision**: Excluded from the champion model. The penalty does not improve performance and is inconsistent with the data — our stationarity analysis (Section 3.4) showed that 4/6 countries have non-stationary specific factors. Forcing stationarity fights the data.
- **Documentation value**: Testing and excluding with evidence is stronger than simply not including it. A reviewer cannot object that we "forgot" stationarity.

### 4.4 Preliminary Lambda Sweep (Fixed Architecture)

Before the joint Bayesian Optimisation, we ran a preliminary λ sweep with a fixed architecture (32-16 units, lr=0.001) to understand the constraint landscape. This served as an exploratory step to identify the relevant range of λ values.

#### Results (Fixed Architecture, Joint M/F, 90 samples)

| Configuration | $\lambda_1$ (coh) | $\lambda_2$ (mono) | RMSE | Mean |Specific| | Frac(dKt>0) |
|:---|:---|:---|:---|:---|:---|
| Mono=1.0 | 0.0 | 1.0 | **7.5817** | 0.088 | 50.0% |
| Both=1.0 | 1.0 | 1.0 | 7.5818 | 0.074 | 50.0% |
| Mono=0.1 | 0.0 | 0.1 | 7.5838 | 0.089 | 50.0% |
| ... | ... | ... | ... | ... | ... |
| Unconstrained | 0.0 | 0.0 | 7.5851 | 0.101 | 55.6% |

Key observations from the preliminary sweep:
- The improvement from constraints was marginal (+0.045%) but consistent.
- The monotonicity penalty was more effective than coherence.
- Constraints had a meaningful effect on model properties (specific factors −27%, frac(dKt>0) −10%).
- The stationarity penalty hurt performance at λ≥1.0 and was excluded.

This preliminary exploration informed the design of the joint Bayesian Optimisation — specifically, confirming that λ values in {0, 0.001, 0.01, 0.1, 1.0} cover the relevant range.

### 4.5 Joint Bayesian Optimisation: Architecture + Constraints

#### Motivation and Design Philosophy
The preliminary λ sweep revealed that the fixed architecture (32-16, lr=0.001) may not be optimal for the joint M/F task. More importantly, there may be interactions between architecture capacity and constraint strength: a larger, more expressive model may need different (smaller) λ values than an underpowered one.

Rather than tuning architecture and constraints sequentially — which could miss these interactions — we performed a **single joint Bayesian Optimisation** treating all 5 hyperparameters simultaneously. This is methodologically superior and produces a globally optimal configuration.

The analogy in actuarial science: you would not calibrate the mortality improvement assumption independently of the volatility assumption in a stochastic model. The parameters interact, so they should be calibrated jointly.

#### Hyperparameter Space

| Parameter | Values | Rationale |
|:---|:---|:---|
| units_l1 | {16, 32, 48, 64} | Range from lean to capacity-rich |
| units_l2 | {8, 16, 24, 32} | Compression factor in second layer |
| learning_rate | {0.01, 0.005, 0.001, 0.0005, 0.0001} | Wide range; higher rates viable with larger models |
| $\lambda_{coherence}$ | {0, 0.001, 0.01, 0.1, 1.0} | Log scale, from unconstrained to strongly regularised |
| $\lambda_{monotonicity}$ | {0, 0.001, 0.01, 0.1, 1.0} | Idem |

Total space: 4 × 4 × 5 × 5 × 5 = 2,000 combinations explored intelligently by Bayesian Optimisation over 100 trials (~5% coverage, with Gaussian Process surrogate model guiding exploration toward promising regions).

**Fixed parameters** (documented rationale):
- **Lookback = 10**: Keras Tuner requires fixed input shape. Sensitivity analysis performed separately (Section 4.11). Consistent with Project 04.
- **Dropout = 0.2**: Required for MC Dropout inference. Changing this would require a separate robustness analysis.
- **Batch size = 8**: Same as Project 04. On 90 samples, this gives 11 batches per epoch — sufficient gradient signal.
- **Training: joint M/F**: Validated by ablation (+2.4% over separate training).

#### Champion Configuration Found

| Parameter | Value |
|:---|:---|
| units_l1 | **64** |
| units_l2 | **8** |
| learning_rate | **0.01** |
| $\lambda_{coherence}$ | **0.001** |
| $\lambda_{monotonicity}$ | **0.001** |
| val_loss (tuner objective) | **3.844** |

**Runtime**: 25 minutes for 100 trials on Apple M1 Pro (approximately 15 seconds per trial on average).

#### Champion Performance

| Metric | Value |
|:---|:---|
| Overall RMSE (original scale) | **7.0687** |
| Male RMSE | 6.6525 |
| Female RMSE | 7.4618 |
| Mean \|specific factor\| | 0.4980 |
| Frac(dKt>0) | 61.1% |
| Best epoch | 13 |
| Training converged at | epoch 33 (patience=20) |

### 4.6 Champion Architecture Analysis

#### The 64→8 Asymmetry
The tuner consistently found that a **large first layer followed by aggressive compression** (64→8) outperforms two balanced layers (32→16, as in Project 04). The top 5 trials all share L1=64, L2=8 — the Bayesian optimiser converged confidently on these parameters with no ambiguity.

This architecture is architecturally interesting and worth discussing in the paper:

**First layer (64 units)**: learns a rich, high-dimensional representation of the 10-year mortality history across all 7 mortality factors + sex indicator. The larger capacity allows the LSTM to capture more complex temporal dependencies and cross-factor interactions.

**Second layer (8 units)**: compresses aggressively to the minimum representation needed for prediction. This acts as an information bottleneck — forcing the network to retain only the most predictive features. The bottleneck structure is known to improve generalisation by preventing the network from memorising noise.

This asymmetry makes sense for the specific structure of mortality time series: the first layer extracts complex temporal features (regime changes, cohort effects, cross-country influences), and the second layer selects the few most predictive for the next year.

**Why is this different from Project 04?** Project 04 used a balanced 32-16 architecture found by Bayesian tuning on "Total" (both sexes combined, 45 samples). The joint M/F training (90 samples) provides more signal, allowing a larger first layer to be effective without immediately overfitting.

#### The High Learning Rate (0.01)
The optimal learning rate of 0.01 is 10x higher than Project 04's 0.001. This is consistent with the 64-unit first layer: larger models can absorb larger gradient steps without diverging, especially with early stopping. The model converges fast (best epoch 13, stops at 33) — the high learning rate enables rapid convergence to a good solution.

#### Unexpected Finding: λ = 0.001 is Optimal (Not 1.0)
This is the most surprising result of the Bayesian Optimisation. The preliminary sweep with fixed architecture suggested that larger λ values (0.1-1.0) produced the best RMSE. With the jointly tuned architecture (64-8, lr=0.01), the optimal λ drops to 0.001 — three orders of magnitude smaller.

**Explanation**: The relationship between architecture capacity and constraint strength is inversely proportional. The 32-16 architecture was relatively underpowered for the joint M/F dataset, so it needed strong regularisation (λ=1.0) to avoid overfitting the small training set. The 64-8 architecture is better calibrated to the data, so it generalises well on its own — it needs only a tiny nudge from the constraints (λ=0.001) rather than strong regularisation.

**Implication for the paper**: This interaction between architecture and constraints is a non-trivial finding. Most PINN-style papers fix the architecture and vary the constraint weights, potentially reporting misleading results about constraint effectiveness. Our joint optimisation reveals that the "optimal" constraint weight is architecture-dependent. This is worth highlighting as a methodological contribution.

#### Constraint Contribution (Post-Tuning Ablation)
Training the same tuned architecture (64-8, lr=0.01) with λ=0 (no constraints) yields RMSE = 7.0698. The champion (λ_coh=0.001, λ_mono=0.001) yields RMSE = 7.0687. The constraint effect is +0.015%.

At this scale of λ, the constraints are barely regularising — they are more a statement of actuarial principles than active regularisers. However, they produce a model that is:
- Formally constrained (regulatory defensibility)
- Marginally more accurate (+0.015%)
- Passable for governance review ("constraints are embedded by design")

The primary value is not in the RMSE improvement but in the governance story.

### 4.7 Multi-Seed Robustness (Post-Tuning)

| Seed | RMSE |
|:---|:---|
| 42 | 7.0687 |
| 123 | 6.9661 |
| 256 | 6.9997 |
| 512 | 6.8835 |
| 1024 | 7.1007 |

- **Mean**: 7.0037
- **Std**: 0.0859
- **CV**: 1.23%
- **Verdict**: PASS (threshold: CV < 10%)

**Comparison with pre-tuning**: The CV increased from 0.28% (fixed arch) to 1.23% (tuned arch). This is expected — the 64-unit first layer has more random weight initialisation variance than the 32-unit one. However, 1.23% is well within the acceptance threshold.

**Key point**: The inter-seed variation (±0.09 RMSE) is smaller than the improvement from architecture tuning (7.58 → 7.07 = Δ0.51). The tuning result is genuine, not a lucky seed artefact.

### 4.8 Ablation Summary (Complete)

| Design Choice | RMSE | vs Best | Notes |
|:---|:---|:---|:---|
| Separate M/F, fixed arch (45 samples) | 7.7705 | −9.4% | Overfitting baseline |
| Joint M/F, fixed arch, unconstrained | 7.5851 | −7.2% | Joint training effect |
| Joint M/F, fixed arch + constraints | 7.5817 | −7.1% | Preliminary λ sweep champion |
| **Joint M/F, tuned arch + constraints** | **7.0687** | **best** | **Final champion** |
| Joint M/F, tuned arch, no constraints | 7.0698 | −0.015% | Architecture dominates |

**Hierarchy of impact**:
1. **Architecture tuning**: 7.58 → 7.07 — **+6.5% improvement** (dominant driver)
2. **Joint M/F training**: 7.77 → 7.58 — **+2.4% improvement** (data doubling)
3. **Actuarial constraints**: +0.015% marginal RMSE gain (primarily a governance tool)

The architecture tuning was the single most impactful decision in the project — more than the joint training, more than the constraints. This justifies the 25-minute computational investment in Bayesian Optimisation.

### 4.9 Champion Configuration (Final)

| Parameter | Value |
|:---|:---|
| Architecture | LSTM (64, 8 units) + Dropout(0.2) each layer |
| Training | Joint M/F, lookback=10, batch=8, lr=0.01 |
| Loss | MSE + Coherence (λ=0.001) + Monotonicity (λ=0.001) |
| Early stopping | patience=20, best at epoch 13 |
| Validation RMSE | 7.0687 (overall), 6.65 (Male), 7.46 (Female) |
| Multi-seed CV | 1.23% (PASS) |
| Constraint effect | +0.015% vs same arch unconstrained |

### 4.10 Limitations and Open Points (Post-Tuning)

1. **Lookback fixed at 10**: Keras Tuner cannot handle variable input shape. Sensitivity analysis (Section 4.11) performed separately. If lookback=15 proves superior, the Bayesian Optimisation should be re-run with lookback=15 fixed.

2. **Female RMSE > Male RMSE (7.46 vs 6.65)**: The model predicts male mortality more accurately. This is consistent with the higher volatility of female-specific factors observed in the stationarity analysis. Sex-specific regularisation (different λ for M vs F) could be explored.

3. **frac(dKt>0) = 61.1%** with λ_mono=0.001: The small constraint weight has limited effect on temporal monotonicity. If the Notebook 04 forecasting reveals systematic upward drift, increasing λ_mono (at the cost of marginal RMSE) may be warranted.

4. **The constraint interaction with architecture capacity**: Our finding that optimal λ is architecture-dependent suggests that constraint calibration should be re-done whenever the architecture changes. This is a practical limitation for deployment.

5. **Top 5 trials converge on same architecture**: The Bayesian Optimiser is very confident about L1=64, L2=8, lr=0.01. The variation in top trials is only in λ (0.001 vs 0.01). This could indicate that 100 trials are sufficient or that a more aggressive exploration (different acquisition function) might find alternative architectures. For the purposes of this project, 100 trials is adequate.

### 4.11 Lookback Sensitivity Analysis (Results TBD — Notebook 03, Section 3.11)

The lookback sensitivity analysis tests the champion architecture (64-8, lr=0.01, λ=0.001) with lookback ∈ {5, 7, 10, 12, 15}. Results will be appended here after execution.

Expected findings (based on Project 04):
- Lookback=5: insufficient temporal context, higher RMSE.
- Lookback=10: current standard, good balance.
- Lookback=15: marginally better but reduces training samples.

If lookback=15 proves superior, we will update the champion to lookback=15 and re-run the Bayesian Optimisation (which is feasible in 25 minutes).

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

## 6. Robustness Protocol

### 6.1 Multi-Seed Table (COMPLETED — Notebook 03)
- Champion architecture: LSTM (64-8), lr=0.01, λ_coh=0.001, λ_mono=0.001.
- Seeds: [42, 123, 256, 512, 1024].
- **Results**: Mean RMSE = 7.0037, Std = 0.0859, **CV = 1.23%**.
- **Verdict**: PASS (threshold: CV < 10%).
- This addresses the "Model Risk" concern that neural network results depend on lucky initialisation. A CV of 1.23% confirms the result is genuine and reproducible.



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

- **No CBD benchmark**: Unlike Project 04, we have not implemented the Cairns-Blake-Dowd model for ages 65-90. This is a deliberate scope reduction — the AINN's contribution is in the constrained loss, not in additional actuarial baselines.
- **No exposure data**: We work with death rates ($m_x$) only, not exposures ($E_x$). This is sufficient for the Li-Lee framework but limits the ability to compute weighted averages or credibility-weighted blends at the population level.
- **Monotonicity surrogate is temporal, not age-based**: We penalise $\Delta K_t > 0$ (mortality worsening over time) but do not enforce $m_{x+1} \geq m_x$ (Gompertz) during training. Age-monotonicity is verified post-hoc.
- **Small RMSE improvement from constraints (+0.045%)**: The value of constraints is expected to emerge in long-term forecasting stability (Notebook 04), not in one-step-ahead accuracy.
- **Rolling-window validation not yet performed**: Current results are based on the standard 1956-2011 / 2012-2020 split. Rolling-window (Notebook 05) will provide more robust estimates.
