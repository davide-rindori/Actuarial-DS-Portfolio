# Research Notes: Multi-Population Longevity Forecasting

## 1. Data Selection & Preprocessing
- **The Mortality Matrix ($m_{x,t}$)**: The fundamental block of our research.
    - *Definition*: Each element represents the Central Death Rate, calculated as $m_{x,t} = D_{x,t} / E_{x,t}$, where $D$ is the number of deaths and $E$ is the exposure (average population at risk) for age $x$ in year $t$.
    - *Structure*: Rows ($x$) represent ages (0-90), columns ($t$) represent years (1956-2021).
- **Logarithmic Transformation ($\ln(m_{x,t})$)**:
    - *Linearization*: Human mortality follows an exponential growth with age (Gompertz Law). Taking the natural logarithm transforms this into a near-linear relationship, making it suitable for SVD-based modeling.
    - *Positivity Guarantee*: Modeling mortality in the log-domain ensures that when we project back ($e^{\ln(m)}$), the predicted mortality rates are strictly positive, avoiding the biological impossibility of negative death rates.
- **Cluster**: CHE (Switzerland), SWE (Sweden), NOR (Norway), DEUTW (West Germany), NLD (Netherlands), and JPN (Japan).
- **Time Window**: 1956-2021.
    - *Decision*: We truncated the historical series (which for SWE/CHE are much longer) to include West Germany (DEUTW), ensuring a more representative European cluster.
    - *Technical Note*: A common time window is a mathematical prerequisite for the Li-Lee Common Factor Model to calculate a balanced average trend across all populations.
- **Age Range**: 0-90.
    - *Decision*: Capped at 90 to avoid "oldest-old" volatility and small-sample noise in the HMD data. High-age mortality (90+) often suffers from low exposure, leading to erratic $m_x$ values.
- **Data Integrity**: Implemented a $10^{-10}$ epsilon for log-mortality transformation.
    - *Reasoning*: Smaller populations (e.g., Norway) occasionally report zero deaths for specific age/year cells. Since $\ln(0)$ is undefined, this epsilon ensures numerical stability during Singular Value Decomposition (SVD).

### 1.1 Cluster Rationale (Coherence vs. Volume)
- **Why 6 countries?**: While increasing the number of populations (e.g., adding Denmark, Austria, or Belgium) might reduce statistical variance, it risks "trend dilution".
- **The "Pure Signal" Strategy**: We selected a "High-Longevity Gold Standard Cluster". Including countries like the USA or UK would introduce structural breaks (e.g., distinct social health shocks) that contaminate the "frontier" mortality signal shared by the selected nations.
- **Data Quality**: Countries like SWE, CHE, and JPN possess the most reliable long-term historical records in the Human Mortality Database (HMD).
- **XAI Benefit**: A focused, high-performing cluster allows for clearer interpretability. In the future LSTM/XAI phase, mapping reciprocal influences is more effective when the underlying populations belong to a coherent socio-economic and medical system.

## 2. Preliminary Observations (EDA & LC)
- **Norway Volatility (Fig. 03)**: Norway shows significant instability in its mortality index ($k_t$) post-2010 compared to larger populations.
    - *Statistical Insight*: Smaller populations have higher variance. A few deaths more or less significantly shift the $k_t$ in a rank-1 SVD model.
    - *Demographic Insight*: Evidence of a "harvesting effect"—a period of exceptionally low mortality (2011-2015) followed by a rebound as the fragile population "catch-up" with biological limits.
- **The 2011 Inflection**: Most countries exhibit a "deceleration gap"—a visible change in the slope of $k_t$ around 2011. Standard linear models struggle to adapt to this structural change.
- **Japan's Catch-up**: Japan started with the highest mortality in 1956 but achieved the fastest rate of improvement, crossing all other countries by the 1980s to become the global longevity leader.

## 3. Lee-Carter Baseline (Independent Modeling)
- **Mathematical Framework**: $\ln(m_{x,t}) = a_x + b_x k_t + \epsilon_{x,t}$.
- **Parameter Extraction via SVD**:
    1. **$a_x$ (Age Profile)**: Calculated as the mean of log-mortality over time for each age: $a_x = \frac{1}{T} \sum_t \ln(m_{x,t})$. It represents the "biological baseline" of each country.
    2. **Centering**: We subtract $a_x$ from the matrix to isolate the time-varying components.
    3. **SVD (Singular Value Decomposition)**: The centered matrix is decomposed into $U \Sigma V^T$. 
        - $k_t$ (Time Trend) is derived from the first right singular vector ($V$).
        - $b_x$ (Age Sensitivity) is derived from the first left singular vector ($U$).
    4. **$\epsilon_{x,t}$**: The residual representing noise or higher-order dynamics not captured by the first principal component.
- **Identifiability Constraints**: To avoid infinite combinations of $b_x$ and $k_t$ yielding the same product, we impose $\sum_x b_x = 1$. This "anchors" the sensitivity scale, ensuring $k_t$ captures the full magnitude of the time trend.
- **Core Limitation**: Independent modeling allows for "divergent forecasts," where geographically and socio-economically similar countries could reach biologically implausible differences in future life expectancy.

## 4. Multi-Population Strategy: Li-Lee (2005)
- **The Theory**: Li-Lee expands Lee-Carter by assuming mortality is composed of a **Common Factor** (shared by the cluster) and a **Specific Factor** (local deviation).
- **Mathematical Framework**: $\ln(m_{x,t,i}) = a_{x,i} + B_x K_t + b_{x,i} k_{t,i} + \epsilon_{x,t,i}$.

### 4.1 Common Factor Extraction
- **The Consensus Matrix**: Computed by averaging log-mortality matrices across all $N$ countries: $\bar{M} = \frac{1}{N} \sum_i \ln(m_{x,t,i})$. This represents the "Super-Population" trend.
- **Foundation**: $K_t$ and $B_x$ are extracted via SVD from this average matrix.
- **Constraint ($\sum B_x = 1$)**: Crucial for cross-cluster interpretability. It ensures that a unit change in $K_t$ reflects a unit change in the cluster's average log-mortality.
- **Denoising Effect**: The Common Factor acts as a robust signal, filtering out local anomalies.
- **Observation on Deceleration**: The common trend confirms that the post-2011 deceleration is a systemic shift across the entire cluster.

### 4.2 Country-Specific Factors (Specific Residuals)
- **Extraction (Cell 2.5)**: Computed by applying a second SVD to the residuals after removing the Common Factor and the country-specific baseline $a_{x,i}$.
- **Visual Assessment (Fig. 05)**:
    - **Japan's Outlier Status**: JPN shows a massive downward trend in its specific factor ($k_{t,JPN}$) from 1956 to 1990. This reflects the "miracle" phase in which Japan improved much faster than the European average.
    - **Norway's Volatility**: Recent spikes (2015-2020) are isolated in the specific factor, demonstrating that these are local "shocks" rather than systemic changes within the cluster.
    - **European Persistency**: CHE, SWE, and DEUTW exhibit very smooth and slow-moving residuals. Although visually "stable," they do not oscillate rapidly around zero, suggesting a persistent drift.

### 4.3 Statistical Validation: The Stationarity Paradox (Cell 2.6)
- **ADF Test Results**:
    - **FAIL (Non-Stationary)**: Switzerland (0.76), Sweden (0.92), West Germany (0.94), Netherlands (0.10).
    - **PASS (Stationary)**: Norway (0.00), Japan (0.04).
- **Solving the Paradox**: 
    - The Augmented Dickey-Fuller (ADF) test measures the **speed of mean reversion**, not visual stability.
    - **The Inertia Problem**: In countries like SWE or CHE, deviations from the common trend are highly persistent. Even if they move slowly, they do not "rush" back to zero. This is a structural flaw in the Li-Lee assumption: deviations are persistent local trends.
    - **The Elasticity of Volatiles**: Paradoxically, NOR and JPN pass the test because their movements are more "reactive". When they drift away, they tend to return or cross the mean with enough momentum to allow the test to detect stazionarity.
- **Research Implication**:
    - The Li-Lee model mandates stationarity to ensure "coherence". Our results prove that for "core" European countries, this coherence is a mathematical imposition that contradicts the data. 
    - This creates the perfect entry point for the **LSTM**: while actuarial models are forced to "ignore" these persistent trends, Deep Learning can model this underlying structure.

## 4.4 Confirmatory Stationarity Analysis: ADF vs KPSS (Cell 2.6b)
To verify the validity of the ADF results, we performed a **Conflict Analysis** by cross-referencing the results with the KPSS test ($H_0$: Series is Stationary).

| Country | ADF (H0: Non-Stat) | KPSS (H0: Stat) | Status | Statistical Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Norway** | **PASS** | **PASS** | **Safe** | Truly stationary; high "elasticity" (reverts quickly). |
| **Sweden** | **FAIL** | **FAIL** | **Unit Root** | Pure non-stationarity; "divorcing" from the common trend. |
| **W. Germany**| **FAIL** | **FAIL** | **Unit Root** | Persistent structural drift; fails both criteria. |
| **Japan** | **PASS** | **FAIL** | **Conflict** | Grey Zone; volatility masks a non-linear trend. |
| **Switzerland**| **FAIL** | **PASS** | **Inertial** | High autocorrelation; behaves like a Random Walk. |

- **Critical Observations**:
    - **The Sweden/Germany Unit Root**: Both tests agree that for these countries, the deviation from the cluster mean is a persistent local trend. Li-Lee would produce biased forecasts by forcing a return to the mean.
    - **The Japan Conflict**: Japan passes the ADF but fails the KPSS. This suggests Japan is "Trend-Stationary": its mean is drifting drastically away from the cluster.
    - **The Persistence Problem**: This analysis proves that linear actuarial models are filtering out "predictable signals" by treating them as "random noise".

## 5. Alternative Models & Future Steps

### 5.1 CBD Implementation Results (Ages 65-90)
The Cairns-Blake-Dowd model was implemented to analyze mortality dynamics in advanced age groups.
- **Factor 1 ($\kappa_t^{(1)}$)**: Results confirm a consistent decline in the overall level of mortality. Japan exhibits the most aggressive decline, systematically surpassing European benchmarks.
- **Factor 2 ($\kappa_t^{(2)}$)**: The analysis reveals a "steepening" phenomenon of the mortality curve. The biological aging rate appears to accelerate, concentrating deaths at advanced ages.
- **Non-Linear Dynamics**: High volatility in Factor 2 confirm that the "aging slope" is an unstable parameter, difficult to capture with standard linear projections.

### 5.2 Synthesis of Actuarial Benchmarking (Notebook 02)
Three fundamental criticalities were identified:
1. **Stationarity Breach**: Residuals from Li-Lee exhibit unit roots and persistent drifts.
2. **Structural Breaks**: The post-2011 mortality deceleration is a systemic signal linear models underestimate.
3. **Parameter Drift**: CBD parameters show that the rotation of the mortality curve follows non-linear dynamics.

### 5.3 LSTM (Deep Learning) & XAI
- **Proposed Innovation**: Utilization of Long Short-Term Memory (LSTM) Recurrent Neural Networks to capture long-term dependencies and manage non-stationary time series.
- **Research Goal**: To determine if an LSTM can implicitly learn both the "Common Factor" and the "Specific Drifts".
- **Explainability (XAI)**: Integration of interpretability techniques (such as SHAP) to "open the black box," mapping cluster member influences.

## 6. Bridge to Machine Learning: Feature Engineering & Windowing

### 6.1 Rationale for the "Hybrid" Input Vector
In Notebook 03, we transition to Deep Learning by exporting the latent factors from the Li-Lee model ($K_t$ and $k_{t,i}$). 
- **The "Common Anchor" Strategy**: By including $K_t$ as a feature, we provide the LSTM with a global "clock". 
- **CBD as a Strategic Benchmark**: The CBD analysis remains a critical "Stress Test" for the LSTM regarding the 65-90 age group.

### 6.2 The Sliding Window (Lookback) Approach
We utilized a **Sliding Window** (Lookback) approach (10-year sequences) to handle temporal dependency.
- **Contextual Memory**: A 10-year window allows the LSTM to identify non-linear patterns (like post-2011 deceleration).

### 6.3 Feature Scaling & Stationarity via First Differences
- **From Levels to Variations**: To handle non-stationarity, we transitioned to modeling **First Differences** ($\Delta K_t$). 
- **Standardization (StandardScaler)**: We shifted to **Standardization** (fitting Mean 0, Variance 1 on the training set) to provide a stable numerical environment.

## 7. Deep Learning Implementation: The Hierarchical LSTM (Notebook 03)

### 7.1 Architecture Rationale
- **Multi-Output Strategy**: The network predicts the entire 7-dimensional vector of mortality indices (1 Common + 6 Specific) simultaneously.
- **Layer Stacking**: We implemented a stacked LSTM (32-16 units) after Bayesian optimization.
    - *Observation*: Initial attempts with larger networks led to immediate overfitting due to the low sample size.

### 7.2 Bayesian Hyperparameter Optimization
We utilized **Bayesian Optimization** (Keras Tuner) to explore the configuration space (units, dropout, learning rates).
- **Optimal Result**: The tuner identified a lean architecture (32/16 units) with a learning rate of 0.01.

## 8. Methodological Pivots: Fighting Data Leakage & Drift

### 8.1 The Anti-Leakage Protocol
A critical refinement was made to fit the scaler exclusively on the training set (1956-2011) and apply it to the validation set. 
- *Consequence*: This revealed a massive "Drift Bias" where post-2011 mortality reached values lower than the training minimum, proving the non-stationarity of the signal.

### 8.2 The "First Differences" Pivot ($\Delta K_t$)
To solve the drift bias, we transitioned to modeling **First Differences** (annual changes) instead of absolute levels.
- **Result**: RMSE on the validation set dropped from **21.3** (levels) to **4.56** (differences).

## 9. Performance Analysis: Out-of-Sample Validation (2012-2020)

### 9.1 The "Conservative Bias" Discovery (Fig. 08)
- **Visual Analysis**: The LSTM forecast shows a smooth downward trend, whereas actuals exhibit erratic volatility and a plateau around 2017-2020.
- **Actuarial Implication**: The model is "Prudently Optimistic," suggesting that fundamental biological longevity drivers remain active. 

### 9.2 Robustness & Early Stopping
- **Training Stability**: The model reaches optimal weights around Epoch 10-15. Early Stopping prevents memorizing noise.

## 10. Stochastic Forecasting & Inference Strategy (Notebook 04)

### 10.1 Bayesian LSTM Inference: Monte Carlo Dropout (MCD)
We implemented **Monte Carlo Dropout** to address the deterministic nature of standard LSTMs during inference.
- **Theoretical Foundation**: By keeping Dropout layers active during the prediction phase, the model acts as a Bayesian approximation, sampling 1,000 trajectories.
- **Expectation Realized**: The "Fan Chart" (Fig. 09) exhibits a natural expansion of uncertainty over time, critical for solvency frameworks.

### 10.2 Recursive Forecasting Framework
Predictions are generated through a recursive feedback loop where each predicted variation ($\Delta K_{t+1}$) is integrated back into the input window.
- **Technical Implementation**: We used a 10-year sliding window of variations and `np.roll` for updates.
- **The Pivot to Stochastic Integration**: The model forecasts variations, then cumulatively summed starting from 2020.

### 10.3 Technical Challenges and Adjustments
- **Optimizer Mismatch**: Loaded the model with `compile=False` to avoid irrelevant optimizer warnings.
- **Input Structure Validation**: Ensured explicit casting to `tf.float32` and batch-dimensioned tensors.

### 10.4 Key Observations: Fan Chart Interpretation (Fig. 09)
- **Deep Learning Advantage**: The LSTM median exhibits non-linear curvatures, projecting inflections or rhythmic stalls learned from history.
- **Uncertainty Asymmetry**: Higher probability of "longevity shocks" (lower $K_t$) compared to mortality spikes.
- **Drift Stability**: The median $K_t$ reaches **-123.63** by 2050 from approx. -60 in 2020.

## 11. Demographic Impact: Life Expectancy Reconstruction

### 11.1 Back-Transformation Methodology
Latent factors were back-transformed into death rates ($m_x$) using baseline age profiles ($a_{x,i}$) and common sensitivities ($B_x$).
- **Actuarial Life Table**: Calculated life expectancy at birth ($e_0$) via the trapezoidal rule across 1,000 trajectories.

### 11.2 Cluster-Wide Results and Longevity Convergence
- **Systemic Longevity Signal**: Gains in $e_0$ range between **+3.43 and +3.92 years** by 2050.
- **The Convergence Effect**: Countries from lower baselines, such as West Germany ($e_0$ 80.37), exhibit faster improvement (+3.92 years) than leaders like Switzerland (+3.48 years). 

### 11.3 Case Study: Switzerland (CHE) Findings
- **Forecast Resilience**: CHE projects a median $e_0$ increase from **81.71** (2020) to **85.19** (2050).
- **Sensitivity to 2020 Shocks**: Interprets the pandemic as a transitory shock rather than a permanent structural shift.

### 11.4 Comparative Multi-Country Visualization Analysis (Fig. 10)
- **The "Parallelism" of Longevity**: Switzerland and Japan converge toward the ~85.2-year mark by 2050, as the LSTM perceives both as "Frontier Leaders".
- **Evidence of Catch-up Dynamics (West Germany)**: DEUTW maintains a steeper improvement slope, confirming the "Convergence Effect".

## 12. Explainable AI (XAI): Deciphering the Black Box (Fig. 11)

### 12.1 Temporal Saliency Analysis
Measured sensitivity of the 2021-2050 forecast to each of the 10 years in the input window (2011-2020).

### 12.2 Bimodal Memory Profile
XAI results reveal a distinct **Bimodal Importance** distribution:
1. **Recency Bias (t-1: 21.6%)**: Most recent observed year has the highest power.
2. **Deep Contextual Memory (t-8: 20.1%)**: Identifies structural patterns or cyclical echoes from a decade ago.

### 12.3 Research Insights: Expected vs. Discovered Patterns
- **Discovery of Cyclical Memory**: High importance of lag t-8 was an unexpected discovery.
- **Justification for Lookback Window**: Non-zero importance at t-10 (6.4%) validates the 10-year window.

## 13. Actuarial Validation and Synthesis (Notebook 05)

### 13.1 Quantitative Benchmarking Results (Table 1)
Synthesis of stochastic results into Table 1.

### Table 1: Stochastic Longevity Projections Summary (2020-2050)

| Country | Code | e0 (2020) | e0 (2050) Median | 95% CI (2050) | Net Gain (Yrs) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Switzerland** | CHE | 81.71 | 85.19 | [85.15 - 85.22] | +3.48 |
| **Sweden** | SWE | 81.71 | 85.14 | [85.11 - 85.18] | +3.43 |
| **Norway** | NOR | 81.53 | 85.05 | [85.01 - 85.08] | +3.51 |
| **West Germany**| DEUTW | 80.37 | 84.29 | [84.25 - 84.33] | +3.92 |
| **Netherlands** | NLD | 81.30 | 84.89 | [84.85 - 84.93] | +3.59 |
| **Japan** | JPN | 81.72 | 85.20 | [85.16 - 85.23] | +3.47 |

### 13.2 Statistical Robustness and Risk Implications
- **Confidence Interval Stability**: Narrow 95% CI (approx. ±0.04 years). 
- **Prudence vs. Optimism**: A projected gain of ~3.5 years over 30 years is "Prudently Optimistic".

## 14. Actuarial Stress Testing: Model Resilience (Fig. 12)

### 14.1 Stress Scenario Rationale: The "Medical Breakthrough" Shock
Simulated a breakthrough in 2026 reducing $m_x$ by 10%.

### 14.2 Technical Observations and Resilience Analysis
- **Structural Jump vs. Trend Stability**: Immediate leap of +1.0 year in life expectancy. Accepts the shock as a new baseline.
- **Quantifying Longevity Risk**: Median $e_0$ for CHE shifts from 85.19 to 86.19.

| Metric | Baseline Forecast (CHE) | Shocked Scenario | Delta |
| :--- | :--- | :--- | :--- |
| **Median e0 (2050)** | 85.19 years | 86.19 years | **+1.00 yr** |

## 15. Final Consolidation: Multi-Population Convergence (Fig. 13)

### 15.1 The Longevity Frontier Synthesis
Visualizes median trajectories for all six nations.

### 15.2 Key Takeaways and Project Conclusions
- **Implicit Convergence Mechanism**: West Germany exhibits the steepest trajectory, reducing the gap with leaders.
- **Quantification of Risk**: Actuarial Risk Margin for Switzerland calculated at **0.0654 years**.

## 16. Biological Consistency & Monotonicity Audit (Fig. 14)

### 16.1 Testing Rigor and Gompertzian Compliance
Ensured death rates ($m_x$) non-decrease with age. 
- **Gompertz Consistency**: All nations exhibit near-perfect exponential growth in mortality from age 40 onwards.

### 16.2 Analysis of Results: PASS vs. FAIL
- **FAIL (Young Ages)**: CHE and NOR triggered a FAIL due to low mortality levels and youth accidents creating non-monotonic ripples between 20-30. 
- **Actuarial Verdict**: For longevity risk (65-90), the model is highly consistent.

## 17. Regulatory Capital & Full Cluster Tail Risk Analysis (Fig. 15)

### 17.1 Introduction to Longevity Tail Risk
Quantified risk using **VaR 99.5%** (Solvency II) and **Expected Shortfall 99.0%** (SST).

### 17.2 Results: Table 2 - Full Cluster Breakdown (2050 Horizon)

| Country | Median e0 (2050) | SCR (VaR 99.5%) | SCR (ES 99.0%) |
| :--- | :--- | :--- | :--- |
| **Switzerland** | 85.19 | +0.087 yrs | +0.089 yrs |
| **Japan** | 85.20 | +0.087 yrs | +0.088 yrs |
| **Sweden** | 85.14 | +0.087 yrs | +0.088 yrs |
| **W. Germany** | 84.29 | +0.100 yrs | +0.101 yrs |
| **Netherlands** | 84.89 | +0.091 yrs | +0.092 yrs |
| **Norway** | 85.05 | +0.089 yrs | +0.090 yrs |

### 17.3 Deep Dive & Methodological Observations
- **Risk Convergence**: CHE, JPN, and SWE share nearly identical SCR of **0.087 years**. This suggests a "Longevity Frontier" where risk is driven by systemic ceilings.
- **The German "Catch-up" Penalty**: West Germany exhibits the highest relative risk (+0.100 yrs) because rapid catch-up is harder to predict.
- **Leptokurtic Distributions**: Narrow tails with minimal gap between VaR and ES indicate a lack of "Fat Tails" or explosive scenarios.

## 18. Statistical Exhaustiveness: Cluster-Wide Lexis Analysis (Fig. 16)

### 18.1 Understanding the "Residual" logic
Verified that residuals (observed - reconstructed) look like random noise. 

### 18.2 Lexis Map Interpretation: Hunting for Ghost Patterns
- **瑞士 Case Study**: Deep purple near-zero error surface. Absence of diagonal artifacts confirms successful internalization of cohort dynamics.
- **The 2020 Anomaly**: Localized pandemic shock correctly isolated.

### 18.3 Table 3: Cluster-Wide Performance (2012-2020)
| Country | MAE (Log-Scale) |
| :--- | :--- |
| **Netherlands** | 0.0675 |
| **Japan** | 0.0899 |
| **W. Germany** | 0.1111 |
| **Switzerland** | 0.1243 |
| **Sweden** | 0.1369 |
| **Norway** | 0.1382 |

## 19. Explainable AI (XAI): SHAP Multi-Country Influence Mapping (Fig. 18)

### 19.1 Theoretical Framework
Utilized **SHAP** to measure fair distribution of prediction impact among players (countries).

### 19.2 Methodology
Used a "Flattening" strategy to handle 3D tensors for KernelExplainer. Operated on **First Differences**.

### 19.3 Results and Observations: The Swiss Hierarchy (Fig. 18)
| Input Feature | Mean Abs SHAP Value |
| :--- | :--- |
| **West Germany** | **0.02604** |
| **Common Factor (Kt)**| **0.01549** |
| **Japan** | **0.01301** |

- **Regional Lead-Lag (West Germany)**: Dominant predictor due to geographic and healthcare proximity.
- **Biological Frontier (Japan)**: Acts as a "Biological Compass" defining what is possible.

## 20. Performance Benchmarking: The Challenge of Level Reconstruction

### 20.1 Objective
Reconstructed absolute **levels** ($K_t$) from predicted variations. 

### 20.2 The Problem: Accumulated Integration Drift
Recursive systematic error accumulates linearly, producing negative improvement scores before correction. 

### 20.3 The Solution: Mean-Bias Correction (MBC)
Implemented a hybrid **Mean-Bias Correction (MBC)**: $Level_{t} = Level_{t-1} + (\Delta_{LSTM} + \text{Bias})$.

### 20.4 Analysis of Official Clean Run Results (Table 20.1)

| Country | Li-Lee RMSE (Level) | LSTM+MBC RMSE | Improvement (%) |
| :--- | :--- | :--- | :--- |
| **Japan** | 4.56460 | 3.56607 | **+21.88%** |
| **Sweden** | 2.75366 | 2.27791 | **+17.28%** |
| **West Germany** | 0.90146 | 0.81137 | **+9.99%** |
| **Netherlands** | 2.60239 | 2.42319 | **+6.89%** |
| **Norway** | 7.98333 | 7.56753 | **+5.21%** |
| **Switzerland** | 1.31148 | 1.36631 | -4.18% |

### 20.5 Academic Discussion
- **The Japanese Frontier**: **21.88%** improvement where mortality is most non-linear.
- **Selective Superiority**: Switzerland remains favorable to Li-Lee (-4.18%) due to historical linearity.

## 21. Lookback Sensitivity Analysis (Robustness Audit)

### 21.1 Methodology and Results
To validate the choice of a 10-year sliding window, we conducted a sensitivity test across three temporal horizons: 5, 10, and 15 years.
* **5-Year Window (RMSE: 7.14169)**: Highest error. Suggests that a short horizon is insufficient to capture the persistent structural drifts and the 2011 deceleration effect.
* **10-Year Window (RMSE: 7.05426)**: Current project standard. Demonstrates stable convergence and captures the bimodal memory (t-1, t-8) identified in XAI.
* **15-Year Window (RMSE: 6.91248)**: Lowest error. Confirms that mortality dynamics benefit from deep historical context.

### 21.2 Discussion for arXiv and Swiss Re
While the 15-year window yields the lowest RMSE, the **10-year lookback** remains our "Champion" configuration for two strategic reasons:
1. **Data Parsimony**: Using a 15-year window sacrifices 5 additional years of training data per country. In a dataset spanning from 1956, this reduction is material.
2. **Model Generalization**: The marginal improvement of 15y over 10y does not justify the risk of overfitting on very old demographic regimes that may no longer be relevant to current medical standards.

### 21.3 Conclusion
The analysis proves that the LSTM's superiority is rooted in its ability to process at least a decade of history. This empirical evidence directly addresses the "Model Risk" concerns by proving that the architecture is optimized for the specific "memory depth" of frontier mortality.