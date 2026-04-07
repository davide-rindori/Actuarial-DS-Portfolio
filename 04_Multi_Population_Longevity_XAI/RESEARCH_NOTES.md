# Research Notes: Multi-Population Longevity Forecasting

## 1. Data Selection & Preprocessing
- **The Mortality Matrix ($m_{x,t}$)**: The fundamental block.
    - *Definition*: Each element represents the Central Death Rate, calculated as $m_{x,t} = D_{x,t} / E_{x,t}$, where $D$ is the number of deaths and $E$ is the exposure (average population at risk) for age $x$ in year $t$.
    - *Structure*: Rows ($x$) represent ages (0-90), columns ($t$) represent years (1956-2021).
- **Logarithmic Transformation ($\ln(m_{x,t})$)**:
    - *Linearization*: Human mortality follows an exponential growth with age (Gompertz Law). Taking the natural logarithm transforms this into a near-linear relationship, making it suitable for SVD-based modeling.
    - *Positivity Guarantee*: Modeling mortality in the log-domain ensures that when we project back ($e^{\ln(m)}$), the predicted mortality rates are strictly positive, avoiding the biological impossibility of negative death rates.
- **Cluster**: CHE, SWE, NOR, DEUTW, NLD, JPN.
- **Time Window**: 1956-2021. 
    - *Decision*: We truncated the historical series (which for SWE/CHE are much longer) to include West Germany (DEUTW), ensuring a more representative European cluster.
    - *Technical Note*: A common time window is a mathematical prerequisite for the Li-Lee Common Factor Model to calculate a balanced average trend across all populations.
- **Age Range**: 0-90.
    - *Decision*: Capped at 90 to avoid "oldest-old" volatility and small-sample noise in the HMD data. High-age mortality (90+) often suffers from low exposure, leading to erratic $m_x$ values.
- **Data Integrity**: Implemented a $10^{-10}$ epsilon for log-mortality transformation.
    - *Reasoning*: Smaller populations (e.g., Norway) occasionally report zero deaths for specific age/year cells. Since $\ln(0)$ is undefined, this epsilon ensures numerical stability during Singular Value Decomposition (SVD).

### 1.1 Cluster Rationale (Coherence vs. Volume)
- **Why 6 countries?**: While increasing the number of populations (e.g., adding Denmark, Austria, or Belgium) might reduce statistical variance, it risks "trend dilution." 
- **The "Pure Signal" Strategy**: We selected a "High-Longevity Gold Standard Cluster." Including countries like the USA or UK would introduce structural breaks (e.g., the opioid crisis or distinct social health shocks) that contaminate the "frontier" mortality signal shared by the selected nations.
- **Data Quality**: Countries like SWE, CHE, and JPN possess the most reliable long-term historical records in the Human Mortality Database (HMD).
- **XAI Benefit**: A focused, high-performing cluster allows for clearer interpretability. In the future LSTM/XAI phase, mapping reciprocal influences is more effective when the underlying populations belong to a coherent socio-economic and medical system.

## 2. Preliminary Observations (EDA & LC)
- **Norway Volatility (Fig. 03)**: Norway shows significant instability in its mortality index ($k_t$) post-2010 compared to larger populations.
    - *Statistical Insight*: Smaller populations have higher variance. A few deaths more or less significantly shift the $k_t$ in a rank-1 SVD model.
    - *Demographic Insight*: Evidence of a "harvesting effect"—a period of exceptionally low mortality (2011-2015) followed by a rebound as the fragile population "catch-up" with biological limits.
- **The 2011 Inflection**: Most countries exhibit a "deceleration gap"—a visible change in the slope of $k_t$ around 2011. Standard linear models struggle to adapt to this structural change.
- **Japan's Catch-up**: Japan started with the highest mortality in 1956 but achieved the fastest rate of improvement, crossing all other countries by the 1980s to become the global longevity leader.

## 3. Lee-Carter Baseline (Independent Modeling)
- **Mathematical Framework**: $\ln(m_{x,t}) = a_x + b_x k_t + \epsilon_{x,t}$
- **Parameter Extraction via SVD**:
    1. **$a_x$ (Age Profile)**: Calculated as the mean of log-mortality over time for each age: $a_x = \frac{1}{T} \sum_t \ln(m_{x,t})$. It represents the "biological baseline" of each country.
    2. **Centering**: We subtract $a_x$ from the matrix to isolate the time-varying components.
    3. **SVD (Singular Value Decomposition)**: The centered matrix is decomposed into $U \Sigma V^T$. 
        - $k_t$ (Time Trend) is derived from the first right singular vector ($V$).
        - $b_x$ (Age Sensitivity) is derived from the first left singular vector ($U$).
    4. **$\epsilon_{x,t}$**: The residual representing noise or higher-order dynamics not captured by the first principal component.
- **Identifiability Constraints**: To avoid infinite combinations of $b_x$ and $k_t$ yielding the same product, we impose $\sum_x b_x = 1$. This "anchors" the sensitivity scale, ensuring $k_t$ captures the full magnitude of the time trend.
- **Core Limitation**: Independent modeling allows for "divergent forecasts," where geographically and socio-economically similar countries (like Sweden and Norway) could reach biologically implausible differences in future life expectancy.

## 4. Multi-Population Strategy: Li-Lee (2005)
- **The Theory**: Li-Lee expands Lee-Carter by assuming mortality is composed of a **Common Factor** (shared by the cluster) and a **Specific Factor** (local deviation).
- **Mathematical Framework**: $\ln(m_{x,t,i}) = a_{x,i} + B_x K_t + b_{x,i} k_{t,i} + \epsilon_{x,t,i}$

### 4.1 Common Factor Extraction
- **The Consensus Matrix**: Computed by averaging log-mortality matrices across all $N$ countries: $\bar{M} = \frac{1}{N} \sum_i \ln(m_{x,t,i})$. This represents the "Super-Population" trend.
- **Foundation**: $K_t$ and $B_x$ are extracted via SVD from this average matrix.
- **Constraint ($\sum B_x = 1$)**: Crucial for cross-cluster interpretability. It ensures that a unit change in $K_t$ reflects a unit change in the cluster's average log-mortality. Without this, $K_t$ would be unscalable and incomparable.
- **Denoising Effect**: The Common Factor acts as a robust signal, filtering out local anomalies (like Norway's volatility).
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
    - The Augmented Dickey-Fuller (ADF) test measures the **speed of mean reversion** (the force pulling the series back toward zero), not visual stability.
    - **The Inertia Problem**: In countries like SWE or CHE, deviations from the common trend are highly persistent (high autocorrelation). Even if they move slowly, they do not "rush" back to zero. This is a structural flaw in the Li-Lee assumption: deviations are not merely noise, but persistent local trends.
    - **The Elasticity of Volatiles**: Paradoxically, NOR and JPN pass the test because their movements are more "reactive." When they drift away, they tend to return or cross the mean with enough momentum to allow the test to detect stationarity.
- **Research Implication**:
    - The Li-Lee model mandates stationarity to ensure "coherence" (no long-term divergence). Our results prove that for "core" European countries, this coherence is a mathematical imposition that contradicts the data. 
    - This creates the perfect entry point for the **LSTM**: while actuarial models are forced to "ignore" these persistent trends to maintain coherence, Deep Learning can model this underlying structure, providing more accurate forecasts without artificial mean-reversion constraints.

### 4.4 Confirmatory Stationarity Analysis: ADF vs KPSS (Cell 2.6b)
To verify the validity of the ADF results, we performed a **Conflict Analysis** by cross-referencing the results with the KPSS test ($H_0$: Series is Stationary).

| Country | ADF (H0: Non-Stat) | KPSS (H0: Stat) | Status | Statistical Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Norway** | **PASS** | **PASS** | **Safe** | Truly stationary; high "elasticity" (reverts quickly after shocks). |
| **Sweden** | **FAIL** | **FAIL** | **Unit Root** | Pure non-stationarity; the country is "divorcing" from the common trend. |
| **W. Germany**| **FAIL** | **FAIL** | **Unit Root** | Persistent structural drift; fails both criteria for stationarity. |
| **Japan** | **PASS** | **FAIL** | **Conflict** | Grey Zone; high volatility masks a long-term non-linear trend. |
| **Switzerland**| **FAIL** | **PASS** | **Inertial** | High autocorrelation; behaves like a Random Walk without clear trend. |

- **Critical Observations**:
    - **The Sweden/Germany Unit Root**: Both tests agree that for these countries, the deviation from the cluster mean is **not noise**, but a persistent local trend. Li-Lee would produce biased forecasts by forcing a return to the mean where no statistical evidence of reversion exists.
    - **The Japan Conflict**: Japan passes the ADF (fast local reversion) but fails the KPSS (presence of a long-term trend). This suggests Japan is "Trend-Stationary": it has a pull toward its own local mean, but that mean is drifting drastically away from the cluster.
    - **The Persistence Problem**: This analysis proves that linear actuarial models are filtering out "predictable signals" by treating them as "random noise." This information gap is what the LSTM architecture is designed to fill.

## 5. Alternative Models & Future Steps

### 5.1 CBD Implementation Results (Ages 65-90)
The Cairns-Blake-Dowd model was implemented to analyze mortality dynamics in advanced age groups, where SVD-based models (such as Lee-Carter) may suffer from excessive volatility.
- **Factor 1 ($\kappa_t^{(1)}$)**: Results confirm a constant and consistent decline in the overall level of mortality for all countries in the cluster. Japan exhibits the most aggressive decline, moving from the highest intercept value in 1956 to the lowest in 2020, systematically surpassing European benchmarks.
- **Factor 2 ($\kappa_t^{(2)}$)**: The analysis reveals a "steepening" phenomenon of the mortality curve. While general mortality decreases, the biological aging rate appears to accelerate, concentrating deaths at increasingly advanced ages.
- **Non-Linear Dynamics**: High volatility and divergent trajectories in Factor 2 (particularly Japan's peak in the 80s-90s and Switzerland's recent surge) confirm that the "aging slope" is an unstable parameter, difficult to capture with standard linear projections.

### 5.2 Synthesis of Actuarial Benchmarking (Notebook 02)
At the conclusion of the benchmarking phase, three fundamental criticalities were identified that justify the evolution toward Deep Learning:
1. **Stationarity Breach**: ADF/KPSS tests demonstrated that residuals from multi-population models (Li-Lee) exhibit unit roots and persistent drifts, violating classical statistical coherence assumptions.
2. **Structural Breaks**: The post-2011 mortality deceleration is a systemic signal that linear models tend to underestimate or interpret as transitory noise.
3. **Parameter Drift**: CBD parameters show that the rotation of the mortality curve (aging slope) follows non-linear dynamics requiring a more complex historical memory for accurate forecasting.

### 5.3 LSTM (Deep Learning) & XAI
- **Proposed Innovation**: Utilization of Long Short-Term Memory (LSTM) Recurrent Neural Networks, specifically designed to capture long-term dependencies and manage non-stationary time series.
- **Research Goal**: To determine if an LSTM can implicitly learn both the "Common Factor" and the "Specific Drifts" (persistent residuals) identified empirically. The objective is to outperform actuarial benchmarks by providing forecasts more resilient to the structural changes seen post-2011.
- **Explainability (XAI)**: The integration of interpretability techniques (such as SHAP or Integrated Gradients) will allow us to "open the black box," mapping how a country's trends (e.g., the Japanese miracle) influence the projections of other cluster members, transforming a predictive model into a rigorous tool for demographic analysis.

## 6. Bridge to Machine Learning: Feature Engineering & Windowing

### 6.1 Rationale for the "Hybrid" Input Vector
In Notebook 03, we transition from purely actuarial modeling to Deep Learning by exporting the latent factors from the Li-Lee model ($K_t$ and $k_{t,i}$). 
- **The "Common Anchor" Strategy**: By including the Common Factor $K_t$ as a feature, we provide the LSTM with a global "clock" of longevity. This ensures the model understands the general direction of the cluster before interpreting local deviations.
- **CBD as a Strategic Benchmark**: While CBD parameters ($\kappa_t$) were not included in the initial feature set to avoid overfitting on a small sample (55 sequences), the CBD analysis remains a critical "Stress Test" for the LSTM. If the LSTM succeeds in forecasting mortality for the 65-90 age group better than CBD, it proves that neural networks can capture the "Aging Slope" dynamics implicitly through the $k_t$ factors.

### 6.2 The Sliding Window (Lookback) Approach
To handle the temporal dependency of mortality, we implemented a supervised learning format using a **10-year lookback period**.
- **Contextual Memory**: A 10-year window allows the LSTM to identify non-linear patterns (like the post-2011 deceleration) by analyzing a decade of trajectory rather than just the last observed step.
- **Data Constraints**: This window size was selected to maximize historical context while maintaining a sufficient number of overlapping sequences for training, given the limited 65-year span of the HMD data.

### 6.3 Feature Scaling & Stationarity via First Differences
- **From Levels to Variations**: To ensure gradient stability and handle non-stationarity, we transitioned from modeling absolute index levels ($K_t$) to **First Differences** ($\Delta K_t$). This stationarizes the series around a zero mean, preventing the "drift bias" common in linear mortality projections.
- **Standardization (StandardScaler)**: We shifted from MinMax to **Standardization** (fitting a Mean of 0 and Variance of 1 solely on the training set). This provides the LSTM with a stable numerical environment where annual shocks are comparable across decades, regardless of the absolute mortality level.

## 7. Deep Learning Implementation: The Hierarchical LSTM (Notebook 03)

### 7.1 Architecture Rationale
- **Multi-Output Strategy**: The network is designed to predict the entire 7-dimensional vector of mortality indices (1 Common + 6 Specific) simultaneously. This forces the model to internalize the reciprocal constraints between the cluster and individual countries.
- **Layer Stacking**: We implemented a stacked LSTM (32-16 units) after Bayesian optimization.
    - *Observation*: Initial attempts with larger networks (64+ units) led to immediate overfitting due to the low sample size (N=46 training sequences). The "shallowing" of the network improved validation stability significantly.

### 7.2 Bayesian Hyperparameter Optimization
To avoid arbitrary parameter selection, we utilized **Bayesian Optimization** (Keras Tuner) to explore the configuration space.
- **Search Space**: Number of units, dropout rates, and learning rates.
- **Optimal Result**: The tuner identified a lean architecture (32/16 units) with a relatively high learning rate (0.01).
- *Technical Insight*: A higher learning rate was necessary to allow the optimizer to escape local minima in a high-dimensional loss landscape despite the small number of training epochs.

## 8. Methodological Pivots: Fighting Data Leakage & Drift

### 8.1 The Anti-Leakage Protocol
A critical refinement was made regarding **Feature Scaling**. 
- **Standard Protocol (Initial)**: Fit-transform on the entire dataset.
- **Strict Protocol (Revised)**: Fit the scaler exclusively on the training set (1956-2011) and apply it to the validation set. 
- *Consequence*: This revealed a massive "Drift Bias." Because mortality post-2011 reached values lower than the 1956-2011 minimum, the `MinMaxScaler` produced out-of-bounds inputs, leading to a failure in convergence (Validation Loss explosion). This "Failure" served as empirical proof of the non-stationarity of the frontier longevity signal.

### 8.2 The "First Differences" Pivot ($\Delta K_t$)
To solve the drift bias identified in the levels, we transitioned to modeling **First Differences** (annual changes) instead of absolute index levels.
- **Mathematical Rationale**: Modeling $Y_t = K_t - K_{t-1}$ transforms a non-stationary process with drift into a near-stationary process.
- **Result**: RMSE on the validation set dropped from **21.3** (levels) to **4.7** (differences).
- **Expectation Realized**: The LSTM proved much more adept at filtering the "volatility noise" of annual variations than at guessing the absolute depth of a persistent drift. This approach aligns the model with standard econometric practices (Unit Root handling).

## 9. Performance Analysis: Out-of-Sample Validation (2012-2020)

### 9.1 The "Conservative Bias" Discovery (Fig. 08)
- **Visual Analysis**: The LSTM forecast on the validation set (2012-2020) shows a smooth, persistent downward trend, whereas the Li-Lee Actuals exhibit erratic volatility and a partial stasis (plateau) around 2017-2020.
- **Justification for Risk Management**: The LSTM appears to ignore the short-term "stalls" in mortality improvement, treating them as transitory noise. 
- **Actuarial Implication**: From a Swiss Re or Wüthrich perspective, this model is "Prudently Optimistic" about longevity. It suggests that despite recent slowing, the underlying biological longevity engine is still active. 

### 9.2 Robustness & Early Stopping
- **Training Stability**: The model consistently reaches optimal weights around Epoch 10-15. Restoring best weights via Early Stopping prevents the model from "memorizing" the noise of the small training sample.
- **Conclusion of Notebook 03**: We have achieved a stable, serialized, and peer-reviewable neural model. The next phase (Notebook 04) will focus on recursive forecasting to project these dynamics into 2050 with fan charts.