# Research Notes: Multi-Population Longevity Forecasting

## 1. Data Selection & Preprocessing
- **The Mortality Matrix ($m_{x,t}$)**: The fundamental building block.
    - *Definition*: Each element represents the Central Death Rate, calculated as $m_{x,t} = D_{x,t} / E_{x,t}$, where $D$ is the number of deaths and $E$ is the exposure (average population at risk) for age $x$ in year $t$.
    - *Structure*: Rows ($x$) represent ages (0-90), columns ($t$) represent years (1956-2021).
- **Logarithmic Transformation ($\ln(m_{x,t})$)**:
    - *Linearization*: Human mortality follows an exponential growth with age (Gompertz Law). Taking the natural logarithm transforms this into a near-linear relationship, making it suitable for SVD-based modeling.
    - *Positivity Guarantee*: Modeling mortality in the log-domain ensures that when we project back ($e^{\ln(m)}$), the predicted mortality rates are strictly positive, avoiding the biological impossibility of negative death rates.
- **Cluster**: CHE, SWE, NOR, DEUTW, NLD, JPN.
- **Time Window**: 1956-2021. 
    - *Decision*: We truncated the historical series to include West Germany (DEUTW), ensuring a representative European cluster.
    - *Technical Note*: A common time window is a mathematical prerequisite for the Li-Lee Common Factor Model to calculate a balanced average trend across all populations.
- **Age Range**: 0-90.
    - *Decision*: Capped at 90 to avoid "oldest-old" volatility. High-age mortality (90+) often suffers from low exposure, leading to erratic $m_x$ values.
- **Data Integrity**: Implemented a $10^{-10}$ epsilon for log-mortality transformation.
    - *Reasoning*: Smaller populations occasionally report zero deaths for specific cells. Epsilon ensures numerical stability as $\ln(0)$ is undefined.

### 1.1 Cluster Rationale (Coherence vs. Volume)
- **Why 6 countries?**: While increasing the number of populations (e.g., adding Denmark, Austria, or Belgium) might reduce statistical variance, it risks "trend dilution." 
- **The "Pure Signal" Strategy**: We selected a "High-Longevity Gold Standard Cluster." Including countries like the USA or UK would introduce structural breaks (e.g., the opioid crisis or distinct social health shocks) that contaminate the "frontier" mortality signal shared by the selected nations.
- **Data Quality**: Countries like SWE, CHE, and JPN possess the most reliable long-term historical records in the Human Mortality Database (HMD).
- **XAI Benefit**: A focused, high-performing cluster allows for clearer interpretability. In the future LSTM/XAI phase, mapping reciprocal influences is more effective when the underlying populations belong to a coherent socio-economic and medical system.

## 2. Preliminary Observations (EDA & LC)
- **Norway Volatility (Fig. 03)**: Norway shows significant instability in its mortality index ($k_t$) post-2010.
    - *Statistical Insight*: Smaller populations have higher variance; few deaths can shift a rank-1 SVD model significantly.
    - *Demographic Insight*: Evidence of a "harvesting effect"—exceptionally low mortality (2011-2015) followed by a rebound.
- **The 2011 Inflection**: Most countries exhibit a "deceleration gap"—a change in the slope of $k_t$ around 2011.
- **Japan's Catch-up**: Japan started with the highest mortality in 1956 but achieved the fastest improvement, becoming the global longevity leader by the 1980s.

## 3. Lee-Carter Baseline (Independent Modeling)
- **Mathematical Framework**: $\ln(m_{x,t}) = a_x + b_x k_t + \epsilon_{x,t}$
- **Parameter Extraction via SVD**:
    1. **$a_x$ (Age Profile)**: Calculated as the mean of log-mortality over time for each age: $a_x = \frac{1}{T} \sum_t \ln(m_{x,t})$. It represents the "biological identity" of the population.
    2. **Centering**: We subtract $a_x$ from the matrix to isolate the time-varying components.
    3. **SVD (Singular Value Decomposition)**: The centered matrix is decomposed into $U \Sigma V^T$. 
        - $k_t$ (Time Trend) is derived from the first right singular vector ($V$).
        - $b_x$ (Age Sensitivity) is derived from the first left singular vector ($U$).
    4. **$\epsilon_{x,t}$**: The residual representing noise or higher-order dynamics not captured by the first principal component.
- **Identifiability Constraints**: To avoid infinite combinations of $b_x$ and $k_t$ yielding the same product, we impose $\sum_x b_x = 1$. This "anchors" the sensitivity scale, ensuring $k_t$ captures the full magnitude of the time trend.
- **Core Limitation**: Independent modeling allows for "divergent forecasts," where similar countries reach biologically implausible differences in future life expectancy.

## 4. Multi-Population Strategy: Li-Lee (2005)
- **The Theory**: Li-Lee expands Lee-Carter by assuming mortality is composed of a **Common Factor** (shared by the cluster) and a **Specific Factor** (local deviation).
- **Mathematical Framework**: $\ln(m_{x,t,i}) = a_{x,i} + B_x K_t + b_{x,i} k_{t,i} + \epsilon_{x,t,i}$

### 4.1 Common Factor Extraction
- **The Consensus Matrix**: Computed by averaging log-mortality matrices across all $N$ countries: $\bar{M} = \frac{1}{N} \sum_i \ln(m_{x,t,i})$. This represents the "Super-Population" trend.
- **Foundation**: $K_t$ and $B_x$ are extracted via SVD from this average matrix.
- **Constraint ($\sum B_x = 1$)**: Crucial for cross-cluster interpretability. It ensures that a unit change in $K_t$ reflects a unit change in the cluster's average log-mortality. Without this, $K_t$ would be unscalable and incomparable.
- **Denoising Effect**: The Common Factor acts as a robust signal, filtering out local anomalies (like Norway's volatility).
- **Observation on Deceleration**: The common trend confirms that the post-2011 deceleration is a systemic shift across the entire cluster.

### 4.2 Specific Factors (In Progress)
- **The Residual Principle**: After removing the common components ($a_{x,i} + B_x K_t$) from each country's matrix, we apply a second SVD to the residuals to find $b_{x,i}$ and $k_{t,i}$.
- **Coherence**: A fundamental requirement of Li-Lee is that specific deviations ($k_{t,i}$) must be **stationary (mean-reverting)**. This prevents long-term forecast divergence by ensuring countries "orbit" the common trend.

## 5. Alternative Models & Future Steps
- **CBD (Cairns-Blake-Dowd)**:
    - *Focus*: Specifically designed for higher ages (65-90).
    - *Theory*: Instead of age-specific parameters ($b_x$), it uses a logistic-link function where mortality is modeled as: $logit(q_{x,t}) = \kappa_t^{(1)} + \kappa_t^{(2)}(x - \bar{x})$. 
    - *Purpose*: Captures changes in the "intercept" (general level) and "slope" (rate of aging) of the mortality curve.
- **LSTM (Deep Learning)**:
    - *Proposed Innovation*: Recurrent Neural Networks designed to capture long-term dependencies in time series.
    - *Research Goal*: To determine if an LSTM can implicitly learn the "Common Factor" and "Mean-Reverting Residuals" without the rigid linear and stationary constraints of Li-Lee, potentially offering better adaptation to the post-2011 structural break.