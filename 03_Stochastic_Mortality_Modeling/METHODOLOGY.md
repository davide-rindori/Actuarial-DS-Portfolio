# Methodology: Neural Longevity Framework
## Advanced Probabilistic Forecasting and Uncertainty Quantification (CHE 1950-2024)

**Author:** Davide Rindori  
**Strategic Focus:** Longevity Trend Risk, Solvency II Internal Models, and Basis Risk Analysis  

---

## 1. Executive Summary & Business Context
Traditional actuarial benchmarks, such as the Lee-Carter model, rely on the assumption of a constant rate of mortality improvement. However, recent data from high-income countries like Switzerland shows a non-linear deceleration—a "regime change"—that linear models fail to capture, leading to potential under-reserving.

This project implements a **Bayesian-style Deep Learning framework** using **Long Short-Term Memory (LSTM)** networks and **Monte Carlo Dropout**. The objective is to move beyond point-estimates to a full distribution of outcomes, providing a rigorous statistical basis for **Longevity SCR (Solvency Capital Requirement)** and pricing of longevity swaps.

## 2. Data Strategy & Hydrological Consistency
### 2.1 Swiss Cohort Granularity
The dataset is sourced from the **Human Mortality Database (HMD)**, covering the Swiss population from 1950 to 2024. We focus on central death rates ($m_x$) across a complete age-grid (0-95), allowing the model to learn the "rectangularization" of the survival curve over seven decades.

### 2.2 Feature Engineering for Trend Stability
* **Log-Space Graduation:** Mortality rates are transformed into $\log(m_x)$ to linearize exponential age-effects and optimize the neural network for relative improvements (mortality reduction factors).
* **Decennary Temporal Smoothing:** A **10-year sliding window** was implemented. In life reinsurance, capturing long-term structural trends is more critical than reacting to idiosyncratic annual noise (e.g., severe flu seasons). This window size forces the LSTM to learn decadal momentum.

## 3. The "Mortality Derby": Benchmarking Architectures
We conducted an extensive backtest (2011-2024) comparing four distinct modeling philosophies:

1. **SVD Lee-Carter (Actuarial Baseline):** A deterministic decomposition into $\alpha_x$, $\beta_x$, and $\kappa_t$. This serves as the pricing floor and the interpretability anchor.
2. **Multi-Layer Perceptron (MLP):** A static deep learning approach focused on capturing complex non-linearities between Age and Year without temporal memory.
3. **Hybrid Residual Model:** A two-stage architecture where the neural network learns only the residuals (unexplained variance) of the Lee-Carter model. This "Expert-in-the-loop" approach ensures biological consistency while improving precision.
4. **LSTM Champion (The Winning Architecture):** A recurrent model that treats the mortality index ($\kappa_t$) as a sequence. The LSTM proved superior in identifying the post-2010 "stagnation" in Swiss longevity improvements.

## 4. Probabilistic Inference & Risk Diagnostics
### 4.1 Monte Carlo Dropout (Epistemic Uncertainty)
To quantify **Model Risk**, we implemented **Monte Carlo Dropout**. By keeping dropout layers active during the prediction phase (`training=True`), we performed 100 stochastic forward passes. This generates an ensemble of possible future trajectories, where the variance represents our "model doubt." This spread is essential for calculating **Value-at-Risk (VaR)** in capital management.

### 4.2 Diagnostic Heatmap (Basis Risk Analysis)
The final model was validated through an **Age-Period Residual Map**. By analyzing the scart (actual - predicted) across all ages and years, we ensured:
* **No Systematic Bias:** Residuals are distributed as white noise, confirming the model hasn't missed structural signals.
* **Coorte Effect Validation:** Diagonal patterns were checked to ensure that specific generations (e.g., Baby Boomers) do not exhibit unexplained mortality shifts that could lead to basis risk in portfolio pricing.

## 5. Results & Strategic Implications
The LSTM framework achieved a **33% reduction in RMSE** compared to the Lee-Carter baseline. For a 65-year-old in 2025, the model provides a 95% confidence corridor for survival up to age 90, enabling a more granular assessment of the "1-in-200" stress scenarios required by regulators.

## 6. Future Roadmap
* **Multi-Population Basis Risk (Li-Lee Model):** Integrating Swiss data with larger European cohorts to manage trend divergence.
* **Cause-of-Death (CoD) Drivers:** Incorporating exogenous drivers like smoking prevalence and cardiovascular health into the neural input layer.