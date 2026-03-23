# Stochastic Longevity Forecasting: A Neural Approach to SST Capital Calibration

![Project Status](https://img.shields.io/badge/Status-Complete-green)
![Target Industry](https://img.shields.io/badge/Industry-Life_Reinsurance-blue)
![Regulatory Focus](https://img.shields.io/badge/Compliance-SST--Solvency_II-red)

**Author:** [Davide Rindori, PhD](https://www.linkedin.com/in/davide-rindori/)  
**Technical White Paper:** [Download PDF (Overleaf/Main.pdf)](#)  
**Research Goal:** Modernizing Longevity Risk frameworks for high-income markets through Deep Learning and Regulatory Capital standards.

---

## Executive Summary
This repository implements a **State-of-the-Art Longevity Risk Framework** designed to address the limitations of linear mortality drift in modern demographic regimes. By transitioning from traditional benchmarks to **Probabilistic Deep Learning (LSTM + Monte Carlo Dropout)**, this framework identifies the recent Swiss mortality "regime shifts" and quantifies the **Prudence Gap** required for capital adequacy under **Swiss Solvency Test (SST)** and **Solvency II** standards.

## Key Technical Features
* **Sequential Intelligence:** LSTM-based modeling with a 10-year look-back window to internalize decadal mortality momentum and structural plateaus.
* **Bayesian Uncertainty:** Quantification of **Epistemic Risk** through Monte Carlo Dropout, generating 100 stochastic forward passes for robust tail-risk estimation.
* **Regulatory Calibration:** Automated derivation of **Expected Shortfall (ES 99%)** and **Value-at-Risk (VaR)** for Longevity SCR shock calibration.
* **Audit-Ready Pipeline:** A modular 4-stage workflow ensuring full transparency from raw Human Mortality Database (HMD) ingestion to final risk reporting.

## Visual Insights & Benchmarking

### 1. The "Mortality Derby" (Model Selection)
The framework was validated via out-of-sample backtesting (2011-2024). The LSTM outperformed classical SVD-based benchmarks by adapting to the recent deceleration in mortality improvements, preventing the systematic over-optimism inherent in the Lee-Carter model.

| Model Architecture | RMSE (2011-2024) | Risk Management Insight |
| :--- | :--- | :--- |
| **SVD Lee-Carter (Baseline)** | 0.1682 | Rigid linear drift; fails to detect the post-2010 plateau. |
| **Hybrid Residual Model** | 0.1536 | Improved graduation but maintains a linear extrapolation bias. |
| **LSTM Champion** | **0.1174** | **Superior adaptation to Swiss regime shifts and decadal memory.** |

![Model Derby](reports/figures/04_model_derby_comparison.png)

### 2. Model Risk & The "55-Point Prudence Gap"
A core finding of this study is the divergence between neural and stochastic benchmarks. By 2050, the **LSTM Median Projection is 55.27 points higher** than the Lee-Carter trend. This gap quantifies the **Model Risk**: relying on linear assumptions in a post-linear era leads to significant under-reserving of longevity liabilities.

![Model Risk Benchmark](reports/figures/08_lstm_vs_lc_benchmark.png)

### 3. Capital Requirements (SST Metrics)
Following Swiss Re and FINMA regulatory standards, the framework generates a full probability distribution for the 2050 mortality index ($k_t$). We derive the **Expected Shortfall (ES 99%)** to calibrate the required capital buffer.

* **Best Estimate (Median $k_t$):** -65.14
* **Expected Shortfall (ES 99%):** -67.86
* **Longevity SCR Shock ($\Delta k_t$):** **2.58**

![SST Metrics](reports/figures/09_longevity_risk_distribution.png)

## Repository Structure
* `01_actuarial_baseline_prep.ipynb`: Data ingestion (HMD) and SVD-based parameter extraction ($\alpha_x, \beta_x, k_t$).
* `02_neural_model_selection.ipynb`: Competitive benchmarking, Hyperparameter tuning, and Backtesting.
* `03_model_diagnostics_validation.ipynb`: Residual Heatmaps, Error Analysis, and Actuarial Stress Testing.
* `04_stochastic_projections_capital_metrics.ipynb`: MC Dropout inference, SST calibration, and Final Risk Metrics.

## License & Citation
This project is licensed under the **MIT License**.

**Citation:** *Davide Rindori, PhD. "Stochastic Longevity Forecasting: A Neural Approach to SST Capital Calibration" (2026).*

---
## Contact & Author

**Davide Rindori, PhD** *Data Scientist / Risk Modeller* **Expertise:** Machine Learning for Life Reinsurance & Longevity Risk Uncertainty.  
**LinkedIn:** [linkedin.com/in/davide-rindori/](https://www.linkedin.com/in/davide-rindori/)

*This project was developed as a benchmark study for modernizing longevity risk frameworks in high-income markets, focusing on the integration of Deep Learning with regulatory capital standards.*