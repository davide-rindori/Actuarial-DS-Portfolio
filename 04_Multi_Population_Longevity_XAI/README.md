# Project 04: Multi-Population Longevity Forecasting with XAI

Research pipeline for forecasting mortality rates across a 6-country cluster (CHE, SWE, NOR, DEUTW, NLD, JPN) using classical actuarial models (Lee-Carter, Li-Lee) and Deep Learning (LSTM) with Explainable AI (XAI).

## Key Visualizations
- **Historical Trends**: Comparative analysis of mortality deceleration post-2011.
- **Stationarity Paradox**: Visual evidence of non-stationary residuals in core European populations despite visual stability.
- **Model Convergence**: Bayesian-optimized learning curves for Hierarchical LSTM.
- **Out-of-Sample Validation**: Performance of LSTM vs. Li-Lee benchmarks on the 2012-2020 window.

## Research Focus
The project evaluates the "Gold Standard" of actuarial literature against Deep Learning architectures, focusing on the **post-2011 mortality deceleration** and the empirical failure of classical coherence assumptions.

### Model Benchmarks & Innovation
- **Lee-Carter (1992)**: Baseline independent stochastic modeling via SVD.
- **Li-Lee (2005)**: Multi-population coherent modeling. Our analysis reveals structural breaches in stationarity assumptions for core countries (SWE, CHE, DEUTW).
- **CBD (2006)**: Parametric modeling (intercept/slope) for the 65-90 age bracket.
- **Hierarchical LSTM**: A neural architecture designed to process both the Common Factor ($K_t$) and Specific Residuals ($k_{t,i}$).
    - **Optimization**: Bayesian Search for optimal depth and learning rates.
    - **Stationarity Pivot**: Transition from absolute levels to **First Differences** ($\Delta K_t$) to eliminate drift bias and handle non-stationarity.
    - **Anti-Leakage Protocol**: Strict separation of training and validation scaling to ensure academic rigor (arXiv-ready).

## Project Structure
- `data/`: Processed mortality data, stationarity reports, and serialized validation results.
- `models/`: Serialized LSTM "Champion" models (.keras) and standardized scalers (.pkl).
- `notebooks/`: 
    - `01_data_extraction_and_eda.ipynb`: Data ingestion and professional EDA.
    - `02_actuarial_benchmarking.ipynb`: Implementation of LC, Li-Lee, and CBD. (**Completed**).
    - `03_lstm_hierarchical_forecasting.ipynb`: Bayesian Tuning, Anti-Leakage Training, and Stationarization. (**Completed**).
    - `04_forecasting_xai_and_results.ipynb`: Recursive projection (2021-2050), Fan Charts, and XAI. (**Next Step**).
- `reports/figures/`: High-resolution visualizations (Viridis/Helvetica).
- `RESEARCH_NOTES.md`: Detailed methodological journal and mathematical foundations.

## Standards & Methodology
- **Cluster**: CHE, SWE, NOR, DEUTW, NLD, JPN (1956-2021).
- **Source**: Human Mortality Database (HMD).
- **Validation**: Out-of-sample testing (2012-2020) with RMSE/MAE metrics on original mortality scales.
- **Tech Stack**: TensorFlow/Keras, Keras Tuner (Bayesian), Scikit-Learn, Joblib.
- **Design**: Viridis palette for perceptually uniform visualizations; Helvetica typography.