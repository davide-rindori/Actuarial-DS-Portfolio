# Project 04: Multi-Population Longevity Forecasting with XAI

Research pipeline for forecasting mortality rates across a 6-country cluster (CHE, SWE, NOR, DEUTW, NLD, JPN) using classical actuarial models (Lee-Carter, Li-Lee) and Deep Learning (LSTM) with Explainable AI (XAI).

## Key Visualizations
- **Historical Trends**: Comparative analysis of mortality deceleration post-2011.
- **Stationarity Paradox**: Visual evidence of non-stationary residuals in core European populations despite visual stability.
- **CBD Dynamics**: Rotation and steepening of the mortality curve for ages 65-90.

## Research Focus
The project evaluates the "Gold Standard" of actuarial literature against Deep Learning architectures, focusing on the **post-2011 mortality deceleration** and the empirical failure of classical coherence assumptions.

### Model Benchmarks
- **Lee-Carter (1992)**: Baseline independent stochastic modeling via SVD.
- **Li-Lee (2005)**: Multi-population coherent modeling. Our analysis reveals structural breaches in stationarity assumptions for core countries (SWE, CHE, DEUTW).
- **CBD (2006)**: Parametric modeling (intercept/slope) for the 65-90 age bracket, identifying non-linear aging acceleration.
- **Innovation**: **LSTM (Long Short-Term Memory)** networks designed to capture the persistent drifts and non-linearities identified in the benchmarking phase, interpreted via **XAI (SHAP/Integrated Gradients)**.

## Project Structure
- `data/`: Processed mortality data and stationarity test reports.
- `notebooks/`: 
    - `01_data_extraction_and_eda.ipynb`: Data ingestion and professional EDA.
    - `02_actuarial_benchmarking.ipynb`: Full implementation of LC, Li-Lee (with ADF/KPSS conflict analysis), and CBD. (**Completed**).
    - `03_lstm_modeling.ipynb`: Neural network architecture and training (**Next Step**).
- `reports/figures/`: High-resolution visualizations (Viridis/Helvetica).
- `RESEARCH_NOTES.md`: Detailed methodological journal, mathematical foundations, and statistical paradox analysis.

## Standards & Methodology
- **Cluster**: CHE, SWE, NOR, DEUTW, NLD, JPN (1956-2021).
- **Source**: Human Mortality Database (HMD).
- **Validation**: Confirmatory Stationarity Analysis (Augmented Dickey-Fuller & KPSS tests).
- **Design**: Viridis palette for perceptually uniform visualizations; Helvetica typography.