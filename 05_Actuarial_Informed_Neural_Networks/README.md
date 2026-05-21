# Project 05: Actuarial-Informed Neural Networks
## *Constrained Deep Learning for Multi-Population Mortality Forecasting*

This project develops an alternative internal model for multi-population longevity forecasting that embeds actuarial domain knowledge directly into the neural network training process. It serves as a **challenger model** to both the Li-Lee benchmark and the unconstrained LSTM framework ([Project 04](../04_Multi_Population_Longevity_XAI)), satisfying the governance, robustness, and explainability requirements expected by regulators (FINMA/EIOPA) for internal model validation.

## Research Question
*Can physics-informed actuarial constraints embedded in the training loss improve neural mortality forecasting — particularly for near-linear populations where unconstrained LSTMs show no advantage — while preserving the model's ability to capture non-linear regime shifts?*

## Key Innovations

### 1. Constrained Loss Function (AINN)
Actuarial constraints embedded as differentiable penalties during training:
- **Coherence**: soft penalty on divergence of country-specific factors (Li-Lee assumption as regulariser).
- **Monotonicity**: temporal mortality improvement enforced during training (mortality should improve over time in developed countries).
- **Stationarity**: tested and excluded — inconsistent with data (4/6 countries violate stationarity).

### 2. Joint Male/Female Training
A single model trained on Male and Female mortality jointly, with a binary sex indicator as input feature. This doubles the effective sample size (90 training samples vs 45) and produces coherent sex-specific forecasts.

### 3. Joint Bayesian Optimisation (Architecture + Constraints)
Architecture hyperparameters (units, learning rate) and constraint weights (λ) are optimised simultaneously over 100 Bayesian trials. This reveals an interaction between architecture capacity and constraint strength that sequential tuning would miss.

**Champion**: LSTM (64-8 units), lr=0.01, λ_coherence=0.001, λ_monotonicity=0.001.
**RMSE**: 7.0687 (overall), 6.65 (Male), 7.46 (Female). Multi-seed CV: 1.23%.

### 4. Regulatory-Grade Robustness
- Multi-seed robustness (CV = 1.23%, PASS).
- Lookback sensitivity analysis.
- Rolling-window validation (Notebook 05).

### 5. True Model-Based Stress Test
Mortality shocks translated into the ΔK_t domain and injected into the recursive forecast (Notebook 06).

## Project Structure
- `data/`: Mortality data assets (HMD, same cluster as Project 04).
- `models/`: Serialized AINN models and scalers.
- `notebooks/`:
    - `01_data_and_baseline.ipynb`: Data loading, EDA, log-mortality matrices. ✓
    - `02_actuarial_benchmarking.ipynb`: Li-Lee sex-specific, stationarity analysis. ✓
    - `03_training_ablation_lambda.ipynb`: Joint Bayesian Optimisation, multi-seed robustness, lookback sensitivity. ✓
    - `04_forecasting_xai_and_results.ipynb`: Recursive forecasting, life expectancy, SHAP.
    - `05_rolling_window_validation.ipynb`: Rolling-window robustness protocol.
    - `06_model_based_stress_test.ipynb`: Shock injection in ΔK_t domain.
    - `07_fat_tail_uncertainty_scr.ipynb`: Non-Gaussian process noise and SCR comparison.
- `src/`: Modular source code (custom losses, reproducibility, styling).
- `reports/figures/`: High-resolution visualizations.
- `latex/`: Paper source files.
- `RESEARCH_NOTES.md`: Methodological journal.
- `ROADMAP.md`: Research plan and execution phases.
- `MODEL_PASSPORT.md`: Governance report for regulatory audit.
- `requirements.txt`: Pinned dependencies.

## Standards & Methodology
- **Cluster**: CHE, SWE, NOR, DEUTW, NLD, JPN (1956-2020).
- **Source**: Human Mortality Database (HMD).
- **Benchmarks**: Li-Lee (2005), Project 04 LSTM+MBC.
- **Validation**: Rolling-window, multi-seed, biological monotonicity audit.
- **Governance**: Model Passport, SHAP, Lexis Maps, Reverse Stress Test.
- **Financials**: SST (Expected Shortfall) and Solvency II (VaR) standards.
- **Design**: Viridis colour palette; Helvetica typography.

## Status
🚧 In development. Notebooks 01-03 complete (data, Li-Lee benchmarking, AINN training). Proceeding to Notebook 04 (Forecasting & Results).
