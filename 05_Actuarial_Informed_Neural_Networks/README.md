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

### 3. Joint Bayesian Optimisation via Optuna (6D)
Architecture (units, learning rate), temporal context (lookback window), and actuarial constraints (λ values) are optimised simultaneously in a single 100-trial joint search using Optuna's TPE sampler. This is methodologically superior to sequential tuning and reveals that the optimal constraint weight is architecture-dependent.

**Champion**: LSTM (48-32 units), lookback=15, lr=0.001, λ_coherence=0.001, λ_monotonicity=0.001.
**RMSE**: 6.1725 (overall), 5.72 (Male), 6.59 (Female). Multi-seed CV: 8.90% (PASS).
**Runtime**: 48 minutes on Apple M1 Pro.

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
    - `04_stochastic_forecasting.ipynb`: MC Dropout forecasting, observation-anchored e0, MBC analysis. ✓
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
🚧 In development. Notebooks 01-04 complete (data, Li-Lee, AINN training, stochastic forecasting). Proceeding to Notebook 05 (XAI & Validation).
