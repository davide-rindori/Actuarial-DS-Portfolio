# Project 05: Actuarial-Informed Neural Networks
## *Constrained Deep Learning for Multi-Population Mortality Forecasting*

This project develops an alternative internal model for multi-population longevity forecasting that embeds actuarial domain knowledge directly into the neural network training process. It serves as a **challenger model** to both the Li-Lee benchmark and the unconstrained LSTM framework ([Project 04](../04_Multi_Population_Longevity_XAI)), satisfying the governance, robustness, and explainability requirements expected by regulators (FINMA/EIOPA) for internal model validation.

## Research Question
*Can physics-informed actuarial constraints embedded in the training loss improve neural mortality forecasting — particularly for near-linear populations where unconstrained LSTMs show no advantage — while preserving the model's ability to capture non-linear regime shifts?*

## Key Innovations

### 1. Constrained Loss Function (AINN)
Actuarial constraints embedded as differentiable penalties during training:
- **Coherence**: soft penalty on divergence of country-specific factors (Li-Lee assumption as regulariser).
- **Monotonicity**: Gompertzian compliance enforced during training, not verified post-hoc.
- **Stationarity**: penalise unit-root behaviour in predicted specific factors.

### 2. MBC as Bayesian Shrinkage
Formalisation of Mean-Bias Correction in learning-theoretic terms, connecting to credibility theory and Bayesian shrinkage toward the Li-Lee drift prior.

### 3. Regulatory-Grade Robustness
- Multi-seed robustness table (5+ seeds).
- Rolling-window validation (expanding windows).
- Fat-tail process noise (Student-t, bootstrap) with SCR comparison.

### 4. True Model-Based Stress Test
Mortality shocks translated into the ΔK_t domain and injected into the recursive forecast, quantifying the model's transient response function for FINMA/EIOPA validation.

## Project Structure
- `data/`: Mortality data assets (HMD, same cluster as Project 04).
- `models/`: Serialized AINN models and scalers.
- `notebooks/`:
    - `01_data_and_baseline.ipynb`: Data reproduction and Li-Lee baseline.
    - `02_constrained_loss_design.ipynb`: AINN loss function implementation and λ-sweep.
    - `03_training_ablation_lambda.ipynb`: Training with constraints, ablation studies.
    - `04_credibility_blending_mbc_theory.ipynb`: Credibility weighting and MBC formalisation.
    - `05_rolling_window_multiseed.ipynb`: Robustness validation protocol.
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
🚧 In development.
