# Project 04: Multi-Population Longevity Forecasting with XAI

Research pipeline for forecasting mortality rates across a 6-country cluster (CHE, SWE, NOR, DEUTW, NLD, JPN) using classical actuarial models (Lee-Carter, Li-Lee) and Deep Learning (LSTM) with Explainable AI (XAI).

## Key Visualization
![Figure 1](reports/figures/fig01_comparison_age_65.png)

## Research Focus
The project compares the "Gold Standard" of actuarial literature with Deep Learning architectures, focusing on the **post-2011 mortality deceleration** observed in advanced economies.

- **Actuarial Benchmarks**: 
    - **Lee-Carter (1992)**: Independent stochastic modeling.
    - **Li-Lee (2005)**: Multi-population coherent modeling to prevent forecast divergence.
    - **CBD (2006)**: Specialized modeling for the 65-90 age bracket.
- **Innovation**: **LSTM (Long Short-Term Memory)** networks to capture non-linearities and cross-population influences, interpreted via **XAI (SHAP/Integrated Gradients)**.

## Project Structure
- `data/`: Processed mortality data (Raw .txt files are git-ignored).
- `notebooks/`: 
    - `01_data_extraction_and_eda.ipynb`: Data ingestion and professional EDA.
    - `02_actuarial_benchmarking.ipynb`: Lee-Carter & Li-Lee implementation (**In Progress**).
- `reports/figures/`: High-resolution visualizations (Viridis/Helvetica/Centered).
- `RESEARCH_NOTES.md`: Detailed methodological journal and mathematical foundations.

## Standards
- **Palette**: Viridis (Perceptually Uniform).
- **Typography**: Helvetica / Sans-Serif.
- **Source**: Human Mortality Database (HMD).
- **Cluster**: CHE, SWE, NOR, DEUTW, NLD, JPN (1956-2021).