# Project 04: Multi-Population Longevity Forecasting with XAI

This repository contains the research pipeline for forecasting mortality rates across a 6-country cluster (CHE, SWE, NOR, DEUTW, NLD, JPN) using a combination of classical actuarial models (Lee-Carter, Li-Lee) and Deep Learning (LSTM) with Explainable AI (XAI) interpretations.

## Project Structure
- `data/`: Raw and cleaned mortality data from the Human Mortality Database (HMD).
- `notebooks/`: 
    - `01_data_extraction_and_eda.ipynb`: Data ingestion and historical trend analysis.
    - `02_actuarial_benchmarking.ipynb`: Lee-Carter & Li-Lee model implementation. (In Progress)
- `reports/figures/`: High-resolution visualizations (Viridis/Helvetica standard).
- `src/`: Core functions for the Li-Lee and LSTM models.

## Methodology
The study focuses on the "Deceleration Gap" observed in mortality trends post-2011, comparing the robustness of the **Li-Lee (2005)** common factor model against a specialized **LSTM** architecture.

## Setup
1. Clone the repository.
2. Download `.txt` files from [mortality.org](https://www.mortality.org) for the selected cluster.
3. Place files in `data/raw/` following the `CODE_mx.txt` and `CODE_ex.txt` convention.
4. Run the notebooks in sequential order.