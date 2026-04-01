# Project 04: Neural Multi-Population Mortality
## *Beyond Linear Coherence with LSTM and Explainable AI (XAI)*

### 🎯 Project Objective
To evolve the LSTM-based mortality framework (from Project 03) into a global, auditable, and biologically consistent research paper. This project aims to demonstrate that Deep Learning outperforms classical actuarial models (Lee-Carter, Li-Lee, CBD) in terms of predictive accuracy, cross-population coherence, and financial utility, meeting the standards for an **arXiv preprint**.

---

### 🏗️ The 4 Pillars of Research

#### 1. Global Generalization (Multi-Country Dataset)
The model will be tested on a diversified cluster of **6 countries** to ensure it captures global longevity trends and local "regime shifts" (post-2011 deceleration).
* **DACH Core:** **Switzerland & Germany** (Crucial for SST and local market relevance).
* **Nordic/Strategic:** **Sweden & Norway** (Sweden provides the world's longest high-quality historical series; Norway aligns with strategic career goals in Scandinavia).
* **SOTA Benchmark:** **Netherlands** (Global leaders in pension and longevity modeling).
* **Extreme Case:** **Japan** (The world's oldest population, essential for testing model stability at high ages).

#### 2. SOTA Benchmarking (State-of-the-Art)
Direct challenge between Neural Innovation and Actuarial "Gold Standards":
* **Li-Lee:** Benchmark for multi-population coherence (preventing long-term divergence).
* **CBD (Cairns-Blake-Dowd):** Reference for mortality at older ages (65-90+).
* **Metrics:** Comparative RMSE, MAE, and residual analysis across all frameworks.

#### 3. XAI & Biological Consistency (Transparency & Governance)
Addressing the "Black-Box" problem for regulatory acceptance (FINMA/EIOPA).
* **XAI (SHAP):** Decomposing the LSTM cell state to explain which historical lags or cohorts drive future projections.
* **Monotonicity Tests:** Ensuring mortality curves do not cross (e.g., mortality at 90 must remain higher than at 80).
* **Lexis Maps:** Visualizing residuals to ensure no "phantom patterns" (cohort effects) remain uncaptured.

#### 4. Financial Application (Pricing & Capital)
Translating stochastic projections into economic value.
* **Longevity Swaps:** Pricing a longevity derivative using LSTM-based stochastic trajectories.
* **SST vs. Market:** Comparing the internal model's capital requirement (SCR) with the Market Risk Premium.

---

### 📝 Why this belongs on arXiv?
This project provides three original contributions to the field of Quantitative Actuarial Science:
1.  **Architecture:** Application of LSTM with Bayesian Uncertainty (MC Dropout) to regulatory solvency frameworks.
2.  **Validation:** Extensive multi-population comparison against Li-Lee and CBD across 6 diverse economies.
3.  **Governance:** Systematic use of Explainable AI (XAI) to overcome the "Black Box" hurdle in insurance model validation.

---

### 🚀 Execution Phases
- [ ] **Phase A:** Data extraction from HMD for the 6 target countries.
- [ ] **Phase B:** Implementation of Li-Lee and CBD baseline models.
- [ ] **Phase C:** LSTM training and Multi-Country backtesting.
- [ ] **Phase D:** SHAP analysis, Biological consistency tests, and Paper drafting.