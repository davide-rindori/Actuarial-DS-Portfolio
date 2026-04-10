# Actuarial Data Science Portfolio
## Davide Rindori — SAV Actuarial Candidate, Data Scientist & PhD in Physics

Welcome to my professional portfolio. This repository showcases a collection of projects where Actuarial Science meets modern Data Engineering and Machine Learning, with a specific focus on Risk Modeling, Longevity, and NatCat analysis.

---

## 📂 Project Index

### 4. [Neural Multi-Population Mortality: Beyond Linear Coherence](./04_Neural_MultiPopulation_Longevity)
**Focus:** *Global Longevity Risk, Hierarchical LSTM, XAI (SHAP), Model Governance*
* **Core Output:** Developed a hybrid **LSTM+MBC (Mean-Bias Correction)** framework for a 6-country frontier cluster (CHE, SWE, NOR, DEUTW, NLD, JPN).
* **Innovation:** Outperformed the Li-Lee "Gold Standard" by **21.88% in Japan** and **17.28% in Sweden**, effectively capturing non-linear regime shifts post-2011.
* **Risk Governance:** Quantified a high-precision **SCR Risk Margin of ±0.037 years** (95% CI) for Switzerland and implemented **SHAP influence mapping** to ensure regulatory auditatibility (SST/Solvency II).
* **Documentation:** Includes a comprehensive **[Model Passport](./04_Neural_MultiPopulation_Longevity/MODEL_PASSPORT.md)** and detailed **[Research Notes](./04_Neural_MultiPopulation_Longevity/RESEARCH_NOTES.md)**.

### 3. [Stochastic Longevity Forecasting: A Neural Approach to SST](./03_Stochastic_Mortality_Modeling)
**Focus:** *Life & Health Reinsurance, Deep Learning, Swiss Solvency Test (SST)*
* **Core Output:** Developed an LSTM-based framework for Swiss mortality, quantifying a **3.90 Longevity SCR Shock**.
* **Model Risk:** Identified a **38.5-point "Prudence Gap"** by benchmarking neural projections against traditional linear drift assumptions.
* **Documentation:** Includes a full **Technical Paper** calibrating capital buffers under Swiss Solvency Test standards.

### 2. [Zurich Extreme Precipitation Analysis (EVT)](./02_Climate_Risk_EVT)
**Focus:** *Climate Risk, Extreme Value Theory (EVT), Big Data Engineering*
* Modeled "Tail Risk" for catastrophic flooding events in Zurich using 44 years of ERA5 hourly climate data.
* Engineered a pipeline to process **~385,000 observations**, identifying a **43% underestimation** in risk intensity when using low-resolution data snapshots.
* Fitted **Gumbel distributions** to estimate 100-year return levels, providing key inputs for PML and Solvency assessments.

### 1. [Motor Pricing & Actuarial Interpretability](./01_Motor_Pricing_Interpretability)
**Focus:** *P&C Pricing, Gradient Boosting (XGBoost), Explainable AI (XAI)*
* Implemented a frequency-severity pricing framework, achieving a **4.67% improvement** in Poisson deviance over traditional GLM benchmarks.
* Leveraged **SHAP (Shapley Additive Explanations)** to ensure model transparency and decompose non-linear risk factor interactions.

---

## 🛠️ Skills & Tools
* **Actuarial:** Stochastic Longevity Modeling (Hierarchical/Multi-population), SCR Calibration (VaR/ES), EVT, GLM, Pricing.
* **Technical:** Python (PyTorch/TensorFlow, Scikit-learn, Pandas), Git, LaTeX, SQL.
* **Domain:** Swiss Solvency Test (SST), Solvency II, Model Risk Management (Governance), NatCat Modeling.

---

## 🔬 Education & Certification
* **SAV Actuarial Candidate:** Actively pursuing certification with the Swiss Association of Actuaries.
* **PhD in Physics — University of Florence:** Specialized in complex systems modeling and stochastic processes.

---
## 📫 Contact
* **LinkedIn:** [linkedin.com/in/davide-rindori/](https://www.linkedin.com/in/davide-rindori/)
* **Email:** [rindori.d@gmail.com](mailto:rindori.d@gmail.com)

---
*This portfolio is continuously updated with new research and technical implementations.*