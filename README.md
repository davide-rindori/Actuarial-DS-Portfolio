# Actuarial Data Science Portfolio
## Davide Rindori — SAV Actuarial Candidate, Data Scientist & PhD in Physics

Welcome to my professional portfolio. This repository showcases a collection of projects where Actuarial Science meets modern Data Engineering and Machine Learning, with a specific focus on Risk Modeling, Longevity, and NatCat analysis.

---

## 📂 Project Index

### 3. [Stochastic Longevity Forecasting: A Neural Approach to SST Capital Calibration](./03_Stochastic_Mortality_Modeling)
**Focus:** *Life & Health Reinsurance, LSTM Networks, Swiss Solvency Test (SST)*
* Developed a framework for Swiss mortality forecasting, replacing linear drift assumptions with **LSTM-based sequential modeling**.
* Identified a **50-point "Prudence Gap"** and quantified Model Risk by benchmarking neural projections against traditional Lee-Carter benchmarks.
* Calibrated the **Longevity SCR Shock (ES 99%) at 3.48** via Monte Carlo Dropout, ensuring robust tail-risk estimation for regulatory capital compliance.
* Validated architectural stability through a comprehensive sensitivity analysis of temporal windows and neural hyperparameters.

### 2. [Zurich Extreme Precipitation Analysis (EVT)](./02_Climate_Risk_EVT)
**Focus:** *Climate Risk, Extreme Value Theory (EVT), Big Data Engineering*
* Modeled "Tail Risk" for catastrophic flooding events in Zurich using 44 years of ERA5 hourly climate data.
* Engineered a pipeline to process **~385,000 observations**, identifying a **43% underestimation** in risk intensity when using low-resolution data snapshots.
* Fitted **Gumbel distributions** to estimate 100-year return levels, providing key inputs for PML (Probable Maximum Loss) and Solvency assessments.

### 1. [Motor Pricing & Actuarial Interpretability](./01_Motor_Pricing_Interpretability)
**Focus:** *P&C Pricing, Gradient Boosting (XGBoost), Explainable AI (XAI)*
* Implemented a frequency-severity pricing framework, achieving a **4.67% improvement** in Poisson deviance over traditional GLM benchmarks.
* Leveraged **SHAP (Shapley Additive Explanations)** to ensure model transparency and decompose non-linear risk factor interactions.
* Focused on bridging the gap between "Black Box" Machine Learning and actuarial requirements for model governance under Solvency II standards.

---

## 🛠️ Skills & Tools
* **Actuarial:** Stochastic Longevity modeling, Extreme Value Theory (EVT), GLM, Pricing, SCR Calibration.
* **Technical:** Python (Pandas, Xarray, Scikit-learn, PyTorch/TensorFlow), SQL, Git, API Data Engineering.
* **Domain:** Swiss Solvency Test (SST), Solvency II, NatCat Modeling, Model Risk Management.

---

## 🔬 Education & Certification
* **SAV Actuarial Candidate:** Actively pursuing certification with the Swiss Association of Actuaries.
* **PhD in Physics — University of Florence:** Specialized in complex systems and advanced mathematical modeling.

---

## 📫 Contact
* **LinkedIn:** [linkedin.com/in/davide-rindori/](https://www.linkedin.com/in/davide-rindori/)
* **Email:** [rindori.d@gmail.com](mailto:rindori.d@gmail.com)

---
*This portfolio is continuously updated with new research and technical implementations.*