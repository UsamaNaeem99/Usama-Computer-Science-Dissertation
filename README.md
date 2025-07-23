# Fairness-Aware Machine Learning Evaluation on COMPAS and German Credit Datasets

This repository contains the full implementation of experiments conducted to assess **bias and fairness** in machine learning models using the **COMPAS** and **German Credit** datasets. The models were trained using traditional and fairness-enhanced techniques and evaluated based on performance and fairness metrics.

## 📁 Contents

- `data/`: Raw and processed datasets (COMPAS and German Credit)
- `notebooks/`: Jupyter Notebooks for all models and bias mitigation techniques
- `results/`: Performance and fairness metrics summaries
- `models/`: Model definitions (Logistic Regression, Random Forest, FFNN)
- `README.md`: This documentation file

---

## 📊 Datasets Used

### 1. **COMPAS Dataset** (from ProPublica via Kaggle)
- Target Variable: `two_year_recid` (binary recidivism prediction)
- Protected Attribute: `race_binary` (0 = White, 1 = Black)

### 2. **German Credit Dataset** (from UCI repository)
- Target Variable: `kredit` (1 = Good Credit, 0 = Bad Credit)
- Protected Attribute: `sex_binary` (0 = Female, 1 = Male)

---

## 🧠 Models Trained

Each model was evaluated both with and without fairness-enhancing interventions:

- Logistic Regression
- Random Forest
- Feedforward Neural Network (FFNN)

---

## 🛠️ Bias Mitigation Techniques

### 1. **Preprocessing**
- **Reweighing**: Adjusts weights of training samples to mitigate bias

### 2. **In-processing**
- **Adversarial Debiasing** (via AIF360): Uses adversarial training to minimize dependence on protected attributes

### 3. **Post-processing**
- **Equalized Odds Postprocessing**: Adjusts output labels to equalize true positive rates across groups

---

## ✅ Evaluation Metrics

### Performance Metrics:
- Accuracy
- Precision
- Recall
- F1 Score

### Fairness Metrics:
- Statistical Parity Difference (SPD)
- Equal Opportunity Difference (EOD)
- Disparate Impact Ratio (DIR)
- Selection Rate by group

---

## 📈 Notable Results Summary

| Dataset | Model | Accuracy | SPD | EOD | DIR |
|--------|--------|----------|------|------|------|
| COMPAS | Logistic Regression | 0.7994 | 0.2703 | 0.1759 | 2.0366 |
| COMPAS | Reweighed FFNN | 0.8028 | 0.2120 | 0.1132 | 1.8269 |
| COMPAS | Equalized Odds (FFNN) | 0.7433 | 0.0863 | 0.0059 | 1.2372 |
| German | Adversarial Debiasing (FFNN) | 0.7567 | 0.2243 | 0.5465 | 1.3043 |
| German | Reweighed Logistic Regression | 0.7233 | 0.0148 | 0.0051 | 1.0150 |
| German | Equalized Odds (Random Forest) | 0.7667 | 0.0163 | 0.0286 | 1.0173 |

> *See the full table in the results section for all configurations.*

---

## 📦 Requirements

- Python 3.11
- `pandas`, `numpy`, `scikit-learn`, `tensorflow==1.15`, `fairlearn`, `aif360`, `matplotlib`, `seaborn`

Install all dependencies:

```bash
pip install -r requirements.txt
