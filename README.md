# Customer Churn Prediction Analysis (MATH 516)

This project focuses on predicting customer churn using multiple machine learning and statistical models. The goal is to compare model performance across different approaches and understand how model complexity impacts predictive accuracy on tabular data.

---

## Group Members

- Arun Sharma  
- Umar Mohammed Yousuf  
- Alex Vukovic  

---

## Models Implemented

The following models were used in this project:

- Logistic Regression (R)
- XGBoost (Python)
- FT-Transformer (Python)

These models represent increasing levels of complexity, from simple linear methods to advanced deep learning techniques.

---

## Dataset

The dataset used is the **Telco Customer Churn dataset**, which contains customer-level information such as:

- Demographics  
- Services subscribed  
- Account details  
- Billing information  

The dataset is included in this repository:


data/WA_Fn-UseC_-Telco-Customer-Churn.csv


---

## Project Structure


data/
WA_Fn-UseC_-Telco-Customer-Churn.csv

src/
python/
ft_transformer.py
train_xgboost_telco_ordinal.py

R/
train_logistic_telco.R

README.md
final_report.md
requirements.txt


---

## Evaluation Metrics

The models were evaluated using:

- ROC-AUC  
- PR-AUC  
- Brier Score  
- F1 Score  

A classification threshold of 0.3 was used to better identify churn cases.

---

## Results Summary

| Model                | ROC-AUC | PR-AUC | Brier Score | F1 Score |
|---------------------|--------:|-------:|------------:|---------:|
| Logistic Regression | 0.8557  | 0.6743 | 0.1314      | 0.6399   |
| XGBoost             | 0.8388  | 0.6676 | 0.1575      | 0.5857   |
| FT-Transformer      | 0.8485  | 0.6601 | 0.1364      | 0.6187   |

Logistic Regression achieved the best overall performance across all metrics. While XGBoost and FT-Transformer captured more complex feature interactions, they did not outperform the simpler linear model on this dataset.

---

## Final Report

The complete report is available here:

[View Final Report](final_report.md)

This report includes detailed methodology, model explanations, and FT-Transformer analysis.

## Requirements

To run the Python code:

```bash
pip install -r requirements.txt

For R:

install.packages(c("tidyverse", "caret", "glmnet", "pROC", "PRROC", "DescTools"))

How to Run

Python

python3 src/python/ft_transformer.py
python3 src/python/train_xgboost_telco_ordinal.py

R

source("src/R/train_logistic_telco.R")

Notes:
All code is reproducible using the dataset included in the repository.
File paths are set relative to the project structure.
The project compares both classical statistical models and modern deep learning approaches for tabular data.
Results are reproducible by running the scripts provided in the `src/` directory.
