# Final Report: Customer Churn Prediction

## Introduction

Customer churn prediction is an important problem for telecom companies, as retaining customers is often more cost-effective than acquiring new ones. The goal of this project was to build and compare multiple models to predict whether a customer is likely to churn based on historical data.

We explored three different approaches:
- Logistic Regression (R)
- XGBoost (Python)
- FT-Transformer (Python)

These models represent increasing levels of complexity, allowing us to evaluate how model sophistication impacts performance.

## Dataset

The dataset used is the Telco Customer Churn dataset. It includes customer-level information such as demographics, services subscribed, and billing details.

The target variable is `Churn`, which was converted into a binary variable (1 = Yes, 0 = No).

Basic preprocessing steps included:
- Removing the `customerID` column
- Handling missing values
- Converting categorical variables into usable formats
- Converting `TotalCharges` into numeric form

The dataset was split into training and testing sets using stratified sampling to preserve class distribution.

## Models

### Logistic Regression

Logistic regression was used as a baseline model due to its simplicity and interpretability. Regularization techniques were applied to improve performance:

- Ridge Regression
- Lasso Regression
- Elastic Net

These methods help reduce overfitting and improve generalization.

### XGBoost

XGBoost is a tree-based ensemble model that builds multiple decision trees sequentially. It is known for strong performance on structured/tabular data.

It is capable of capturing:
- Non-linear relationships
- Feature interactions
- Complex decision boundaries

### FT-Transformer

The FT-Transformer is a deep learning model specifically designed for tabular data.

Key ideas:
- Each feature is converted into an embedding
- Self-attention is used to model feature interactions
- A classification token aggregates information for prediction

This model automatically learns relationships between features without manual feature engineering.

## Evaluation Metrics

To compare model performance, the following metrics were used:

- **ROC-AUC**: Measures how well the model separates classes  
- **PR-AUC**: Useful for imbalanced datasets  
- **Brier Score**: Measures probability calibration  
- **F1 Score**: Balances precision and recall  

A classification threshold of 0.3 was used to better capture churn cases.

## Results

| Model                | ROC-AUC | PR-AUC | Brier Score | F1 Score |
|---------------------|--------|--------|------------|----------|
| Logistic Regression | 0.8557 | 0.6743 | 0.1314     | 0.6399   |
| XGBoost             | 0.8388 | 0.6676 | 0.1575     | 0.5857   |
| FT-Transformer      | 0.8485 | 0.6601 | 0.1364     | 0.6187   |

Logistic Regression achieved the best overall performance across all evaluation metrics. It produced the highest ROC-AUC and F1 score, while also maintaining the lowest Brier score, indicating well-calibrated predictions.

XGBoost performed slightly worse than expected, particularly in terms of calibration and F1 score. While it captures non-linear relationships effectively, it likely requires further tuning.

The FT-Transformer model performed competitively and showed strong ROC-AUC values. However, its F1 score was slightly lower than Logistic Regression, suggesting that increased model complexity does not always guarantee better classification performance.

## Why Logistic Regression Performed Best

- **Fits tabular data well**  
  The dataset consists of structured features where linear relationships are sufficient.

- **Effective regularization**  
  Ridge, Lasso, and Elastic Net helped reduce overfitting.

- **Better calibration**  
  Logistic Regression achieved the lowest Brier score.

- **Lower complexity**  
  Simpler models can outperform complex ones when the data does not require deep feature interactions.

## Discussion

Each model has its own strengths:

- Logistic Regression:
  - Simple and interpretable  
  - Strong baseline performance  

- XGBoost:
  - Captures non-linear patterns  
  - Requires tuning for best results  

- FT-Transformer:
  - Learns feature interactions automatically  
  - More complex and computationally expensive  

The results show that while advanced models are powerful, they do not always outperform simpler models on structured datasets.

## Conclusion

This project compared traditional machine learning models with a modern deep learning approach for customer churn prediction.

The results show that Logistic Regression remains a strong and reliable baseline, even when compared to more complex models like XGBoost and FT-Transformer.

Future improvements could include:
- Hyperparameter tuning  
- Feature engineering  
- Model ensembling  
- Cross-validation  

## Project Structure

data/
src/
python/
ft_transformer.py
train_xgboost_telco_ordinal.py
R/
train_logistic_telco.R
README.md
requirements.txt
final_report.md

## Contributors

- Arun Sharma  
- Umar Mohammed Yousuf  
- Alex Vukovic  

## Repository

https://github.com/sharmaArun12/stat-516-customer-churn-analysis
