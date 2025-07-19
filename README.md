# Credit Risk Classification

## Overview

This project focuses on building a binary classification model to assess the creditworthiness of borrowers. Using logistic regression, we predicted loan status (`0` = healthy, `1` = high-risk) across **20,000** loan applications.

---

## Dataset Description

| File Path                      | Rows  | Columns | Notes                                      |
|:------------------------------|:-----:|:-------:|:-------------------------------------------|
| `Resources/lending_data.csv`  | 20,000| 8       | 7 feature columns + 1 target label (`loan_status`) |

### Features

| Column               | Type   | Description                              |
|----------------------|--------|------------------------------------------|
| `loan_size`          | float  | Principal loan amount                    |
| `interest_rate`      | float  | Annual interest rate (%)                |
| `borrower_income`    | float  | Borrower's yearly income (USD)          |
| `debt_to_income_ratio`| float | Total debt divided by income            |
| `open_accounts`      | int    | Number of open credit lines             |
| `derogatory_marks`   | int    | Negative credit events                  |
| `total_debt`         | float  | Total current outstanding debt          |
| `loan_status`        | 0 / 1  | Target label: 0 = healthy, 1 = high-risk|

---

## Methodology

### 1. Separating Features and Labels

```python
X = df.drop(columns=["loan_status"])
y = df["loan_status"]
```

### 2. Splitting into Training & Test Sets

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1
)
```

### 3. Training the Logistic Regression Model

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=200, random_state=1)
model.fit(X_train, y_train)
```

### 4. Making Predictions and Evaluating Model Performance

```python
from sklearn.metrics import confusion_matrix, classification_report

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

print("Classification Report:")
print(cr)
```

---

## Model Evaluation

### Confusion Matrix

```
[[18558   207]
 [   37   582]]
```

- **18,558** True Negatives (correctly predicted healthy loans)  
- **207** False Positives (predicted risky loan but was healthy)  
- **37** False Negatives (predicted healthy loan but was risky)  
- **582** True Positives (correctly predicted risky loans)  


   The model rarely missed risky loans (only 37 false negatives), which is crucial in financial decision-making.

---

### Classification Report

```
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.84      0.94      0.89       619

    accuracy                           0.99     19384
   macro avg       0.92      0.97      0.94     19384
weighted avg       0.99      0.99      0.99     19384
```

- **Precision (1.0)**: The model made very few false predictions overall.
- **Recall (94%) for risky loans**: High sensitivity to identifying default-prone applicants.
- **F1-score (0.89)**: Balanced performance between precision and recall for high-risk loans.
- **Accuracy (99%)**: Strong classification across nearly 20,000 samples.

---

## Key Takeaways

- The logistic regression model demonstrated **exceptional accuracy** (99%).
- It **correctly identified most risky loans** with high recall and very few false negatives (only 37).
- The model’s performance makes it a viable option for **early-stage loan screening** and risk assessment.

---

## Recommendations for Future Improvements

- **Threshold Tuning**: Adjust decision boundary to further optimize recall or precision based on business needs.
- **Model Comparison**: Try ensemble classifiers (Random Forest, XGBoost) for potential performance gains.
- **Cross-Validation**: Validate model stability across different subsets of the data.
- **Feature Importance Analysis**: Gain interpretability into what features influence predictions most.

---