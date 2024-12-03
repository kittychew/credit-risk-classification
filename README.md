# Credit-Risk-Classification

## Overview of the Analysis

This analysis focuses on building a machine learning model to assess the creditworthiness of borrowers. Specifically, we developed a binary supervised learning model to predict loan status (`0` for healthy loans and `1` for high-risk loans). The dataset includes financial information on borrowers, such as loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, and total debt.

The process involved several stages:  
1. **Separating Features and Labels:**  
   - Features (`X`): Variables that the model uses to make predictions (e.g., loan size, interest rate, etc.).  
   - Labels (`y`): The target variable we are trying to predict (loan status).  

2. **Splitting Data into Training and Testing Sets:**  
   - We divided the dataset into training data (used to train the model) and testing data (used to evaluate the model) using the `train_test_split` module. This process created four subsets: `X_train`, `X_test`, `y_train`, and `y_test`.

3. **Training the Logistic Regression Model:**  
   - We used the `LogisticRegression` module to create and train the model. The training process involved fitting the `X_train` and `y_train` data to allow the model to learn patterns from the features.

4. **Making Predictions and Evaluating the Model:**  
   - After training, we used the model to predict the loan status of the test data (`X_test`).  
   - We evaluated its performance using a confusion matrix and a classification report. The confusion matrix provided a breakdown of correct and incorrect predictions, while the classification report summarized precision, recall, and accuracy metrics.

### Key Metrics:  
- **Precision:** Measures the accuracy of the model's positive predictions (e.g., how many loans predicted as high-risk were truly high-risk).  
- **Recall:** Measures how well the model identifies actual positives (e.g., how many of the actual high-risk loans were correctly identified).  
- **Accuracy:** Measures the overall correctness of the model across all predictions.  

---

## Results

### Logistic Regression Model Metrics:
- **Accuracy:** The model achieved a high overall accuracy, demonstrating its reliability in predicting loan status.  
- **Precision and Recall:**  
  - **Healthy Loans (`0`):**  
    - Precision: **100%** – Every loan predicted as healthy was actually healthy.  
    - Recall: **99%** – The model correctly identified 99% of all actual healthy loans.  
  - **High-Risk Loans (`1`):**  
    - Precision: **84%** – 84% of loans predicted as high-risk were actually high-risk.  
    - Recall: **94%** – The model identified 94% of all actual high-risk loans.  

These results highlight that the model is exceptional at predicting healthy loans and performs well at identifying high-risk loans, albeit with some misclassifications.

---

## Summary

### Performance Analysis:
The logistic regression model performs exceptionally well for predicting healthy loans (`0`), with nearly perfect precision and recall. For high-risk loans (`1`), while it successfully identifies most high-risk loans (94% recall), it has a lower precision of 84%, meaning 16% of healthy loans are misclassified as high-risk.

### Recommendation:
The choice to use this model depends on the business objective:  
- If minimizing false negatives (missing high-risk loans) is the priority, this model is recommended due to its high recall for high-risk loans (94%). This ensures that most high-risk loans are caught, even at the cost of some false positives.  
- If minimizing false positives (misclassifying healthy loans as high-risk) is more critical, further model tuning may be needed to improve the precision for high-risk loans.

Given the goal of identifying high-risk loans, this model is suitable because its high recall ensures that most high-risk loans are flagged.

