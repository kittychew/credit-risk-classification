# Credit-Risk-Classification

## Overview

This analysis focuses on building a machine learning model to assess the creditworthiness of borrowers. Specifically, we developed a **binary logistic-regression model**  to predict loan status (`0` for healthy loans and `1` for high-risk loans) for 10,000 loan applications. 

### Data Description

| File                      | Rows  | Columns | Notes                                         |
|:-------------------------:|:-----:|:-------:|:---------------------------------------------:|
| `Resources/lending_data.csv` | 10,000 | 8       | 7 features + `loan_status` label              |

### Feature List

   | Column                   | Type    | Description                                                 |
   |:-------------------------|:-------:|:------------------------------------------------------------|
   | `loan_size`              | float   | Principal amount                                           |
   | `interest_rate`          | float   | Annual interest rate (%)                                   |
   | `borrower_income`        | float   | Annual income in USD                                       |
   | `debt_to_income_ratio`   | float   | Total debt ÷ income                                        |
   | `open_accounts`          | int     | Number of currently open credit lines                      |
   | `derogatory_marks`       | int     | Count of negative credit events                            |
   | `total_debt`             | float   | Sum of all outstanding debt                                |
   | `loan_status`            | 0 / 1   | **Label**: 0=healthy, 1=high-risk                           |


## Methodology:  
1. **Separating Features and Labels:**  
   - Features (X): Variables that the model uses to make predictions (e.g., loan size, interest rate, etc.).  
   - Labels (y): The target variable we are trying to predict (`loan_status`).

   ```python
   # Separate the data into labels and features

   # Separate the X variable, the features
   X = df.drop(columns=["loan_status"])

    # Separate the y variable, the labels
   y = df["loan_status"]
   ```

2. **Splitting Data into Training and Testing Sets:**  
   We divided the dataset into training data (used to train the model) and testing data (used to evaluate the model) using the `train_test_split` module. This process created four subsets: `X_train`, `X_test`, `y_train`, and `y_test`.
   ```python
      # Import the train_test_learn module
      from sklearn.model_selection import train_test_split

      # Split the data using train_test_split
      # Assign a random_state of 1 to the function
      X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.25, random_state=1 #25% will be testing, 75% for training
      )
      ```

3. **Training the Logistic Regression Model:**  
   We used the `LogisticRegression` module to create and train the model. The training process involved fitting the `X_train` and `y_train` data to allow the model to learn patterns from the features.
   
   ```python
   # Import the LogisticRegression module from SKLearn
   from sklearn.linear_model import LogisticRegression

   # Instantiate the Logistic Regression model
   # Assign a random_state parameter of 1 to the model
   model = LogisticRegression(max_iter=200, random_state=1)

   # Fit the model using training data
   model.fit(X_train, y_train) #Put the training data into the model
   ```

4. **Making Predictions and Evaluating the Model:**  
   After training, we used the model to predict the loan status of the testing feature data (`X_test`).  

   We evaluated its performance using a confusion matrix and a classification report. The confusion matrix provided a breakdown of correct and incorrect predictions, while the classification report summarized precision, recall, and accuracy metrics.

   ```python
   # Make a prediction using the testing data
   y_pred = model.predict(X_test) #By now the model should have been trained, so we are going to make predictions 

   cm = confusion_matrix(y_test, y_pred)
   print("Confusion Matrix:")
   print(cm)

   cr = classification_report(y_test, y_pred)
   print("Classification Report:")
   print(cr)

   ```


      ### Confusion Matrix

      ![Confusion Matrix](20_Supervised_Learning/credit-risk-classification/Images/confusion_matrix.png)
         Where:
         - **4948** True Negatives (healthy loans correctly identified)
         - **42** False Positives (risky loans predicted as healthy)
         - **226** False Negatives (healthy loans predicted as risky)
         - **284** True Positives (risky loans correctly identified)
         The model was conservative in predicting risky loans, leading to a higher number of false negatives (226), which can be risky in real-world lending decisions.

      ### Classification Report

      ![Classification Report](20_Supervised_Learning/credit-risk-classification/Images/classification_report.png)

      These metrics help explain:

      - The model had strong **precision** for both classes, especially class 1.0 (85%), meaning when it predicted a risky loan, it was usually correct.
      - However, the **recall** for risky loans was only **56%**, meaning it missed nearly half of the actual risky loans.


## Summary

- **Strong overall accuracy** (94%) and high performance on healthy loans.
- **Lower recall on risky loans** (56%) suggests under-identification of default risk.
- This could result in financial risk if deployed in production without further tuning.

### Model Recommendation:
The choice to use this model depends on the business objective:  
- If minimizing false negatives (missing high-risk loans) is the priority, this model is recommended due to its high recall for high-risk loans (94%). This ensures that most high-risk loans are caught, even at the cost of some false positives.  
- If minimizing false positives (misclassifying healthy loans as high-risk) is more critical, further model tuning may be needed to improve the precision for high-risk loans.
- Given the goal of identifying high-risk loans, this model is suitable because its high recall ensures that most high-risk loans are flagged.

### Future Uses

To improve the model’s sensitivity to risky loans and reduce false negatives:
- **Balance the dataset** using techniques like SMOTE or undersampling.
- **Try ensemble methods** like Random Forest or XGBoost for better class boundary separation.
- **Tune the decision threshold** to prioritize recall if business goals require higher risk detection.
- **Refine features** to better differentiate borrower profiles.