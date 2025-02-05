# Bank Customer Churn Prediction

This project predicts bank customer churn using an end-to-end data science pipeline. The focus is on preparing the data, exploring it, handling class imbalance, and training multiple machine learning models to accurately identify customers likely to churn.

## Project Structure


## Dataset

- **Source:** Bank Customer Churn dataset from Kaggle.
- **Description:** The dataset contains the following columns:
  - `customer_id`: Unique identifier for each customer.
  - `credit_score`: Customer credit score.
  - `country`: Customer's country.
  - `gender`: Customer's gender.
  - `age`: Age of the customer.
  - `tenure`: Years the customer has been with the bank.
  - `balance`: Account balance.
  - `products_number`: Number of bank products used.
  - `credit_card`: Indicates if the customer has a credit card.
  - `active_member`: Indicates if the customer is active.
  - `estimated_salary`: Estimated annual salary.
  - `churn`: Target variable (0 = stayed, 1 = churned).

- **Data Preparation:**
  - Removed columns such as `credit_card` and `tenure`.
  - Filtered out customers from Spain.
  - Handled missing values and performed one-hot encoding for categorical variables.
  - Standardized numerical features.

## Data Preprocessing

The preprocessing steps are implemented in a Jupyter Notebook and include:

1. **Loading the Dataset:**  
   Reading the CSV file and creating the features (`X`) and target (`y`).

2. **Data Splitting:**  
   Splitting the data into training and testing sets with stratification to preserve the imbalance in classes.

3. **Balancing the Data:**  
   Applying SMOTE (Synthetic Minority Over-sampling Technique) to the training set to generate synthetic examples for the minority class (churn).

4. **Standardization:**  
   Using `StandardScaler` to normalize the feature values.

5. **Saving the Preprocessing Artifacts:**  
   The scaler is saved for use during model deployment.

## Modeling

### Logistic Regression Baseline
- **Objective:** Establish a baseline for performance.
- **Outcome:** High overall accuracy but very low recall for churned customers.

### Random Forest
- **Objective:** Improve detection of churned customers.
- **Outcome:**  
  - Accuracy: ~81.5%
  - Churn recall improved from ~23% (Logistic Regression) to ~61% with an adjusted decision threshold.
- **Tuning:** Initial hyperparameter tuning with GridSearchCV was attempted but did not yield significant improvements over threshold adjustments.

### XGBoost
- **Objective:** Achieve better recall with a more sophisticated model.
- **Outcome:**  
  - With a decision threshold of 0.4, achieved:
    - Accuracy: ~76.5%
    - Churn Precision: ~47%
    - Churn Recall: ~74%
- **Hyperparameter Tuning:**  
  - RandomizedSearchCV was used to tune parameters such as `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, and `scale_pos_weight`.
  - Version compatibility issues with XGBoost and scikit-learn were encountered and addressed through various workarounds.

## Environment Setup

- **Python Version:**  
  Initially using Python 3.13 (beta), which led to compatibility issues. Instructions were provided to switch to a stable release (e.g., Python 3.10 or 3.11).

- **Dependencies:**  
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `xgboost`
  - `imbalanced-learn`
  - `joblib`

Install the required packages via:
```bash
pip install -r requirements.txt
