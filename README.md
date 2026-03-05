ML Classification Project
Problem Statement

The objective of this project is to build a machine learning model capable of accurately classifying data into two categories based on the provided features.

The dataset contains multiple numerical features along with a target variable Class. The goal is to learn the underlying patterns in the training data and generate predictions for the unseen test dataset.

Project Pipeline

<p align="center">
  <img src="IEEE_ML_CHALLENGE/images/pipeline.png" alt="ML Pipeline" width="600"/>
</p>

Two datasets were provided:

Training Dataset

Used for learning the relationship between features and the target variable.

TRAIN.csv

Contains:

Feature columns

Target column Class

Test Dataset
TEST.csv

Contains:

Feature columns

ID column

No target variable

Predictions are generated for this dataset.

Data Preparation

The dataset was loaded using Pandas.

Example:

train = pd.read_csv("TRAIN.csv")
test = pd.read_csv("TEST.csv")

Feature matrix and target variable were separated:

X → feature variables
y → target variable (Class)
Baseline Model: Logistic Regression

A Logistic Regression model was trained as the baseline model to understand the dataset.

Why Logistic Regression?

Simple and interpretable

Provides insight into feature importance

Good baseline for classification problems

Model training:

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
Model Evaluation

The model was evaluated using multiple classification metrics.

Metrics used:

Accuracy

Precision

Recall

Confusion Matrix

Example:

accuracy_score
precision_score
recall_score
confusion_matrix

These metrics help evaluate how well the model performs on unseen data.

Handling Class Imbalance

To improve the baseline model, class balancing was introduced.

class_weight = "balanced"

Example:

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

This helps the model give equal importance to both classes.

Feature Importance Analysis

Logistic Regression coefficients were used to understand feature impact.

Example:

coef = pd.Series(model.coef_[0], index=X.columns)
coef.sort_values(ascending=False).head(10)

This step helps identify the most influential features in the dataset.

Random Forest Model

To improve performance, a Random Forest Classifier was trained.

Random Forest is an ensemble learning method that combines multiple decision trees to produce more robust predictions.

Model configuration:

RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

Advantages:

Handles nonlinear relationships

Robust to noise

Performs well on complex datasets

Model Validation

The Random Forest model was evaluated using:

Accuracy

Precision

Recall

Confusion Matrix

Example:

accuracy_score(y_val, y_pred_rf)
confusion_matrix(y_val, y_pred_rf)
Cross Validation

To ensure the model generalizes well, 5-fold cross validation was applied.

scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy")

Cross-validation helps:

Reduce overfitting

Provide reliable performance estimates

Evaluate model stability

Feature Importance from Random Forest

Random Forest provides built-in feature importance scores.

Example:

importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(10)

This helps identify the features that contribute most to predictions.

Final Model Training

Once the model was validated, the final Random Forest model was trained on the entire training dataset.

rf_final.fit(X, y)

Training on all available data improves model learning before generating final predictions.

Test Prediction

The test dataset was prepared by separating the ID column.

Example:

test_ids = test["ID"]
X_test = test.drop(columns=["ID"])

Predictions were generated using the trained Random Forest model.

predictions = rf_final.predict(X_test)
Submission File Generation

The final predictions were stored in a submission file.

Example:

submission = pd.DataFrame({
    "ID": test_ids,
    "CLASS": predictions
})

submission.to_csv("FINAL.csv", index=False)

Final output:

FINAL.csv

This file can be uploaded to the competition platform for evaluation.

Technologies Used

Python

Pandas

Scikit-learn

NumPy

Jupyter Notebook

Models Used
Model	Purpose
Logistic Regression	Baseline model
Random Forest	Final model
Results

The Random Forest model showed strong predictive performance and was used for generating the final submission.

Evaluation metrics included:

Accuracy

Precision

Recall

F1 Score

Project Structure
project/
│
├── TRAIN.csv
├── TEST.csv
│
├── notebook.ipynb
│
├── FINAL.csv
│
└── README.md
Future Improvements

Possible improvements for better performance:

Hyperparameter tuning

Gradient Boosting models

XGBoost / LightGBM

Feature engineering

SHAP model explainability