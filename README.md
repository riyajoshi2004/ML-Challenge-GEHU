<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ML Classification Project</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      line-height: 1.6;
      max-width: 900px;
      margin: 20px auto;
      padding: 0 20px;
      background-color: #f9f9f9;
      color: #333;
    }
    h1, h2, h3, h4 {
      color: #2c3e50;
    }
    h1 {
      text-align: center;
      margin-bottom: 10px;
    }
    h2 {
      margin-top: 30px;
      border-bottom: 2px solid #2c3e50;
      padding-bottom: 5px;
    }
    img {
      display: block;
      margin: 20px auto;
      max-width: 100%;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    pre {
      background: #2d2d2d;
      color: #f8f8f2;
      padding: 15px;
      border-radius: 5px;
      overflow-x: auto;
    }
    table {
      border-collapse: collapse;
      width: 100%;
      margin: 15px 0;
    }
    table, th, td {
      border: 1px solid #ccc;
    }
    th, td {
      padding: 10px;
      text-align: left;
    }
    th {
      background-color: #ecf0f1;
    }
    .center {
      text-align: center;
    }
  </style>
</head>
<body>

  <h1>ML Classification Project</h1>

  <h2>Problem Statement</h2>
  <p>
    The objective of this project is to build a machine learning model capable of accurately classifying data into two categories based on the provided features.
  </p>
  <p>
    The dataset contains multiple numerical features along with a target variable <strong>Class</strong>. The goal is to learn the underlying patterns in the training data and generate predictions for the unseen test dataset.
  </p>

  <h2>Project Pipeline</h2>
  <img src="pipeline.png" alt="ML Pipeline" width="600"/>

  <h2>Datasets</h2>

  <h3>Training Dataset (TRAIN.csv)</h3>
  <ul>
    <li>Used for learning the relationship between features and the target variable.</li>
    <li>Contains:
      <ul>
        <li>Feature columns</li>
        <li>Target column <strong>Class</strong></li>
      </ul>
    </li>
  </ul>

  <h3>Test Dataset (TEST.csv)</h3>
  <ul>
    <li>Contains feature columns and <strong>ID</strong> column.</li>
    <li>No target variable; predictions are generated for this dataset.</li>
  </ul>

  <h2>Data Preparation</h2>
  <p>The dataset was loaded using <strong>Pandas</strong>:</p>
  <pre><code>import pandas as pd
train = pd.read_csv("TRAIN.csv")
test = pd.read_csv("TEST.csv")
X = train.drop(columns=["Class"])
y = train["Class"]</code></pre>

  <h2>Baseline Model: Logistic Regression</h2>
  <p>Why Logistic Regression?</p>
  <ul>
    <li>Simple and interpretable</li>
    <li>Provides insight into feature importance</li>
    <li>Good baseline for classification problems</li>
  </ul>
  <pre><code>from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)</code></pre>

  <h2>Model Evaluation</h2>
  <p>Metrics used:</p>
  <ul>
    <li>Accuracy</li>
    <li>Precision</li>
    <li>Recall</li>
    <li>Confusion Matrix</li>
  </ul>
  <pre><code>from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
accuracy_score(y_val, y_pred)</code></pre>

  <h2>Handling Class Imbalance</h2>
  <pre><code>model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)</code></pre>

  <h2>Feature Importance Analysis</h2>
  <pre><code>coef = pd.Series(model.coef_[0], index=X.columns)
coef.sort_values(ascending=False).head(10)</code></pre>

  <h2>Random Forest Model</h2>
  <p>Random Forest is an ensemble learning method combining multiple decision trees.</p>
  <pre><code>from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)</code></pre>

  <h3>Advantages</h3>
  <ul>
    <li>Handles nonlinear relationships</li>
    <li>Robust to noise</li>
    <li>Performs well on complex datasets</li>
  </ul>

  <h2>Model Validation</h2>
  <pre><code>accuracy_score(y_val, y_pred_rf)
confusion_matrix(y_val, y_pred_rf)</code></pre>

  <h2>Cross Validation</h2>
  <pre><code>from sklearn.model_selection import cross_val_score
scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy")</code></pre>

  <h2>Feature Importance from Random Forest</h2>
  <pre><code>importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(10)</code></pre>

  <h2>Final Model Training</h2>
  <pre><code>rf_final.fit(X, y)</code></pre>

  <h2>Test Prediction & Submission</h2>
  <pre><code>test_ids = test["ID"]
X_test = test.drop(columns=["ID"])
predictions = rf_final.predict(X_test)

submission = pd.DataFrame({
    "ID": test_ids,
    "CLASS": predictions
})
submission.to_csv("FINAL.csv", index=False)</code></pre>

  <h2>Technologies Used</h2>
  <ul>
    <li>Python</li>
    <li>Pandas</li>
    <li>Scikit-learn</li>
    <li>NumPy</li>
    <li>Jupyter Notebook</li>
  </ul>

  <h2>Models Used</h2>
  <table>
    <tr>
      <th>Model</th>
      <th>Purpose</th>
    </tr>
    <tr>
      <td>Logistic Regression</td>
      <td>Baseline model</td>
    </tr>
    <tr>
      <td>Random Forest</td>
      <td>Final model</td>
    </tr>
  </table>

  <h2>Results</h2>
  <p>The Random Forest model showed strong predictive performance. Metrics included:</p>
  <ul>
    <li>Accuracy</li>
    <li>Precision</li>
    <li>Recall</li>
    <li>F1 Score</li>
  </ul>

  <h2>Project Structure</h2>
  <pre><code>project/
│
├── TRAIN.csv
├── TEST.csv
│
├── notebook.ipynb
├── FINAL.csv
└── README.md</code></pre>

  <h2>Future Improvements</h2>
  <ul>
    <li>Hyperparameter tuning</li>
    <li>Gradient Boosting models</li>
    <li>XGBoost / LightGBM</li>
    <li>Feature engineering</li>
    <li>SHAP model explainability</li>
  </ul>

</body>
</html>