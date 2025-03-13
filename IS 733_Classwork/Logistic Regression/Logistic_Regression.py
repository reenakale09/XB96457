import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# Load the dataset
student_data = pd.read_csv("student_data.csv")

# Features and Target
X = student_data[['Hours_Studied', 'Review_Session']]
y = student_data['Results']

# Initialize and train Logistic Regression model
model = LogisticRegression()
model.fit(X, y)

# Model Coefficients
intercept = model.intercept_[0]
coefficients = model.coef_[0]

# Predict probabilities
y_pred_prob = model.predict_proba(X)[:, 1]
y_pred = model.predict(X)

# Model Performance
accuracy = accuracy_score(y, y_pred)
auc_score = roc_auc_score(y, y_pred_prob)

# Save results to a text file
with open(r"Logistic_Regression_Results.txt", "w") as f:
    f.write(f"Intercept: {intercept}\n")
    f.write(f"Coefficients: {coefficients}\n")
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"AUC Score: {auc_score}\n")

# Plotting ROC Curve
fpr, tpr, _ = roc_curve(y, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.title('ROC Curve - Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

# Save the ROC curve plot
plt.savefig("Logistic_Regression_ROC_Curve.png")
plt.show()

