import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# Load the dataset
student_data = pd.read_csv(r"student_data.csv")

# Features and Target
X = student_data[['Hours_Studied', 'Review_Session']]
y = student_data['Results']

# --- Linear Kernel SVM ---
svm_linear = SVC(kernel='linear', probability=True)
svm_linear.fit(X, y)
y_pred_linear = svm_linear.predict(X)
y_pred_proba_linear = svm_linear.decision_function(X)

# Model Performance - Linear Kernel
accuracy_linear = accuracy_score(y, y_pred_linear)
auc_linear = roc_auc_score(y, y_pred_proba_linear)

# Save results to a text file
with open(r"SVM_Results.txt", "w") as f:
    f.write(f"Linear Kernel - Accuracy: {accuracy_linear}\n")
    f.write(f"Linear Kernel - AUC Score: {auc_linear}\n")

# --- RBF Kernel SVM with Grid Search ---
svm_rbf = SVC(kernel='rbf', probability=True)
param_grid = {'gamma': np.logspace(-3, 3, 7)}
grid_search = GridSearchCV(svm_rbf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

# Best model from grid search
best_model = grid_search.best_estimator_
best_gamma = grid_search.best_params_['gamma']
y_pred_rbf = best_model.predict(X)
y_pred_proba_rbf = best_model.decision_function(X)
accuracy_rbf = accuracy_score(y, y_pred_rbf)
auc_rbf = roc_auc_score(y, y_pred_proba_rbf)

# Save additional results for RBF Kernel
with open(r"SVM_Results.txt", "a") as f:
    f.write(f"\nRBF Kernel - Best Gamma: {best_gamma}\n")
    f.write(f"RBF Kernel - Accuracy: {accuracy_rbf}\n")
    f.write(f"RBF Kernel - AUC Score: {auc_rbf}\n")

# Plotting ROC Curves
fpr_linear, tpr_linear, _ = roc_curve(y, y_pred_proba_linear)
fpr_rbf, tpr_rbf, _ = roc_curve(y, y_pred_proba_rbf)

plt.figure(figsize=(10, 6))
plt.plot(fpr_linear, tpr_linear, label=f'Linear Kernel (AUC = {auc_linear:.2f})')
plt.plot(fpr_rbf, tpr_rbf, label=f'RBF Kernel (AUC = {auc_rbf:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.title('ROC Curve Comparison - Linear vs RBF Kernel SVM')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

# Save the ROC Curve Comparison
plt.savefig(r"SVM_ROC_Curve_Comparison.png")
plt.show()