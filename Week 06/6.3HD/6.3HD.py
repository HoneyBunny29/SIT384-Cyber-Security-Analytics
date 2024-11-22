import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("spambase.data", header=None)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Ensure X is a NumPy array and contiguous in memory
X = np.ascontiguousarray(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Standardize data for Logistic Regression and SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to plot confusion matrix
def plot_conf_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

# 1. K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("KNN - Training Accuracy: {:.3f}".format(knn.score(X_train, y_train)))
print("KNN - Test Accuracy: {:.3f}".format(accuracy_score(y_test, y_pred_knn)))

# Plot Confusion Matrix for KNN
plot_conf_matrix(y_test, y_pred_knn, "KNN")

# 2. Logistic Regression
log_reg = LogisticRegression(solver='lbfgs', max_iter=5000, C=1)
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict(X_test_scaled)
print("Logistic Regression - Training Accuracy: {:.3f}".format(log_reg.score(X_train_scaled, y_train)))
print("Logistic Regression - Test Accuracy: {:.3f}".format(accuracy_score(y_test, y_pred_log_reg)))

# Plot Confusion Matrix for Logistic Regression
plot_conf_matrix(y_test, y_pred_log_reg, "Logistic Regression")

# Plot coefficients for Logistic Regression
plt.figure(figsize=(10, 6))
plt.plot(log_reg.coef_.T, 'o', label="C=1")
plt.xticks(range(X_train.shape[1]), range(X_train.shape[1]), rotation=90)
plt.xlabel("Feature index")
plt.ylabel("Coefficient magnitude")
plt.title("Logistic Regression Coefficients")
plt.legend()
plt.show()

# 3. Random Forest
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)
y_pred_forest = forest.predict(X_test)
print("Random Forest - Training Accuracy: {:.3f}".format(forest.score(X_train, y_train)))
print("Random Forest - Test Accuracy: {:.3f}".format(accuracy_score(y_test, y_pred_forest)))

# Plot Confusion Matrix for Random Forest
plot_conf_matrix(y_test, y_pred_forest, "Random Forest")

# Plot Feature Importances for Random Forest
plt.figure(figsize=(10, 6))
plt.barh(range(X_train.shape[1]), forest.feature_importances_, align='center')
plt.yticks(range(X_train.shape[1]), range(X_train.shape[1]))
plt.xlabel("Feature Importance")
plt.ylabel("Feature Index")
plt.title("Random Forest Feature Importances")
plt.show()

# 4. Support Vector Machines (SVM)
svm_model = SVC(kernel='linear', C=1, random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
print("SVM - Training Accuracy: {:.3f}".format(svm_model.score(X_train_scaled, y_train)))
print("SVM - Test Accuracy: {:.3f}".format(accuracy_score(y_test, y_pred_svm)))

# Plot Confusion Matrix for SVM
plot_conf_matrix(y_test, y_pred_svm, "SVM")

# Plot Classification Reports
classification_reports = {
    "KNN": classification_report(y_test, y_pred_knn, output_dict=True),
    "Logistic Regression": classification_report(y_test, y_pred_log_reg, output_dict=True),
    "Random Forest": classification_report(y_test, y_pred_forest, output_dict=True),
    "SVM": classification_report(y_test, y_pred_svm, output_dict=True)
}

# Visualize classification reports using heatmaps
for model, report in classification_reports.items():
    plt.figure(figsize=(10, 6))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)
    plt.title(f"{model} Classification Report")
    plt.show()
