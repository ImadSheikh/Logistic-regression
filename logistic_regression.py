import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, f1_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
from AI_model import customLogisticRegression
from AI_model import customOVA



# Load Data
X = np.load("cifar10_features.npy")
y = np.load("cifar10_labels.npy")

print("Features shape:", X.shape)
print("Labels shape:", y.shape)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Training set size:", X_train.shape[0])
print("Testing set size:", X_test.shape[0])

# Train Custom OvA Logistic Regression
start_ova = time.time()
ova_model = customOVA(learning_rate=0.01, max_iter=1000)
ova_model.fit(X_train, y_train)
end_ova = time.time()

# Predictions and Evaluation
y_pred_ova = ova_model.predict(X_test)
acc_ova = accuracy_score(y_test, y_pred_ova)
time_ova = end_ova - start_ova

print(f"[Custom OvA] Accuracy: {acc_ova:.4f}, Training time: {time_ova:.2f} seconds")

# Log Loss and F1 Score
probs_ova = ova_model.predict_proba(X_test)
loss_ova = log_loss(y_test, probs_ova)
f1_ova = f1_score(y_test, y_pred_ova, average='macro')
print(f"[Custom OvA] Log Loss: {loss_ova:.4f}")
print(f"[Custom OvA] F1 Macro: {f1_ova:.4f}")

# Confusion Matrix
conf_mat_ova = confusion_matrix(y_test, y_pred_ova)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat_ova, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Custom OvA)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_custom_ova.png")
print("Confusion matrix saved as 'confusion_matrix_custom_ova.png'")

# Binary Classifier for Confused Classes (3 vs 5)
mask_train = (y_train == 3) | (y_train == 5)
X_train_3vs5 = X_train[mask_train]
y_train_3vs5 = (y_train[mask_train] == 5).astype(int) # Encode 3 as 0, 5 as 1

mask_test = (y_test == 3) | (y_test == 5)
X_test_3vs5 = X_test[mask_test]
y_test_3vs5 = (y_test[mask_test] == 5).astype(int)   # Encode 3 as 0, 5 as 1

binary_model = customLogisticRegression(learning_rate=0.01, max_iter=1000)
binary_model.fit(X_train_3vs5, y_train_3vs5)

y_pred_3vs5 = binary_model.predict(X_test_3vs5)
print("Binary Classifier (3 vs 5) Evaluation:")
print(classification_report(y_test_3vs5, y_pred_3vs5, zero_division=1))
