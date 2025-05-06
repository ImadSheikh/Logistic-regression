import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import time


X = np.load("cifar10_features.npy")
y = np.load("cifar10_labels.npy")

print("Features shape: " , X.shape)
print("Labels shape: " , y.shape)

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.3 , random_state=42)

print("Training set size: " , X_train.shape[0])
print("Testing set size: " , X_test.shape[0])


start_ovr = time.time()
model_ovr = LogisticRegression(multi_class='ovr' , solver='lbfgs' , max_iter=1000)
model_ovr.fit(X_train , y_train)
end_ovr = time.time()

y_pred_ovr = model_ovr.predict(X_test)
acc_ovr = accuracy_score(y_test , y_pred_ovr)
time_ovr = end_ovr - start_ovr


print(f"[OvR] Accuracy: {acc_ovr:.4f}, Training time: {time_ovr:.2f} seconds")


start_softmax = time.time()
model_softmax = LogisticRegression(multi_class='multinomial' , solver='lbfgs' , max_iter=1000)
model_softmax.fit(X_train , y_train)
end_softmax = time.time()

y_pred_softmax = model_softmax.predict(X_test)
acc_softmax = accuracy_score(y_test , y_pred_softmax)
time_softmax = end_softmax - start_softmax


print(f"[Softmax] Accuracy: {acc_softmax:.4f}, Training time: {time_softmax:.2f} seconds")


probs_ovr = model_ovr.predict_proba(X_test)
loss_ovr = log_loss(y_test , probs_ovr)
print(f"[OvR] Log Loss: {loss_ovr:.4f}")

probs_softmax = model_softmax.predict_proba(X_test)
loss_softmax = log_loss(y_test, probs_softmax)
print(f"[Softmax] Log Loss: {loss_softmax:.4f}")


f1_ovr = f1_score(y_test, y_pred_ovr, average='macro')
print(f"[OvR] F1 Macro: {f1_ovr:.4f}")


f1_softmax = f1_score(y_test, y_pred_softmax, average='macro')
print(f"[Softmax] F1 Macro: {f1_softmax:.4f}")


conf_mat_ovr = confusion_matrix(y_test, y_pred_ovr)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat_ovr, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (OvR)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_ovr.png")  # Save instead of showing
print("Confusion matrix saved as 'confusion_matrix_ovr.png'")



mask_train = (y_train == 3) | (y_train == 5)
X_train_3vs5 = X_train[mask_train]
y_train_3vs5 = y_train[mask_train]

mask_test = (y_test == 3) | (y_test == 5)
X_test_3vs5 = X_test[mask_test]
y_test_3vs5 = y_test[mask_test]

binary_model = LogisticRegression(solver='lbfgs' , max_iter=1000)
binary_model.fit(X_train_3vs5 , y_train_3vs5)

y_pred_3vs5 = binary_model.predict(X_test_3vs5)
print("Binary classifier (3 vs 5) Evaluation:")
print(classification_report(y_test_3vs5 , y_pred_3vs5))





