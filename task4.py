import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('cancer.csv')
data = data.drop(['id', 'Unnamed: 32'], axis=1)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

X = data.drop('diagnosis', axis=1).values
y = data['diagnosis'].values.reshape(-1, 1)

from numpy import mean, std
X = (X - mean(X, axis=0)) / std(X, axis=0)

np.random.seed(42)
indices = np.random.permutation(len(X))
train_size = int(0.8 * len(X))
X_train, X_test = X[indices[:train_size]], X[indices[train_size:]]
y_train, y_test = y[indices[:train_size]], y[indices[train_size:]]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train(X, y, lr=0.01, epochs=10000):
    n_samples, n_features = X.shape
    weights = np.zeros((n_features, 1))
    bias = 0
    for _ in range(epochs):
        linear = np.dot(X, weights) + bias
        predictions = sigmoid(linear)
        dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
        db = (1 / n_samples) * np.sum(predictions - y)
        weights -= lr * dw
        bias -= lr * db
    return weights, bias

weights, bias = train(X_train, y_train)

def predict(X, weights, bias, threshold=0.5):
    probs = sigmoid(np.dot(X, weights) + bias)
    return (probs >= threshold).astype(int), probs

y_pred, y_prob = predict(X_test, weights, bias)

def confusion(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[TN, FP], [FN, TP]])

def precision_score(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    return TP / (TP + FP)

def recall_score(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP / (TP + FN)

def roc_auc_score(y_true, y_prob):
    thresholds = np.linspace(0, 1, 100)
    tprs = []
    fprs = []
    for thresh in thresholds:
        y_pred_thresh = (y_prob >= thresh).astype(int)
        TP = np.sum((y_true == 1) & (y_pred_thresh == 1))
        TN = np.sum((y_true == 0) & (y_pred_thresh == 0))
        FP = np.sum((y_true == 0) & (y_pred_thresh == 1))
        FN = np.sum((y_true == 1) & (y_pred_thresh == 0))
        TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
        tprs.append(TPR)
        fprs.append(FPR)
    auc = np.trapz(tprs, fprs)
    plt.plot(fprs, tprs, label=f'ROC Curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()
    return auc

cm = confusion(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("Confusion Matrix:")
print(cm)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"ROC-AUC: {auc:.2f}")
