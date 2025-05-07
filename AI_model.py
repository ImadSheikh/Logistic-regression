import numpy as np

# Constants
EPSILON = 1e-10  # Small constant to prevent division by zero and log(0)

class customLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        """The sigmoid function."""
        return 1 / (1 + np.exp(-z))

    def cost_function(self, y_true, y_pred):
        """Binary cross-entropy cost function."""
        m = len(y_true)
        # Add epsilon to y_pred to prevent log(0)
        y_pred = np.clip(y_pred, EPSILON, 1 - EPSILON)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X, y):
        """Fits the logistic regression model to the training data."""
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.max_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            dw = (1 / m) * np.dot(X.T, (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        """Predict probabilities for input X."""
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """Predict class labels (0 or 1)."""
        y_pred_proba = self.predict_proba(X)
        return (y_pred_proba >= threshold).astype(int)


class customOVA:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.models = []
        self.classes_ = None  # Store the class labels

    def fit(self, X, y):
        """Fits N binary classifiers, one for each class."""
        self.classes_ = np.unique(y)
        self.models = []
        for c in self.classes_:
            binary_y = (y == c).astype(int)  # Correct binary label creation
            model = customLogisticRegression(learning_rate=self.learning_rate, max_iter=self.max_iter)
            model.fit(X, binary_y)
            self.models.append(model)

    def predict_proba(self, X):
        """Predict the probability of each sample belonging to each class."""
        probs = np.array([model.predict_proba(X) for model in self.models]).T
        # Normalize probabilities row-wise, with epsilon
        probs = probs / (probs.sum(axis=1, keepdims=True) + EPSILON)
        return probs

    def predict(self, X):
        """Predict the class label for each sample based on the highest probability."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)