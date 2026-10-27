import numpy as np


def add_bias(X):
    """Thêm cột 1 vào cuối ma trận đặc trưng để gộp hệ số bias (Bias trick chuẩn Lab 03)."""
    return np.hstack([X, np.ones((X.shape[0], 1))])

def sigmoid(z):
    """σ(z) tính ổn định số học cho cả z rất lớn và rất âm theo chuẩn Lab 03."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

def softmax(z):
    """Softmax đa lớp trừ max tránh tràn số (Numerical Stability)."""
    z_max = np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class LogisticRegressionNP:
    """Logistic Regression đa lớp (Softmax) thuần NumPy giải bằng Mini-batch GD có điều chuẩn L2 (không phạt bias)."""
    def __init__(self, lr=0.1, epochs=15, batch_size=256, l2=1e-4, seed=42):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        N, D = X.shape
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        Xb = add_bias(X)
        self.W_ = np.zeros((D + 1, K))
        
        # One-hot encoding y
        Y_onehot = np.zeros((N, K))
        Y_onehot[np.arange(N), y] = 1.0

        mask = np.ones((D + 1, 1))
        mask[-1] = 0.0 # Khong phat bias theo quy chuẩn Lab 03

        self.loss_history_ = []
        indices = np.arange(N)

        for epoch in range(self.epochs):
            rng.shuffle(indices)
            for s in range(0, N, self.batch_size):
                b_idx = indices[s:s + self.batch_size]
                xb = Xb[b_idx]
                yb = Y_onehot[b_idx]
                kb = len(b_idx)

                # Lan truyền tiến: tính logits và xác suất dự đoán
                logits = xb @ self.W_
                probs = softmax(logits)

                # Gradient của Cross-Entropy kèm điều chuẩn L2 (không phạt bias)
                grad = (xb.T @ (probs - yb)) / kb + self.l2 * (mask * self.W_)
                self.W_ -= self.lr * grad

            # Tính loss toàn cục cuối mỗi epoch
            full_logits = Xb @ self.W_
            full_probs = softmax(full_logits)
            loss = -np.mean(np.log(np.sum(full_probs * Y_onehot, axis=1) + 1e-12)) + 0.5 * self.l2 * np.sum(self.W_[:-1]**2)
            self.loss_history_.append(loss)

        return self

    def decision_function(self, X):
        return add_bias(X) @ self.W_

    def predict_proba(self, X):
        return softmax(self.decision_function(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]
