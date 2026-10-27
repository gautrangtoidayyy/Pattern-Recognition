import numpy as np


class NaiveBayes:
    # Lớp Naive Bayes tổng quát thuần NumPy (From Scratch) - bám sát chuẩn Lab 02.
    def __init__(self, log_pdf, var_smoothing=1e-9):
        self.log_pdf = log_pdf
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.prior = np.zeros(n_classes)

        # Lượng làm mịn epsilon tránh phương sai bằng 0 (tương tự var_smoothing của scikit-learn)
        eps = self.var_smoothing * X.var(axis=0).max()

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx] = X_c.mean(axis=0)
            self.var[idx] = X_c.var(axis=0) + eps
            self.prior[idx] = X_c.shape[0] / n_samples

        return self

    def predict(self, X):
        scores = []
        for idx in range(len(self.classes)):
            # Tính tổng log-likelihood trên tất cả 784 đặc trưng
            log_likelihood = np.sum(
                self.log_pdf(X, self.mean[idx], self.var[idx]), axis=1
            )
            # log P(C_k | x) tỷ lệ thuận với log P(C_k) + sum(log P(x_i | C_k))
            scores.append(log_likelihood + np.log(self.prior[idx]))

        # Chọn lớp có điểm số log-posterior cao nhất cho mỗi mẫu
        return self.classes[np.argmax(scores, axis=0)]

def gaussian_log_pdf(X, mean, var):
    # log N(x_i | mean_i, var_i), shape (n_samples, n_features).
    # X: (n_samples, n_features), mean, var: (n_features,) -> NumPy broadcasting
    return -0.5 * np.log(2 * np.pi * var) - (X - mean) ** 2 / (2 * var)

def multinomial_log_pdf(X, feature_count, var, alpha=1.0):
    # log P(x_i | C_k) = x_i * log(theta_i), shape (n_samples, n_features).
    # 1. Tinh ti le theta_i cho tung dac trung (kem he so lam min Laplace alpha)
    theta = (feature_count + alpha) / np.sum(feature_count + alpha)
    
    # 2. Tinh log P(x_i | C_k) = x_i * log(theta_i)
    return X * np.log(theta)

class MultinomialNaiveBayes:
    # Lớp Multinomial Naive Bayes thuần NumPy (From Scratch) cho đặc trưng không âm.
    def __init__(self, log_pdf, alpha=1.0):
        self.log_pdf = log_pdf
        self.alpha = alpha

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.feature_count = np.zeros((n_classes, n_features))
        self.prior = np.zeros(n_classes)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.feature_count[idx] = X_c.sum(axis=0)
            self.prior[idx] = X_c.shape[0] / n_samples

        return self

    def predict(self, X):
        scores = []
        for idx in range(len(self.classes)):
            log_likelihood = np.sum(
                self.log_pdf(X, self.feature_count[idx], None, alpha=self.alpha), axis=1
            )
            scores.append(log_likelihood + np.log(self.prior[idx]))
        return self.classes[np.argmax(scores, axis=0)]
