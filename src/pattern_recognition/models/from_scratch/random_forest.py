import numpy as np


class DecisionNode:
    # Dai dien cho mot nut trong cay quyet dinh CART
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self):
        return self.value is not None

class DecisionTreeFromScratch:
    # Cay quyet dinh phan lop thuan NumPy (Gini impurity, ho tro Random Subspace)
    def __init__(self, max_depth=15, min_samples_split=2, max_features="sqrt", random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.root = None
        self.rng = np.random.default_rng(random_state)

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y, minlength=10)
        p = counts / len(y)
        return 1.0 - np.sum(p ** 2)

    def _best_split(self, X, y, feature_indices):
        best_gain = -1.0
        split_idx, split_thresh = None, None
        current_gini = self._gini(y)
        n_samples = len(y)

        for feat in feature_indices:
            X_column = X[:, feat]
            thresholds = np.percentile(X_column, [20, 40, 60, 80])
            thresholds = np.unique(thresholds)

            for thresh in thresholds:
                left_mask = X_column <= thresh
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                n_l, n_r = np.sum(left_mask), np.sum(right_mask)
                gini_l = self._gini(y[left_mask])
                gini_r = self._gini(y[right_mask])
                child_gini = (n_l / n_samples) * gini_l + (n_r / n_samples) * gini_r
                gain = current_gini - child_gini

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat
                    split_thresh = thresh

        return split_idx, split_thresh, best_gain

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels <= 1 or n_samples < self.min_samples_split):
            leaf_value = np.bincount(y, minlength=10).argmax()
            return DecisionNode(value=leaf_value)

        if self.max_features == "sqrt":
            m_feat = max(1, int(np.sqrt(n_features)))
        elif isinstance(self.max_features, int):
            m_feat = min(n_features, self.max_features)
        else:
            m_feat = n_features

        feat_idxs = self.rng.choice(n_features, m_feat, replace=False)
        feat_idx, thresh, gain = self._best_split(X, y, feat_idxs)

        if gain <= 1e-6 or feat_idx is None:
            leaf_value = np.bincount(y, minlength=10).argmax()
            return DecisionNode(value=leaf_value)

        left_mask = X[:, feat_idx] <= thresh
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return DecisionNode(feat_idx, thresh, left, right)

    def fit(self, X, y):
        self.root = self._build_tree(X, y)
        return self

    def _predict_one(self, node, x):
        if node.is_leaf:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(node.left, x)
        return self._predict_one(node.right, x)

    def predict(self, X):
        return np.array([self._predict_one(self.root, x) for x in X])

class RandomForestFromScratch:
    # Rung ngau nhien phan lop thuan NumPy ket hop Bagging va Random Subspace
    def __init__(self, n_estimators=10, max_depth=12, max_features="sqrt", random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        self.trees = []
        n_samples = len(X)
        for i in range(self.n_estimators):
            boot_idxs = self.rng.choice(n_samples, n_samples, replace=True)
            tree = DecisionTreeFromScratch(
                max_depth=self.max_depth,
                max_features=self.max_features,
                random_state=self.rng.integers(1e6)
            )
            tree.fit(X[boot_idxs], y[boot_idxs])
            self.trees.append(tree)
        return self

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        majority_votes = []
        for i in range(X.shape[0]):
            votes = tree_preds[:, i]
            majority_votes.append(np.bincount(votes, minlength=10).argmax())
        return np.array(majority_votes)
