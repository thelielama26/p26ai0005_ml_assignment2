import numpy as np
from sklearn.tree import DecisionTreeClassifier

class CustomBaggingClassifier:
    """
    Custom Bagging Classifier built strictly from scratch using DecisionTreeClassifier base estimators.
    Calculates bootstrap coverage, out-of-bag accuracy, pairwise disagreement, and vote margin.
    """
    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=1, random_state=42, n_classes=40):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_classes = n_classes
        self.estimators = []
        self.bootstrap_indices = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        self.estimators = []
        self.bootstrap_indices = []

        for _ in range(self.n_estimators):
            boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
            self.bootstrap_indices.append(boot_idx)
            
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=rng.integers(0, 100000)
            )
            tree.fit(X[boot_idx], y[boot_idx])
            self.estimators.append(tree)
        return self

    def predict_vote_counts(self, X):
        """Returns raw vote matrix of shape (n_samples, n_classes)."""
        n_samples = X.shape[0]
        votes = np.zeros((n_samples, self.n_classes), dtype=int)
        for tree in self.estimators:
            preds = tree.predict(X)
            for i, p in enumerate(preds):
                votes[i, p] += 1
        return votes

    def predict(self, X):
        votes = self.predict_vote_counts(X)
        return np.argmax(votes, axis=1)

    def predict_proba(self, X):
        votes = self.predict_vote_counts(X)
        return votes / float(len(self.estimators))

    def compute_oob_accuracy(self, X, y):
        """Calculates out-of-bag (OOB) accuracy across training samples."""
        n_samples = X.shape[0]
        oob_votes = np.zeros((n_samples, self.n_classes))
        
        for i, tree in enumerate(self.estimators):
            boot_set = set(self.bootstrap_indices[i])
            oob_mask = [idx for idx in range(n_samples) if idx not in boot_set]
            if len(oob_mask) > 0:
                preds = tree.predict(X[oob_mask])
                for idx, p in zip(oob_mask, preds):
                    oob_votes[idx, p] += 1
                    
        has_oob = np.sum(oob_votes, axis=1) > 0
        if not np.any(has_oob):
            return 0.0
        valid_oob_preds = np.argmax(oob_votes[has_oob], axis=1)
        return float(np.mean(valid_oob_preds == y[has_oob]))

    def compute_metrics(self, X):
        """Computes average pairwise disagreement between trees and mean vote margin."""
        n_samples = X.shape[0]
        B = len(self.estimators)
        if B <= 1:
            return 0.0, 1.0

        all_preds = np.array([tree.predict(X) for tree in self.estimators])  # Shape: (B, N)
        
        # 1. Average Pairwise Disagreement
        disagreements = 0
        pairs = 0
        for i in range(B):
            for j in range(i + 1, B):
                disagreements += np.sum(all_preds[i] != all_preds[j])
                pairs += n_samples
        avg_disagreement = float(disagreements / pairs) if pairs > 0 else 0.0

        # 2. Mean Vote Margin
        votes = self.predict_vote_counts(X)
        sorted_votes = np.sort(votes, axis=1)[:, ::-1]
        top1 = sorted_votes[:, 0] / float(B)
        top2 = sorted_votes[:, 1] / float(B)
        mean_vote_margin = float(np.mean(top1 - top2))

        return avg_disagreement, mean_vote_margin