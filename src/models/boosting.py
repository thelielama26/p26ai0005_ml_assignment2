import numpy as np
from sklearn.tree import DecisionTreeClassifier

class SAMMEAdaBoost:
    """
    Multi-class SAMME AdaBoost classifier implemented strictly from scratch.
    
    Equations applied per prompt specs:
    - Alpha update rule: \alpha_t = log((1 - \epsilon_t) / \epsilon_t) + log(K - 1)
    - Unnormalized score matrix: S_k(x) = \sum_{t} \alpha_t * 1[h_t(x) = k]
    - Normalized weighted vote vector: q_k(x) = S_k(x) / \sum_{t} \alpha_t
    """
    def __init__(self, n_estimators=50, max_depth=1, n_classes=40, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_classes = n_classes
        self.random_state = random_state
        self.estimators = []
        self.estimator_weights = []

    def fit(self, X, y, verbose_log=False):
        n_samples = X.shape[0]
        # Initial sample weights: uniform distribution 1/N
        w = np.ones(n_samples, dtype=np.float64) / n_samples
        rng = np.random.default_rng(self.random_state)

        self.estimators = []
        self.estimator_weights = []
        log_history = []

        for t in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=rng.integers(0, 100000)
            )
            # Fit weak learner on weighted samples
            tree.fit(X, y, sample_weight=w)
            preds = tree.predict(X)
            
            incorrect = (preds != y).astype(np.float64)
            # Compute weighted error: \epsilon_t = \sum w_i * 1(preds_i != y_i) / \sum w_i
            weighted_error = np.sum(w * incorrect) / np.sum(w)
            
            # Theoretical random chance upper bound for multi-class error: 1 - 1/K
            random_chance_error = 1.0 - (1.0 / self.n_classes)
            
            # Explicit handling of weak learners not better than random prediction
            if weighted_error >= random_chance_error:
                if len(self.estimators) == 0:
                    # Edge case: First estimator fails random chance, assign minimal weight
                    alpha_t = 1e-6
                    self.estimators.append(tree)
                    self.estimator_weights.append(alpha_t)
                break
                
            if weighted_error <= 0.0:
                # Perfect classifier reached
                alpha_t = 1.0
                self.estimators.append(tree)
                self.estimator_weights.append(alpha_t)
                
                if verbose_log:
                    log_history.append({
                        "round": t + 1,
                        "weighted_error": 0.0,
                        "alpha": float(alpha_t),
                        "max_sample_weight": float(np.max(w)),
                        "top10_weight_sum": float(np.sum(np.sort(w)[-10:])),
                        "initial_weights": w.tolist(),
                        "predictions": preds.tolist()
                    })
                break

            # SAMME weight computation equation
            alpha_t = np.log((1.0 - weighted_error) / weighted_error) + np.log(self.n_classes - 1)
            
            # Update sample weights: w_i = w_i * exp(\alpha_t * 1(preds_i != y_i))
            w *= np.exp(alpha_t * incorrect)
            # Normalize sample weights so sum(w) = 1
            w /= np.sum(w)
            
            self.estimators.append(tree)
            self.estimator_weights.append(alpha_t)
            
            if verbose_log:
                log_history.append({
                    "round": t + 1,
                    "weighted_error": float(weighted_error),
                    "alpha": float(alpha_t),
                    "max_sample_weight": float(np.max(w)),
                    "top10_weight_sum": float(np.sum(np.sort(w)[-10:])),
                    "sample_weights": w.copy().tolist(),
                    "predictions": preds.tolist()
                })

        return log_history if verbose_log else self

    def predict_score_matrix(self, X):
        """Calculates unnormalized vote scores S_k(x) across all weak learners."""
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, self.n_classes), dtype=np.float64)
        for tree, alpha in zip(self.estimators, self.estimator_weights):
            preds = tree.predict(X)
            for i, p in enumerate(preds):
                scores[i, p] += alpha
        return scores

    def predict_proba(self, X):
        """Returns normalized weighted vote vector q_k(x) = S_k(x) / sum_t(alpha_t)."""
        scores = self.predict_score_matrix(X)
        total_weight = np.sum(self.estimator_weights)
        if total_weight == 0:
            return np.ones((X.shape[0], self.n_classes)) / float(self.n_classes)
        return scores / total_weight

    def predict(self, X):
        scores = self.predict_score_matrix(X)
        return np.argmax(scores, axis=1)