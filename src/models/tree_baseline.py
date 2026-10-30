import numpy as np
from sklearn.tree import DecisionTreeClassifier

class SingleTreeBaseline:
    """Wrapper for scikit-learn's DecisionTreeClassifier to extract specific tree metrics."""
    def __init__(self, max_depth=None, min_samples_leaf=1, random_state=42):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_tree_stats(self, X_train):
        """Calculates leaves count, mean samples per leaf, and single-sample leaf fraction."""
        tree = self.model.tree_
        num_leaves = int(np.sum(tree.children_left == -1))
        
        leaf_indices = self.model.apply(X_train)
        unique_leaves, counts = np.unique(leaf_indices, return_counts=True)
        
        mean_samples = float(np.mean(counts))
        single_sample_leaves = int(np.sum(counts == 1))
        frac_single_sample_leaves = float(single_sample_leaves / len(unique_leaves))
        
        return {
            "num_leaves": num_leaves,
            "mean_samples_per_leaf": mean_samples,
            "frac_single_sample_leaves": frac_single_sample_leaves
        }