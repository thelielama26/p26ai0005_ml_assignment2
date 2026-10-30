import numpy as np

def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def compute_confidence(prob_matrix):
    return np.mean(np.max(prob_matrix, axis=1))

def compute_bias_variance(models_predictions, y_true, n_classes=40):
    """
    Computes Bias^2 and Variance according to Q3 equations[cite: 1]:
    Bias^2 = (1/N) * sum_i || \bar{p}(x_i) - e_i ||_2^2
    Variance = (1 / (N*R)) * sum_{i,r} || p_r(x_i) - \bar{p}(x_i) ||_2^2
    """
    R = len(models_predictions)        # Repetitions
    N = len(y_true)                    # Test samples
    
    # One-hot ground truth matrix (N, K)
    E = np.zeros((N, n_classes))
    for i, label in enumerate(y_true):
        E[i, label] = 1.0
        
    # Stack predictions (R, N, K)
    P = np.array(models_predictions)
    
    # Mean probability vector across repetitions (N, K)
    P_bar = np.mean(P, axis=0)
    
    # Bias^2
    bias_sq = np.mean(np.sum((P_bar - E) ** 2, axis=1))
    
    # Variance
    var_sum = 0.0
    for r in range(R):
        var_sum += np.sum((P[r] - P_bar) ** 2)
    variance = var_sum / (N * R)
    
    return float(bias_sq), float(variance)

def mean_confidence(probabilities):
    return float(np.mean(np.max(probabilities, axis=1)))


def evaluate_model(model, X, y):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    return {
        "accuracy": float(compute_accuracy(y, predictions)),
        "mean_winning_probability": mean_confidence(probabilities),
    }