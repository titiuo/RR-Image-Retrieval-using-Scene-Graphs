from sklearn.linear_model import LogisticRegression
import numpy as np

def train_platt_params(raw_scores, true_labels):
    """
    Learns parameters A and B for Platt scaling.
    
    Args:
        raw_scores: (N,) array of raw classifier scores from validation set.
        true_labels: (N,) array of binary labels (0 or 1).
        
    Returns:
        A, B: Scalar parameters for the sigmoid function.
    """
    # Reshape for sklearn (N, 1)
    X = raw_scores.reshape(-1, 1)
    
    # Train Logistic Regression (which is essentially Platt Scaling here)
    lr = LogisticRegression(C=1e9, solver='lbfgs') # High C to reduce regularization
    lr.fit(X, true_labels)
    
    # Extract parameters
    # Note: sklearn uses form 1 / (1 + exp(-(coef*x + intercept)))
    # So A = -coef, B = -intercept to match the standard Platt formula above
    # Or simply use the predict_proba function directly later.
    A = -lr.coef_[0][0]
    B = -lr.intercept_[0]

    return A,B


def apply_platt_scaling(raw_scores, lr_model):
    """
    Converts raw scores to probabilities.
    """
    X = raw_scores.reshape(-1, 1)
    # Returns [prob_class_0, prob_class_1]
    probs = lr_model.predict_proba(X) 
    
    # We only care about P(class=1)
    return probs[:, 1]