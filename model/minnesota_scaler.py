import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class MinnesotaLagScaler(BaseEstimator, TransformerMixin):
    """
    Implements a Minnesota-style prior for Ridge Regression via feature scaling.
    
    In Bayesian Vector Autoregression (BVAR), the Minnesota Prior assumes that 
    coefficients for distant lags should be shrunk more aggressively towards zero 
    than recent lags.
    
    **Mathematical Formulation:**
    Standard Ridge Regression minimizes:
    $$ ||y - X\beta||^2 + \lambda ||\beta||^2 $$
    
    By scaling a feature $x_j$ by a factor $s < 1$ (i.e., $x'_j = s \cdot x_j$), 
    the coefficient must increase to $\beta'_j = \beta_j / s$ to maintain the same predictive output.
    The penalty term becomes:
    $$ \lambda (\beta'_j)^2 = \lambda \frac{1}{s^2} \beta_j^2 $$
    
    This effectively imposes a variable-specific penalty $\lambda_j = \lambda / s^2$.
    To mimic the Minnesota Prior (where variance $\sigma^2_l \propto 1/l^{2\gamma}$), 
    we scale features at lag $l$ by $s_l = l^{-\text{power}}$. This results in an effective 
    penalty increasing with the lag order:
    $$ \lambda_l \propto l^{2 \cdot \text{power}} $$

    **Input Structure Assumption:**
    $X$ must be ordered by lag: $[Lag_{1}(Var_{1..K}), \dots, Lag_{p}(Var_{1..K})]$.
    """
    
    def __init__(self, n_lags: int, power: float = 2.0):
        self.n_lags = n_lags
        self.power = power

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = X.copy()
        n_samples, n_features = X.shape
        
        # Validation
        if n_features % self.n_lags != 0:
            raise ValueError(f"Feature count ({n_features}) must be divisible by n_lags ({self.n_lags}).")
        
        K = n_features // self.n_lags # Variables per lag
        
        # Apply scaling lag by lag
        for l in range(1, self.n_lags + 1):
            # Scale factor s = 1 / l^power
            scale_factor = 1.0 / (l ** self.power)
            
            start_idx = (l - 1) * K
            end_idx = l * K
            
            # Apply scaling in-place for this block
            X_new[:, start_idx:end_idx] = X[:, start_idx:end_idx] * scale_factor
            
        return X_new