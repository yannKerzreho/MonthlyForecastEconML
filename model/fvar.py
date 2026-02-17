import numpy as np
import warnings
from typing import Dict, Tuple, Optional, Any
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR

# Assuming BaseModel is accessible
from model.base_model import BaseModel

class FVAR(BaseModel):
    """
    Factor-Augmented Vector Autoregression (FA-VAR) / Two-Step DFM.

    This model implements the classic Stock & Watson (2002) two-step approach 
    for forecasting in data-rich environments:
    1.  **Dimensionality Reduction (PCA):** Extract common factors from predictors.
    2.  **Dynamics (VAR):** Model the evolution of factors over time.

    **Mathematical Formulation:**

    1.  **Static Representation (PCA Step):**
        $$ X_t = \Lambda F_t + e_t $$
        where $X_t$ is the $(N \times 1)$ standardized observation vector and 
        $F_t$ is the $(k \times 1)$ vector of latent factors. $\Lambda$ and $F_t$ 
        are estimated via Principal Component Analysis on $X$.

    2.  **Dynamic Representation (VAR Step):**
        $$ F_t = \Phi_1 F_{t-1} + \dots + \Phi_p F_{t-p} + u_t $$
        The factors follow a Vector Autoregression of order $p$.

    3.  **Forecasting:**
        $$ \hat{X}_{T+h} = \hat{\Lambda} \hat{F}_{T+h|T} $$
        where $\hat{F}_{T+h|T}$ is obtained by iterating the VAR recursively.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        # Config mapping with fallbacks
        self.n_factors = config.get('n_factors', config.get('k_factors', 5))
        self.lags = config.get('lags', config.get('factor_order', 2))
        
        self.seed = config.get('seed', 42)
        self.verbose = config.get('verbose', False)
        
        # Model storage per entity
        self.models: Dict[str, Any] = {}

    def train_and_predict(
        self, 
        data: Dict, 
        horizon: int, 
        train: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        
        np.random.seed(self.seed)
        
        countries = data['meta']['countries']
        target_cols = data['meta']['target_cols']
        feature_cols = data['meta']['features']
        
        target_indices = [feature_cols.index(c) for c in target_cols]
        n_targets = len(target_cols)
        
        preds = np.full((len(countries), horizon, n_targets), np.nan)

        for i, country in enumerate(countries):
            ts = data['train'].get(country) # Shape (T_obs, N_features)
            
            if ts is None:
                continue
            
            # 1. TRAINING
            if train:
                try:
                    T, N = ts.shape
                    if T <= self.lags + 2:
                        if self.verbose:
                            print(f"[FVAR] {country}: Insufficient data ({T} obs)")
                        self.models[country] = None
                        continue

                    # A. Standardization (Crucial for PCA)
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(ts)
                    
                    # B. PCA (Factor Extraction)
                    actual_n_factors = min(self.n_factors, N, T - 1)
                    if actual_n_factors < 1: actual_n_factors = 1
                        
                    pca = PCA(n_components=actual_n_factors, random_state=self.seed)
                    factors = pca.fit_transform(X_scaled) # (T, k)
                    
                    # C. VAR on Factors
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        var_model = VAR(factors)
                        # ic=None forces usage of maxlags; otherwise it selects best IC
                        var_res = var_model.fit(maxlags=self.lags, ic=None) 
                    
                    self.models[country] = {
                        'scaler': scaler,
                        'pca': pca,
                        'var_res': var_res
                    }
                    
                except Exception as e:
                    if self.verbose: print(f"[FVAR] Train Error {country}: {e}")
                    self.models[country] = None

            # 2. PREDICTION
            model_bundle = self.models.get(country)
            if model_bundle is None:
                continue
                
            try:
                scaler = model_bundle['scaler']
                pca = model_bundle['pca']
                var_res = model_bundle['var_res']
                k_lag = var_res.k_ar 
                
                if len(ts) < k_lag: continue
                    
                last_X = ts[-k_lag:] 
                last_X_scaled = scaler.transform(last_X)
                last_factors = pca.transform(last_X_scaled) # Shape (k_lag, n_factors)
                
                # 2. Forecast Factors (VAR Recursion)
                # forecast_factors shape: (horizon, n_factors)
                forecast_factors = var_res.forecast(y=last_factors, steps=horizon)
                
                # 3. Project back to Observation Space
                # X_hat = F_hat * Lambda'
                forecast_scaled = pca.inverse_transform(forecast_factors)
                forecast_original = scaler.inverse_transform(forecast_scaled)
                
                # 4. Extract Targets
                preds[i, :, :] = forecast_original[:, target_indices]
                
            except Exception as e:
                if self.verbose: print(f"[FVAR] Pred Error {country}: {e}")
                continue

        return preds, {}