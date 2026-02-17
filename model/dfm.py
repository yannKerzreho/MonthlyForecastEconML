import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional
from model.base_model import BaseModel
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

class DFMModel(BaseModel):
    """
    Robust Dynamic Factor Model (DFM) implementation wrapping `statsmodels`.

    This class implements a Dynamic Factor Model intended for high-dimensional 
    time series data, reducing dimensionality by estimating latent factors that 
    drive the comovement of observed variables.

    **Architecture: PANEL ONLY**
    This implementation enforces a 'panel' structure. A separate, independent 
    Dynamic Factor Model is fitted for each country (or entity) to capture 
    local latent dynamics. It does not pool parameters across entities.

    **Mathematical Formulation (State-Space Representation):**
    
    For a given entity $i$ at time $t$, the model assumes the observed vector 
    $y_{i,t}$ is driven by a lower-dimensional vector of unobserved factors $f_{i,t}$:

    1. **Observation Equation:**
       $$y_{i,t} = \Lambda_i f_{i,t} + \epsilon_{i,t}$$
       Where:
       - $y_{i,t}$ is the $(N \times 1)$ vector of observed features.
       - $\Lambda_i$ is the $(N \times k)$ factor loading matrix.
       - $f_{i,t}$ is the $(k \times 1)$ vector of latent factors.
       - $\epsilon_{i,t}$ is the $(N \times 1)$ idiosyncratic error term (noise).

    2. **Transition Equation (VAR structure on factors):**
       $$f_{i,t} = A_{i,1} f_{i,t-1} + \dots + A_{i,p} f_{i,t-p} + \eta_{i,t}$$
       Where the factors follow a Vector Autoregression of order $p$ (`factor_order`).

    **Configuration Parameters (`config`):**
    - `k_factors` (int): Number of unobserved factors to estimate ($k$). Default: 1.
    - `factor_order` (int): The order of the vector autoregression for the factors ($p$). Default: 1.
    - `error_order` (int): The order of the autoregressive error term $\epsilon_{i,t}$. Default: 0.
    - `structure` (str): Forced to 'panel'.
    - `seed` (int): Random seed for reproducibility.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        # 1. Configuration Validation
        self.k_factors = config.get('k_factors', 1)
        self.factor_order = config.get('factor_order', 1)
        self.error_order = config.get('error_order', 0)
        
        assert self.k_factors >= 1, f"k_factors must be >= 1, got {self.k_factors}"
        assert self.factor_order >= 0, f"factor_order must be >= 0, got {self.factor_order}"
        assert self.error_order >= 0, f"error_order must be >= 0, got {self.error_order}"
        
        # 2. Enforce Panel Structure
        self.structure = config.get('structure', 'panel')
        if self.structure != 'panel':
            if config.get('verbose', False):
                print("DFM implies specific latent factors per entity.")
                print("   → Enforcing 'panel' structure (ignoring 'pooled').")
            self.structure = 'panel'
        
        self.verbose = config.get('verbose', False)
        
        self.results: Dict[str, Any] = {}
        self._fitted = False
        
        self.seed = config.get('seed', 42)

    def set_seed(self, seed: int):
        """Sets the random seed for numpy to ensure reproducibility."""
        self.seed = seed
        np.random.seed(seed)

    def train_and_predict(
        self,
        data: Dict,
        horizon: int,
        train: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Executes the training and prediction pipeline for the Dynamic Factor Model.

        Args:
            data (Dict): Dictionary containing 'meta' (countries, features, targets) 
                         and 'train' (time series data dict).
            horizon (int): Forecasting horizon in time steps.
            train (bool): If True, fits the models before predicting.

        Returns:
            Tuple[np.ndarray, Dict]: 
                - Predictions array of shape (n_countries, horizon, n_targets).
                - Auxiliary dictionary (empty in this implementation).
        """
        
        countries = data['meta']['countries']
        feature_cols = data['meta']['features']
        target_cols = data['meta']['target_cols']
        
        np.random.seed(self.seed)
        target_indices = []
        for c in target_cols:
            try:
                target_indices.append(feature_cols.index(c))
            except ValueError:
                raise ValueError(f"Target column '{c}' not found in features.")
        
        n_targets = len(target_cols)
        n_features = len(feature_cols)
        train_data = data.get('train', {})
        
        if not train_data:
            if self.verbose: print("No training data provided.")
            return np.full((len(countries), horizon, n_targets), np.nan), {}
        
        preds = np.full((len(countries), horizon, n_targets), np.nan)

        if train:
            if self.verbose:
                print(f"   [DFM] Training on {len(countries)} countries...")
                
            for c in countries:
                ts = train_data.get(c)
                
                # A. Data Validation
                if ts is None:
                    continue

                min_len = max(self.k_factors + self.factor_order + 5, 20)
                if len(ts) < min_len:
                    if self.verbose: print(f"Skip {c}: insufficient history ({len(ts)} < {min_len})")
                    self.results[c] = None
                    continue
                
                if ts.shape[1] != n_features:
                    if self.verbose: print(f"Skip {c}: feature mismatch")
                    self.results[c] = None
                    continue

                # B. Fitting Statsmodels
                try:
                    model = DynamicFactor(
                        ts, 
                        k_factors=self.k_factors, 
                        factor_order=self.factor_order, 
                        error_order=self.error_order
                    )
                    res = model.fit(disp=False, maxiter=200)
                    self.results[c] = res
                    
                except Exception as e:
                    if self.verbose: print(f"Fit Error {c}: {e}")
                    self.results[c] = None
            
            self._fitted = True

        if not self._fitted:
            return preds, {}

        failed_count = 0
        
        for i, c in enumerate(countries):
            # Check model availability
            res = self.results.get(c)
            if res is None:
                continue
                
            try:
                # Forecast
                # Statsmodels handles the recursion of factors and the state space 
                # matrices to project forward.
                forecast_res = res.forecast(steps=horizon)
                
                # Output Type Robustness (Series vs DataFrame vs Array)
                if isinstance(forecast_res, (pd.DataFrame, pd.Series)):
                    vals = forecast_res.values
                else:
                    vals = np.asarray(forecast_res)
                
                # Dimensions Robustness
                if vals.ndim == 1:
                    vals = vals.reshape(-1, 1)
                
                # Final Validations
                if vals.shape[0] != horizon:
                    raise ValueError(f"Output length {vals.shape[0]} != horizon {horizon}")
                
                if vals.shape[1] < max(target_indices) + 1:
                    raise ValueError(f"Output width {vals.shape[1]} too small for targets")
                
                # Extraction
                preds[i, :, :] = vals[:, target_indices]
                
            except Exception as e:
                failed_count += 1
                if self.verbose: print(f"Pred Error {c}: {e}")
                continue

        return preds, {}