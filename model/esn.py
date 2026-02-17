import numpy as np
import warnings
from typing import Dict, Tuple, Optional, Any, List
from model.base_model import BaseModel
from model.utils import ohe
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.metrics import mean_squared_error

try:
    from reservoirpy.nodes import Reservoir
except ImportError:
    raise ImportError("reservoirpy is required. Install with: pip install reservoirpy")


class MultiTargetRidgeReadout:
    """
    A wrapper for managing $N$ independent Ridge regression models, one per target variable.
    
    This replaces the standard ReservoirPy Ridge node to allow for variable-specific 
    regularization penalties ($\lambda_j$).

    **Optimization Objective:**
    For each target dimension $j$, we solve:
    $$ \min_{w_j} \sum_{t} (y_{t,j} - w_j^T x_t)^2 + \lambda_j ||w_j||_2^2 $$
    """
    
    def __init__(self, ridge_models: List[SklearnRidge]):
        self.ridge_models = ridge_models
        self.n_targets = len(ridge_models)
        assert self.n_targets > 0, "Must have at least 1 ridge model"
        
    def fit(self, X: np.ndarray, Y: np.ndarray) -> 'MultiTargetRidgeReadout':
        """Fits each internal Ridge model on its corresponding target column."""
        assert Y.shape[1] == self.n_targets, \
            f"Y has {Y.shape[1]} targets, expected {self.n_targets}"
        assert X.shape[0] == Y.shape[0], \
            f"X/Y mismatch: {X.shape[0]} != {Y.shape[0]}"
        
        for i, ridge in enumerate(self.ridge_models):
            ridge.fit(X, Y[:, i])
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Aggregates predictions from all sub-models.
        Handles both 1D (single state vector) and 2D (batch) inputs.
        """
        is_1d = X.ndim == 1
        if is_1d:
            X = X.reshape(1, -1)
            
        n_samples = X.shape[0]
        preds = np.empty((n_samples, self.n_targets))
        
        for i, ridge in enumerate(self.ridge_models):
            preds[:, i] = ridge.predict(X)
            
        return preds[0] if is_1d else preds
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Pipeline compatibility alias."""
        return self.predict(X)


class ESNModel(BaseModel):
    """
    Echo State Network (ESN) with Multi-Ridge optimization.

    This model implements a Reservoir Computing approach where the recurrent kernel 
    is fixed and only the readout weights are trained. It supports both 'Pooled' 
    (one model for all entities) and 'Panel' (one model per entity) structures.

    **State-Space Formulation:**
    
    1.  **Reservoir State Update:**
        $$ x_t = (1 - \delta) x_{t-1} + \delta \tanh(W_{in} u_t + W_{res} x_{t-1}) $$
        Where:
        - $u_t$ is the input vector (features).
        - $x_t$ is the reservoir state (hidden units).
        - $\delta$ is the leaking rate (`lr`).
        - $W_{res}$ is the sparse recurrent weight matrix (spectral radius `sr`).

    2.  **Readout (Prediction):**
        $$ \hat{y}_t = W_{out} x_t $$
        The weights $W_{out}$ are computed via Ridge regression (see `MultiTargetRidgeReadout`).

    Args:
        config (dict): Configuration dictionary containing hyperparameters 
                       (units, sr, lr, input_scaling, ridge_alphas, etc.).
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        # Hyperparameters
        self.units = config.get('units', 100)
        self.sr = config.get('sr', 0.95)
        self.lr = config.get('lr', 0.6)
        self.input_scaling = config.get('input_scaling', 1.0)
        self.seed = config.get('seed', 42)
        
        # Readout & Validation
        self.ridge_alphas = config.get('ridge_alphas', [10**(i) for i in range(-4, 4)])
        self.cv_ratio = config.get('cv_ratio', 0.2)
        self.warmup = config.get('warmup', 12)
        
        self.structure = config.get('structure', 'pooled')
        self.ohe = config.get('ohe', False) # If True, appends Entity ID to input
        self.verbose = config.get('verbose', False)
        
        # State Storage
        self.reservoirs: Dict[str, Any] = {}
        self.readouts: Dict[str, Any] = {}
        self.alphas_per_target: Optional[Dict[int, float]] = None
        self._fitted = False

    def _init_reservoir(self, seed_offset: int = 0) -> Reservoir:
        """Initializes a ReservoirPy node with fixed seeds for reproducibility."""
        current_seed = self.seed + seed_offset if self.seed is not None else None
        return Reservoir(
            units=self.units,
            lr=self.lr,
            sr=self.sr,
            input_scaling=self.input_scaling,
            seed=current_seed
        )

    def _fit_readout_with_cv(
        self, 
        X_train: np.ndarray, Y_train: np.ndarray, 
        X_val: np.ndarray, Y_val: np.ndarray
    ) -> MultiTargetRidgeReadout:
        """
        Performs a manual GridSearch to find the optimal regularization alpha 
        for *each* target variable independently.
        """
        n_targets = Y_train.shape[1]
        
        # Exclude warmup period from training metrics
        X_tr_warm = X_train[self.warmup:]
        Y_tr_warm = Y_train[self.warmup:]
        
        ridge_models = []
        best_alphas = {}
            
        for i in range(n_targets):
            y_tr_col = Y_tr_warm[:, i]
            y_val_col = Y_val[:, i]
            
            best_mse = np.inf
            best_alpha = self.ridge_alphas[0]
            
            # Grid Search (efficient reuse of data matrices)
            for alpha in self.ridge_alphas:
                model = SklearnRidge(alpha=alpha, fit_intercept=True)
                model.fit(X_tr_warm, y_tr_col)
                pred_val = model.predict(X_val)
                mse = mean_squared_error(y_val_col, pred_val)
                
                if mse < best_mse:
                    best_mse = mse
                    best_alpha = alpha
            
            # Store the configuration; final fit happens in wrapper
            ridge_final = SklearnRidge(alpha=best_alpha, fit_intercept=True)
            ridge_models.append(ridge_final)
            best_alphas[i] = best_alpha
            
        if self.verbose:
            alphas = list(best_alphas.values())
            print(f"   [ESN] Alphas: min={min(alphas):.1e}, max={max(alphas):.1e}, mean={np.mean(alphas):.1e}")
            
        self.alphas_per_target = best_alphas
        
        # Fit final wrapper on combined Train + Validation sets
        readout = MultiTargetRidgeReadout(ridge_models)
        X_full = np.vstack([X_train, X_val])
        Y_full = np.vstack([Y_train, Y_val])
        
        readout.fit(X_full[self.warmup:], Y_full[self.warmup:])
        return readout

    def train_and_predict(self, data: Dict, horizon: int, train: bool = True) -> Tuple[np.ndarray, Dict]:
        countries = data['meta']['countries']
        feature_cols = data['meta']['features']
        target_cols = data['meta']['target_cols']
        target_indices = [feature_cols.index(c) for c in target_cols]
        n_targets = len(target_cols)
        n_countries = len(countries)
        
        train_data = data['train']
        preds = np.full((len(countries), horizon, n_targets), np.nan)

        # 1. Training Phase
        if train:
            if self.structure == 'pooled':
                # Collect states across all entities to train a global readout
                reservoir = self._init_reservoir()
                X_tr_list, Y_tr_list, X_val_list, Y_val_list = [], [], [], []
                
                for i, c in enumerate(countries):
                    ts = train_data.get(c)
                    if ts is None or len(ts) < self.warmup + 5: continue
                    
                    # Prepare Inputs/Targets with optional OHE
                    if self.ohe:
                        Y_raw = ts[1:]
                        ts_in = ohe(ts, i, n_countries)
                        X_raw = ts_in[:-1]
                    else:
                        X_raw, Y_raw = ts[:-1], ts[1:]
                    
                    # Generate Reservoir States
                    states = reservoir.run(X_raw)
                    reservoir.reset()
                    
                    # Temporal Split
                    split = int(len(states) * (1 - self.cv_ratio))
                    if split < 1: split = 1
                    
                    X_tr_list.append(states[:split])
                    Y_tr_list.append(Y_raw[:split])
                    X_val_list.append(states[split:])
                    Y_val_list.append(Y_raw[split:])
                
                if X_tr_list:
                    readout = self._fit_readout_with_cv(
                        np.vstack(X_tr_list), np.vstack(Y_tr_list),
                        np.vstack(X_val_list), np.vstack(Y_val_list)
                    )
                    self.reservoirs['GLOBAL'] = reservoir
                    self.readouts['GLOBAL'] = readout

            else: # Panel Structure (Entity-specific models)
                for i, c in enumerate(countries):
                    ts = train_data.get(c)
                    if ts is None or len(ts) < self.warmup: continue
                    
                    if self.ohe:
                        Y_raw = ts[1:]
                        ts_in = ohe(ts, i, n_countries)
                        X_raw = ts_in[:-1]
                    else:
                        X_raw, Y_raw = ts[:-1], ts[1:]
                    
                    reservoir = self._init_reservoir(seed_offset=i)
                    states = reservoir.run(X_raw)
                    reservoir.reset()

                    split = int(len(states) * (1 - self.cv_ratio))
                    readout = self._fit_readout_with_cv(
                        states[:split], Y_raw[:split],
                        states[split:], Y_raw[split:]
                    )
                    self.reservoirs[c] = reservoir
                    self.readouts[c] = readout

            self._fitted = True

        if not self._fitted: return preds, {}

        # 2. Prediction Phase (Generative/Autoregressive)
        for i, c in enumerate(countries):
            ts = train_data.get(c)
            if ts is None: continue
            
            ts_hist = ohe(ts, i, n_countries) if self.ohe else ts
            key = 'GLOBAL' if self.structure == 'pooled' else c
            
            reservoir = self.reservoirs.get(key)
            readout = self.readouts.get(key)
            if reservoir is None or readout is None: continue
            
            # Warmup reservoir state with history
            _ = reservoir.run(ts_hist[:-1])
            current_input = ts_hist[-1, :] 
            
            for h in range(horizon):
                # Update state -> Predict -> Store
                state = reservoir(current_input)
                pred_full = readout.predict(state)
                preds[i, h, :] = pred_full[target_indices]
                
                # Feedback loop: prediction becomes next input
                if self.ohe and n_countries > 1:
                    ohe_vec = np.zeros(n_countries)
                    ohe_vec[i] = 1.0
                    current_input = np.concatenate([pred_full, ohe_vec])
                else:
                    current_input = pred_full
            reservoir.reset()

        return preds, {'alphas_per_target': self.alphas_per_target}