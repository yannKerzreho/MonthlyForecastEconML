import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.base import clone, BaseEstimator, RegressorMixin
from typing import Dict, Any, Optional, List, Union

# Assuming LaggedModel is available in the path
from model.lagged_model import LaggedModel

class MultiTargetRidgeEnsemble(BaseEstimator, RegressorMixin):
    """
    Ensemble wrapper for $N$ independent Ridge regressors.
    
    Allows for target-specific regularization parameters ($\alpha_j$) 
    where standard multi-output Ridge enforces a single global $\alpha$.
    """
    def __init__(self, ridge_estimators: Dict[int, Ridge]):
        self.ridge_estimators = ridge_estimators
        self.n_targets = len(ridge_estimators)
        
    def fit(self, X, y):
        # Estimators are already fitted during construction
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predictions of shape (n_samples, n_targets)."""
        n_samples = X.shape[0]
        preds = np.empty((n_samples, self.n_targets))
        
        for i in range(self.n_targets):
            if i in self.ridge_estimators:
                preds[:, i] = self.ridge_estimators[i].predict(X)
            else:
                raise ValueError(f"No estimator found for target {i}")
        return preds


class HyperoptLaggedModel(LaggedModel):
    """
    LaggedModel with automated hyperparameter optimization via Cross-Validation.

    **Regularization Strategies (`alpha_per_target`):**
    
    1.  **'global'**: Optimizes a single scalar $\alpha$ for the entire target matrix $Y$.
        $$ \min_{\alpha} \sum_{j=1}^N \text{MSE}(y_j, \hat{y}_j(\alpha)) $$
        *Fast, but may under-regularize noisy targets.*

    2.  **'independent'**: Optimizes $\alpha_j$ separately for each target $y_j$.
        $$ \forall j: \min_{\alpha_j} \text{MSE}(y_j, \hat{y}_j(\alpha_j)) $$
        *Precise, but computationally expensive ($N$ separate optimizations).*

    3.  **'two_stage'**: Hybrid approach.
        1. Find global optimum $\alpha^*$.
        2. Fine-tune locally: $\alpha_j \in \{ \alpha^* \cdot m \mid m \in M_{local} \}$.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        if not self.config.get('param_grid'):
            raise ValueError("Config must have 'param_grid'")
        
        # CV Settings
        self.cv_ratio = config.get('cv_ratio', 0.2)
        self.cv_splits = config.get('cv_splits', 5)
        self.search_type = config.get('search_type', 'grid')
        
        # Strategy Validation
        self.alpha_strategy = config.get('alpha_per_target', 'two_stage')
        valid_strats = ['global', 'two_stage', 'independent']
        assert self.alpha_strategy in valid_strats, \
            f"Invalid strategy: {self.alpha_strategy}. Must be in {valid_strats}"
            
        self.alpha_multipliers = config.get('alpha_local_multiplier', [0.5, 0.75, 1.0, 1.5, 2.0])
        
        # State
        self.alpha_global: Optional[float] = None
        self.alphas_per_target: Optional[Dict[int, float]] = None

    def _fit_solver(self, X: np.ndarray, Y: np.ndarray) -> Any:
        """
        Dispatches the fitting process to the configured optimization strategy.
        """
        if Y.ndim == 1: Y = Y.reshape(-1, 1)
        assert X.shape[0] == Y.shape[0], f"X/Y length mismatch: {X.shape[0]} != {Y.shape[0]}"
        
        # Standardize param_grid keys
        param_grid = self.config.get('param_grid').copy()
        if 'estimator__alpha' in param_grid:
            param_grid['alpha'] = param_grid.pop('estimator__alpha')
            
        # Fallback for non-Ridge estimators (e.g., RandomForest)
        if 'alpha' not in param_grid:
             if self.verbose: print("   [Hyperopt] Generic optimization (non-Ridge)...")
             return self._fit_generic(X, Y, param_grid)

        # Dispatch Ridge Strategy
        if self.alpha_strategy == 'global':
            return self._fit_global_alpha(X, Y, param_grid)
        elif self.alpha_strategy == 'independent':
            return self._fit_independent_alpha(X, Y, param_grid)
        elif self.alpha_strategy == 'two_stage':
            return self._fit_two_stage_alpha(X, Y, param_grid)
            
        return None

    def _get_cv_strategy(self, T: int) -> TimeSeriesSplit:
        test_size = max(1, int(np.floor((T * self.cv_ratio) / self.cv_splits)))
        return TimeSeriesSplit(n_splits=self.cv_splits, test_size=test_size)

    def _fit_global_alpha(self, X: np.ndarray, Y: np.ndarray, param_grid: dict) -> Any:
        """Optimizes a single alpha for all targets simultaneously."""
        cv = self._get_cv_strategy(X.shape[0])
        
        search = GridSearchCV(
            estimator=Ridge(fit_intercept=True), 
            param_grid=param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            refit=True, # Fits best estimator on full X, Y
            verbose=0
        )
        
        search.fit(X, Y)
        
        self.alpha_global = search.best_params_['alpha']
        if self.verbose:
            print(f"   [Hyperopt GLOBAL] Best α = {self.alpha_global:.6f}")
            
        return search.best_estimator_

    def _fit_independent_alpha(self, X: np.ndarray, Y: np.ndarray, param_grid: dict) -> MultiTargetRidgeEnsemble:
        """Optimizes alpha independently for each target column."""
        n_targets = Y.shape[1]
        cv = self._get_cv_strategy(X.shape[0])
        
        if self.verbose:
            print(f"   [Hyperopt INDEP] Optimizing {n_targets} targets independently...")
            
        ridge_estimators = {}
        self.alphas_per_target = {}
        
        for i in range(n_targets):
            search = GridSearchCV(
                estimator=Ridge(fit_intercept=True),
                param_grid=param_grid,
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                refit=True,
                verbose=0
            )
            
            search.fit(X, Y[:, i])
            
            ridge_estimators[i] = search.best_estimator_
            self.alphas_per_target[i] = search.best_params_['alpha']
            
            if self.verbose:
                print(f"      Target {i}: α = {self.alphas_per_target[i]:.6f}")
                
        return MultiTargetRidgeEnsemble(ridge_estimators)

    def _fit_two_stage_alpha(self, X: np.ndarray, Y: np.ndarray, param_grid: dict) -> MultiTargetRidgeEnsemble:
        """Hybrid: Global search followed by local fine-tuning."""
        
        # Phase 1: Global Search
        if self.verbose: print(f"   [Hyperopt 2-STAGE] Phase 1: Global Search...")
        _ = self._fit_global_alpha(X, Y, param_grid) # Updates self.alpha_global
        base_alpha = max(self.alpha_global, 1e-9)
        
        # Phase 2: Local Fine-tuning
        n_targets = Y.shape[1]
        cv = self._get_cv_strategy(X.shape[0])
        
        # Define local grid around global optimum
        local_alphas = sorted(list(set([base_alpha * m for m in self.alpha_multipliers if base_alpha * m > 0])))
        
        if self.verbose: 
            print(f"   [Hyperopt 2-STAGE] Phase 2: Fine-tuning {n_targets} targets around {base_alpha:.1e}...")
            
        ridge_estimators = {}
        self.alphas_per_target = {}
        
        for i in range(n_targets):
            search = GridSearchCV(
                estimator=Ridge(fit_intercept=True),
                param_grid={'alpha': local_alphas},
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                refit=True,
                verbose=0
            )
            
            search.fit(X, Y[:, i])
            
            ridge_estimators[i] = search.best_estimator_
            self.alphas_per_target[i] = search.best_params_['alpha']
            
            if self.verbose:
                ratio = self.alphas_per_target[i] / base_alpha
                print(f"      Target {i}: α = {self.alphas_per_target[i]:.6f} ({ratio:.2f}x)")
                
        return MultiTargetRidgeEnsemble(ridge_estimators)

    def _fit_generic(self, X: np.ndarray, Y: np.ndarray, param_grid: dict) -> Any:
        """Fallback for non-Ridge estimators (e.g., Random Forest, SVR)."""
        cv = self._get_cv_strategy(X.shape[0])
        base_est = clone(self.base_estimator)
        
        SearchClass = RandomizedSearchCV if self.search_type == 'random' else GridSearchCV
        
        search = SearchClass(
            estimator=base_est,
            param_grid=param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            refit=True,
            verbose=0
        )
        
        search.fit(X, Y)
        
        if self.verbose:
            print(f"   [Hyperopt Generic] Best params: {search.best_params_}")
            
        return search.best_estimator_