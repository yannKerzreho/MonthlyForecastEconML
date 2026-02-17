from abc import abstractmethod
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import numpy as np
from typing import Dict, Tuple, Any, List
from model.base_model import BaseModel
from model.utils import JAXDataLoader

@eqx.filter_jit
def loss_fn(
    model: eqx.Module, 
    x: jnp.ndarray, 
    y: jnp.ndarray, 
    c_idx: jnp.ndarray, 
    horizon: int, 
    rho: float
) -> jnp.ndarray:
    """
    Computes the Temporally Weighted Mean Squared Error (TWMSE).

    **Mathematical Formulation:**
    $$ \mathcal{L}(\theta) = \frac{1}{B \cdot H} \sum_{b=1}^{B} \sum_{h=1}^{H} \rho^{h-1} || \hat{y}_{t+h}^{(b)} - y_{t+h}^{(b)} ||^2 $$
    
    Where:
    - $\rho \in (0, 1]$ is a discount factor prioritizing near-term accuracy.
    - $H$ is the forecasting horizon.
    - $B$ is the batch size.
    """
    # Vectorize model application over the batch
    preds = jax.vmap(lambda _x, _c: model(_x, _c, horizon))(x, c_idx)
    
    # Temporal weighting vector: [1, rho, rho^2, ...]
    weights = rho ** jnp.arange(horizon)
    weights = weights[None, :, None] # Broadcast to (1, Horizon, 1)
    
    return jnp.mean(((preds - y) ** 2) * weights)

@eqx.filter_jit
def make_step(
    model: eqx.Module, 
    opt_state: optax.OptState, 
    x: jnp.ndarray, 
    y: jnp.ndarray, 
    c_idx: jnp.ndarray, 
    optim: optax.GradientTransformation, 
    horizon: int, 
    rho: float
) -> Tuple[eqx.Module, optax.OptState, jnp.ndarray]:
    """
    Executes a single optimization step (Forward -> Backward -> Update).

    Uses `eqx.filter` to strictly separate differentiable parameters (weights/biases)
    from static configurations (activation functions, structural integers), preventing
    crashes in `optax` updates.
    """
    # 1. Compute Gradients
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y, c_idx, horizon, rho)
    
    # 2. Filter Parameters
    # Extract only differentiable (inexact) arrays for the optimizer
    params = eqx.filter(model, eqx.is_inexact_array)
    
    # 3. Optimizer Update
    # Passing 'params' allows AdamW to apply weight decay correctly:
    # p_new = p - lr * (grad + wd * p)
    updates, opt_state = optim.update(grads, opt_state, params)
    
    # 4. Apply Updates
    model = eqx.apply_updates(model, updates)
    
    return model, opt_state, loss_val

@eqx.filter_jit
def evaluate_batch_loss(model, x, y, c_idx, horizon, rho):
    """Computes TWMSE on a validation batch without gradient tracking."""
    preds = jax.vmap(lambda _x, _c: model(_x, _c, horizon))(x, c_idx)
    weights = rho ** jnp.arange(horizon)
    weights = weights[None, :, None]
    return jnp.mean(((preds - y) ** 2) * weights)

@eqx.filter_jit
def predict_batch(model, x_batch, c_idx_batch, horizon):
    """Inference wrapper for batch prediction."""
    return jax.vmap(lambda x, c: model(x, c, horizon))(x_batch, c_idx_batch)

class BaseJAXEstimator(BaseModel):
    """
    Abstract base class for JAX/Equinox-based time series estimators.
    
    Handles the boilerplate for:
    - JIT compilation and Gradient descent loops.
    - Optax optimizer initialization with proper parameter filtering.
    - Data loading and batching.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        s = config.get('seed')
        self.seed = int(s) if s is not None else 42
        self.rho = config.get('rho', 0.5)
        self.lr = config.get('learning_rate', 1e-3)
        self.epochs = config.get('epochs', 100)
        self.batch_size = config.get('batch_size', 64)
        self.window_size = config.get('window_size', 12)
        self.weight_decay = config.get('weight_decay', 1e-3)
        self.verbose = config.get('verbose', True)
        if self.verbose: print(f'weight decay = {self.weight_decay}')
        self.model = None
        self.optim = None
        self.opt_state = None
    
    def _create_optimizer(self) -> optax.GradientTransformation:
        """Constructs the optimizer chain (Gradient Clipping + AdamW)."""
        return optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=self.lr,
                weight_decay=self.weight_decay / self.lr# Optax handles decoupling
            )
        )

    @abstractmethod
    def build_model(self, key, n_features, n_countries, target_indices):
        """Must return an `eqx.Module` instance."""
        pass

    def _ensure_model_initialized(self, n_features, n_countries, target_indices):
        """Lazy initialization of Model and Optimizer state."""
        if self.model is None:
            key = jr.PRNGKey(self.seed)
            self.model = self.build_model(key, n_features, n_countries, target_indices)
            
            self.optim = self._create_optimizer()
            
            # CRITICAL: Filter parameters before initializing optimizer state.
            # Passing static parts (like integers or activation funcs) to optax.init 
            # causes structure mismatch errors.
            params = eqx.filter(self.model, eqx.is_inexact_array)
            self.opt_state = self.optim.init(params)

    def train_and_predict(self, data: Dict, horizon: int, train: bool = True) -> Tuple[np.ndarray, Dict]:
        loader = JAXDataLoader(data, self.window_size, horizon, self.batch_size)
        
        n_features = loader.n_features
        n_targets = loader.n_targets
        countries = data['meta']['countries']
        n_countries = len(countries)
        
        feature_cols = data['meta']['features']
        target_cols = data['meta']['target_cols']
        target_indices_list = [feature_cols.index(c) for c in target_cols]
        target_indices_jax = jnp.array(target_indices_list, dtype=jnp.int32)

        # --- Training Loop ---
        if train or self.model is None:
            self._ensure_model_initialized(n_features, n_countries, target_indices_jax)
            
            if train and self.verbose:
                print(f"[{self.__class__.__name__}] Training (Epochs={self.epochs}, WD={self.weight_decay})...")

            if train:
                for epoch in range(self.epochs):
                    epoch_losses = []
                    for x_b, y_b, c_b in loader:
                        self.model, self.opt_state, l = make_step(
                            self.model, self.opt_state, x_b, y_b, c_b, 
                            self.optim, horizon, self.rho
                        )
                        epoch_losses.append(float(l))
                    
                    if self.verbose and (epoch+1) % max(1, int(self.epochs/10)) == 0:
                        print(f"   Epoch {epoch+1}: Loss={np.mean(epoch_losses):.5f}")

        # --- Inference ---
        preds_array = np.full((n_countries, horizon, n_targets), np.nan)
        
        # Prepare inference batch (last window for each country)
        X_test_list = []
        valid_country_indices = []
        indices_list = []

        for i, c in enumerate(countries):
            ts = data['train'].get(c)
            if ts is None or len(ts) < self.window_size: 
                continue
            X_test_list.append(ts[-self.window_size:])
            indices_list.append(i)
            valid_country_indices.append(i)

        if X_test_list:
            X_test_batch = jnp.array(np.stack(X_test_list), dtype=jnp.float32)
            C_test_batch = jnp.array(indices_list, dtype=jnp.int32)

            preds_batch = predict_batch(self.model, X_test_batch, C_test_batch, horizon)
            preds_array[valid_country_indices] = np.array(preds_batch)

        return preds_array, {}
    
    def train_with_monitoring(
        self, 
        train_data: Dict, 
        val_bundle: Tuple[np.ndarray, np.ndarray, np.ndarray], 
        horizon: int,
        eval_frequency: int = 10
    ) -> Dict:
        """
        Extended training loop with validation monitoring.
        Returns a history dictionary containing loss curves.
        """
        loader = JAXDataLoader(train_data, self.window_size, horizon, self.batch_size)
        
        n_features = loader.n_features
        feature_cols = train_data['meta']['features']
        target_cols = train_data['meta']['target_cols']
        target_indices_jax = jnp.array([feature_cols.index(c) for c in target_cols], dtype=jnp.int32)

        self._ensure_model_initialized(n_features, len(train_data['meta']['countries']), target_indices_jax)

        # Preload Validation Data
        X_val, Y_val, C_val = val_bundle
        X_val_jax, Y_val_jax, C_val_jax = jnp.array(X_val), jnp.array(Y_val), jnp.array(C_val)

        history = {'train_loss': [], 'val_loss': [], 'epochs': []}
        
        if self.verbose:
            print(f"[{self.__class__.__name__}] Monitoring Train Start...")
        
        for epoch in range(self.epochs):
            batch_losses = []
            for x_b, y_b, c_b in loader:
                self.model, self.opt_state, l = make_step(
                    self.model, self.opt_state, x_b, y_b, c_b, 
                    self.optim, horizon, self.rho
                )
                batch_losses.append(float(l))
            
            train_loss = np.mean(batch_losses)
            history['train_loss'].append(train_loss)
            
            if epoch % eval_frequency == 0 or epoch == self.epochs - 1:
                val_loss = evaluate_batch_loss(
                    self.model, X_val_jax, Y_val_jax, C_val_jax, horizon, self.rho
                )
                history['val_loss'].append(float(val_loss))
                history['epochs'].append(epoch)
                
                if self.verbose:
                    print(f"   Ep {epoch:04d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        return history