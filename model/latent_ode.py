import jax
import jax.numpy as jnp
import jax.random as jr
import diffrax
import equinox as eqx
from typing import List, Any
from model.jax_model import BaseJAXEstimator

class NeuralField(eqx.Module):
    """
    Parameterizes the vector field (dynamics) of the ODE.
    
    Mathematically, this represents the function $f_\theta$ in:
    $$ \frac{dh(t)}{dt} = f_\theta(h(t), t) $$
    
    This implementation defines an autonomous system (time-invariant dynamics),
    meaning the output depends only on the state $y$, not explicit time $t$.
    """
    layers: List[eqx.Module]

    def __init__(self, n_latent: int, width: int, depth: int, key):
        keys = jr.split(key, depth + 1)
        
        # Input Layer
        self.layers = [eqx.nn.Linear(n_latent, width, key=keys[0])]
        
        # Hidden Layers
        for i in range(depth - 1):
            self.layers.append(eqx.nn.Linear(width, width, key=keys[i + 1]))
        
        # Output Layer (project back to latent space dim)
        self.layers.append(eqx.nn.Linear(width, n_latent, key=keys[-1]))

    def __call__(self, t, y, args):
        """
        Forward pass of the vector field.
        Args:
            t: Time (scalar), required by Diffrax but ignored here (autonomous).
            y: Current state vector $h(t)$.
            args: Auxiliary arguments (unused).
        """
        x = y
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply ELU activation between layers, but not after the last one
            if i < len(self.layers) - 1:
                x = jax.nn.elu(x)
        return x


class LatentNODE(eqx.Module):
    """
    Latent Neural Ordinary Differential Equation (Latent ODE) Model.

    This architecture treats the time series forecasting problem as an initial 
    value problem in a latent space.

    **Architecture:**
    1.  **Encoder (GRU):** Compresses the input history $x_{1:T}$ and static metadata $c$ 
        into a latent initial state $h(0)$.
        $$ h(0) = \text{Encoder}(x_{1:T}, c) $$
    
    2.  **Processor (Neural ODE):** Evolves the state forward in continuous time using a numeric solver.
        $$ h(t) = h(0) + \int_{0}^{t} f_\theta(h(\tau)) d\tau $$
    
    3.  **Decoder (Linear):** Projects the evolved latent state $h(t)$ to the target space.
        $$ \hat{y}_{T+t} = W_{out} h(t) + b_{out} $$
    """
    
    encoder: eqx.nn.GRUCell
    ode_func: NeuralField
    decoder: eqx.nn.Linear
    n_latent: int
    n_countries: int
    
    def __init__(
        self, 
        n_features: int, 
        n_targets: int, 
        n_countries: int, 
        n_latent: int, 
        width: int, 
        depth: int, 
        key
    ):
        k1, k2, k3 = jr.split(key, 3)
        
        # Encoder Input: Features + Country One-Hot Encoding
        total_input_dim = n_features + n_countries
        
        self.encoder = eqx.nn.GRUCell(total_input_dim, n_latent, key=k1)
        self.ode_func = NeuralField(n_latent, width, depth, key=k2)
        self.decoder = eqx.nn.Linear(n_latent, n_targets, key=k3)
        
        self.n_latent = n_latent
        self.n_countries = n_countries

    def __call__(self, x_seq: jnp.ndarray, country_idx: int, horizon: int):
        """
        Args:
            x_seq: Input sequence of shape (Window, Features).
            country_idx: Integer index of the entity/country.
            horizon: Number of steps to forecast.
        """
        # 1. Static Feature Preparation (OHE)
        # We perform OHE on-the-fly to handle batching cleanly in JAX
        ohe_vec = jnp.zeros(self.n_countries)
        ohe_vec = ohe_vec.at[country_idx].set(1.0)
        
        # 2. Encoding (Recurrent Pass)
        # Initializes hidden state and consumes the sequence
        h = jnp.zeros((self.n_latent,))
        
        def scan_fn(carry, x_t):
            # Concatenate dynamic features with static country encoding
            x_in = jnp.concatenate([x_t, ohe_vec])
            h_next = self.encoder(x_in, carry)
            return h_next, None

        # scan is more efficient than a python loop for RNNs in JAX
        h_final, _ = jax.lax.scan(scan_fn, h, x_seq)
        
        # 3. ODE Integration (Dynamics)
        # We solve from t=0 to t=horizon, saving at integer steps
        save_times = jnp.arange(1, horizon + 1, dtype=jnp.float32)
        
        sol = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(self.ode_func),
            solver=diffrax.Tsit5(), # Runge-Kutta 5(4)
            t0=0.0,
            t1=float(horizon),
            dt0=0.1, # Initial step size guess
            y0=h_final,
            saveat=diffrax.SaveAt(ts=save_times),
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-3)
            # Note: 1e-3 is usually sufficient for ML; 1e-1 might be too loose
        )
        
        # 4. Decoding
        # Map the trajectory h(1)...h(H) to predictions
        # sol.ys has shape (Horizon, Latent)
        return jax.vmap(self.decoder)(sol.ys)


class LatentNODEModel(BaseJAXEstimator):
    """
    Wrapper integrating LatentNODE into the training pipeline.
    """
    
    def build_model(self, key, n_features, n_countries, target_indices):
        n_targets = target_indices.shape[0]
        
        return LatentNODE(
            n_features=n_features,
            n_targets=n_targets,
            n_countries=n_countries,
            n_latent=self.config.get('n_latent', 32),
            width=self.config.get('hidden_size', 64),
            depth=self.config.get('depth', 2),
            key=key
        )
    
    def _forward(self, model, x_batch, c_idx, horizon):
        # BaseJAXEstimator expects this signature to handle vmap internally
        # or we define vmap here. 
        # Here we vmap the model call over the batch dimension.
        return jax.vmap(lambda x, c: model(x, c, horizon))(x_batch, c_idx)