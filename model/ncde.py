import jax
import jax.numpy as jnp
import jax.random as jr
import diffrax
import equinox as eqx
from model.jax_model import BaseJAXEstimator

# ============================================================================
# 1. VECTOR FIELDS
# ============================================================================

class ControlField(eqx.Module):
    """
    Parameterizes the Controlled Vector Field $f_\theta(z(t))$.

    In a Neural CDE, the latent state $z(t)$ evolves according to:
    $$ dz(t) = f_\theta(z(t)) dX(t) $$
    
    This module computes the matrix-valued function $f_\theta(z(t)) \in \mathbb{R}^{d_z \times d_x}$, 
    which is then multiplied by the control signal derivative $\frac{dX}{dt}$.
    """
    
    mlp: eqx.nn.MLP
    z_dim: int
    x_dim: int
    
    def __init__(self, input_dim: int, hidden_size: int, z_dim: int, x_dim: int, key):
        """
        Args:
            input_dim: Dimension of concatenated input (z_dim + n_countries).
            hidden_size: Width of the hidden layers.
            z_dim: Dimension of the latent state.
            x_dim: Dimension of the control signal (features).
        """
        self.z_dim = z_dim
        self.x_dim = x_dim
        
        self.mlp = eqx.nn.MLP(
            in_size=input_dim,
            out_size=z_dim * x_dim, # Flattens the output matrix
            width_size=hidden_size,
            depth=1,
            activation=jax.nn.tanh, # Tanh is standard for CDEs to ensure boundedness
            key=key
        )
    
    def __call__(self, t, z, args):
        """
        Evaluates $f_\theta(z, c)$.

        Args:
            t: Time (scalar).
            z: Latent state vector $(d_z,)$.
            args: Tuple containing static metadata (OHE country code).

        Returns:
            Matrix of shape $(d_z, d_x)$ representing the local interaction.
        """
        ohe = args
        
        # Condition on static entity attributes
        z_context = jnp.concatenate([z, ohe])
        
        # Compute flat vector and reshape into operator matrix
        flat_matrix = self.mlp(z_context)
        return flat_matrix.reshape(self.z_dim, self.x_dim)


class DynamicsField(eqx.Module):
    """
    Parameterizes the Autonomous Vector Field $g_\phi(z(t))$.
    
    Used for the decoding/forecasting phase where no future control $X(t)$ is available.
    $$ \frac{dz(t)}{dt} = g_\phi(z(t)) $$
    """
    
    mlp: eqx.nn.MLP
    
    def __init__(self, input_dim: int, hidden_size: int, z_dim: int, key):
        self.mlp = eqx.nn.MLP(
            in_size=input_dim, 
            out_size=z_dim,
            width_size=hidden_size,
            depth=2,
            activation=jax.nn.tanh,
            key=key
        )
    
    def __call__(self, t, z, args):
        """
        Evaluates $\frac{dz}{dt}$.
        """
        ohe = args
        z_context = jnp.concatenate([z, ohe])
        return self.mlp(z_context)


# ============================================================================
# 2. MODEL ARCHITECTURE
# ============================================================================

class NCDEtoODE(eqx.Module):
    """
    Neural CDE Encoder-Decoder Architecture.

    This model treats time series processing as a continuous-time differential equation 
    driven by the data itself.

    **1. Encoding Phase (Neural CDE):**
    The history $x_{0:T}$ is interpolated into a continuous path $X(t)$. The latent state $z(t)$ 
    evolves driven by this path:
    $$ z(T) = z(0) + \int_{0}^{T} f_\theta(z(t)) dX(t) $$
    This allows the model to naturally handle irregular sampling or missing data.

    **2. Decoding Phase (Neural ODE):**
    From the terminal encoded state $z(T)$, the system evolves autonomously into the future:
    $$ z(T+h) = z(T) + \int_{T}^{T+h} g_\phi(z(\tau)) d\tau $$

    **3. Readout:**
    Projections are taken at specific future time points:
    $$ \hat{y}_{T+h} = W_{out} z(T+h) + b_{out} $$
    """
    
    cde_func: ControlField
    ode_func: DynamicsField
    readout: eqx.nn.Linear
    
    n_latent: int
    n_features: int
    n_countries: int
    target_indices: jnp.ndarray
    
    def __init__(self, n_features, n_countries, target_indices, n_latent, hidden_size, key):
        k_cde, k_ode, k_read = jr.split(key, 3)
        
        self.n_latent = n_latent
        self.n_features = n_features
        self.n_countries = n_countries
        self.target_indices = target_indices
        
        # === CDE ENCODER (Driven by Data) ===
        # Input to MLP: Latent State + Country Embedding
        cde_input_dim = n_latent + n_countries
        self.cde_func = ControlField(
            input_dim=cde_input_dim,
            hidden_size=hidden_size,
            z_dim=n_latent,
            x_dim=n_features,
            key=k_cde
        )
        
        # === ODE DECODER (Autonomous) ===
        ode_input_dim = n_latent + n_countries
        self.ode_func = DynamicsField(
            input_dim=ode_input_dim,
            hidden_size=hidden_size,
            z_dim=n_latent,
            key=k_ode
        )
        
        # === READOUT ===
        self.readout = eqx.nn.Linear(n_latent, n_features, key=k_read)
    
    def __call__(self, x_seq, country_idx, horizon):
        """
        Forward pass: Interpolate -> Encode (CDE) -> Decode (ODE) -> Project.
        
        Args:
            x_seq: (L, Features) Observation sequence.
            country_idx: Integer index for OHE.
            horizon: Forecasting horizon $H$.
        """
        L = x_seq.shape[0]
        
        # 1. Static Metadata
        ohe = jnp.zeros(self.n_countries).at[country_idx].set(1.0)
        
        # 2. Continuous Path Construction
        # We use Cubic Hermite Splines to ensure the path X(t) is differentiable (C1),
        # which is required for the CDE solver.
        ts = jnp.arange(L, dtype=jnp.float32)
        coeffs = diffrax.backward_hermite_coefficients(ts, x_seq)
        path = diffrax.CubicInterpolation(ts, coeffs)
        
        # 3. ENCODING: Solve CDE from t=0 to t=L-1
        # The ControlTerm handles the matrix-vector multiplication f(z) * dX/dt
        term_cde = diffrax.ControlTerm(self.cde_func, path).to_ode()
        solver_cde = diffrax.Tsit5()
        
        z0 = jnp.zeros((self.n_latent,))
        
        # We solve the CDE. Note: args=ohe passes the country code to the vector field.
        sol_cde = diffrax.diffeqsolve(
            terms=term_cde,
            solver=solver_cde,
            t0=0.0,
            t1=float(L - 1),
            dt0=0.1,
            y0=z0,
            args=ohe,
            stepsize_controller=diffrax.PIDController(rtol=1e-2, atol=1e-3),
            max_steps=4096 # Safety limit for stiff dynamics
        )
        
        z_encoded = sol_cde.ys[0] # Terminal state z(L-1)
        
        # 4. DECODING: Solve ODE from t=L-1 to t=L-1+Horizon
        term_ode = diffrax.ODETerm(self.ode_func)
        solver_ode = diffrax.Tsit5()
        
        # Define evaluation points for the forecast
        # We start forecasting immediately after the encoding window
        save_ts = jnp.arange(L, L + horizon, dtype=jnp.float32)
        
        sol_ode = diffrax.diffeqsolve(
            terms=term_ode,
            solver=solver_ode,
            t0=float(L - 1),
            t1=float(L - 1 + horizon),
            dt0=0.1,
            y0=z_encoded,
            args=ohe,
            saveat=diffrax.SaveAt(ts=save_ts),
            stepsize_controller=diffrax.PIDController(rtol=1e-2, atol=1e-3)
        )
        
        # 5. Readout
        # Project latent trajectory z(t) back to data space
        preds = jax.vmap(self.readout)(sol_ode.ys) # (Horizon, Features)
        
        # Return only the requested target variables
        return preds[:, self.target_indices]


class NCDEModel(BaseJAXEstimator):
    """
    Wrapper integrating the Neural CDE into the standard JAX training pipeline.
    """
    
    def build_model(self, key, n_features, n_countries, target_indices):
        return NCDEtoODE(
            n_features=n_features,
            n_countries=n_countries,
            target_indices=target_indices,
            n_latent=self.config.get('n_latent', 8),
            hidden_size=self.config.get('hidden_size', 32),
            key=key
        )
    
    def _forward(self, model, x_batch, c_idx, horizon):
        """Vectorized forward pass over the batch."""
        return jax.vmap(lambda x, c: model(x, c, horizon))(x_batch, c_idx)