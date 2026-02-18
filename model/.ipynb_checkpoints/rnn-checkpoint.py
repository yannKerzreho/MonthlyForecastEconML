import jax
import jax.numpy as jnp
import jax.random as jr
import jax.lax as lax
import equinox as eqx
from model.jax_model import BaseJAXEstimator 

class RNNAutoregFull(eqx.Module):
    """
    Autoregressive Recurrent Neural Network (RNN) for Multi-Step Forecasting.

    This model employs a standard Sequence-to-Sequence (Seq2Seq) logic without an 
    attention mechanism. It consists of two distinct temporal phases:

    1.  **Encoding Phase (Warmup):**
        The network processes the historical context $x_{1:T}$ to update its hidden state $h_t$.
        $$ h_t = \text{GRU}([x_t, c_i], h_{t-1}) \quad \forall t \in [1, T] $$
        Here, $c_i$ is the static entity embedding (One-Hot Encoded).

    2.  **Decoding Phase (Generative):**
        The network generates the forecast horizon $h \in [1, H]$ in a closed loop. 
        The prediction $\hat{x}_{T+h}$ is fed back as the input for step $T+h+1$.
        $$ \hat{x}_{T+h} = W_{out} h_{T+h} + b_{out} $$
        $$ h_{T+h+1} = \text{GRU}([\hat{x}_{T+h}, c_i], h_{T+h}) $$
    """
    
    rnn: eqx.nn.GRUCell
    readout: eqx.nn.Linear
    hidden_size: int
    n_countries: int
    target_indices: jnp.ndarray
    
    def __init__(self, n_features, n_countries, target_indices, hidden_size, key):
        k1, k2 = jr.split(key)
        
        # Input Dimension: D (Features) + K (Countries)
        input_size = n_features + n_countries
        
        self.rnn = eqx.nn.GRUCell(input_size, hidden_size, key=k1)
        
        # Output Dimension: D (All features needed for feedback loop)
        self.readout = eqx.nn.Linear(hidden_size, n_features, key=k2)
        
        self.hidden_size = hidden_size
        self.n_countries = n_countries
        self.target_indices = target_indices

    def __call__(self, x_history: jnp.ndarray, country_idx: int, horizon: int):
        """
        Args:
            x_history: (Window, D) Historical observation sequence.
            country_idx: Integer index for OHE.
            horizon: Forecasting steps H.
        """
        # 1. Static Feature Construction (OHE)
        # Construct OHE vector on-the-fly for batching compatibility
        ohe_vec = jnp.zeros(self.n_countries)
        ohe_vec = ohe_vec.at[country_idx].set(1.0)
        
        # 2. Encoding Phase (Warmup)
        # Condition the hidden state h on the entire history window
        h = jnp.zeros((self.hidden_size,))
        
        # Scan over history to reach final state h_T
        def encoder_step(carrier, x_t):
            x_in = jnp.concatenate([x_t, ohe_vec])
            h_next = self.rnn(x_in, carrier)
            return h_next, None

        h_final, _ = lax.scan(encoder_step, h, x_history)
        
        # Last observed data point used to seed generation
        x_last = x_history[-1] 

        # 3. Decoding Phase (Autoregressive Generation)
        def decoder_step(carrier, _):
            h_prev, x_prev = carrier # x_prev is \hat{x}_{t-1}
            
            # Construct input: [Predicted Feature | Static OHE]
            gru_input = jnp.concatenate([x_prev, ohe_vec])
            
            # Update State
            h_next = self.rnn(gru_input, h_prev)
            
            # Predict Full Feature Vector (D,)
            x_next_pred = self.readout(h_next)
            
            # Return new state/input tuple, and the prediction for storage
            return (h_next, x_next_pred), x_next_pred

        # Run the loop H times
        _, preds_full = lax.scan(decoder_step, (h_final, x_last), None, length=horizon)
        
        # 4. Filter Targets
        # Select only the columns corresponding to target variables
        return preds_full[:, self.target_indices]


class RNNModel(BaseJAXEstimator):
    """
    Wrapper integrating the Autoregressive RNN into the JAX training pipeline.
    """
    
    def build_model(self, key, n_features, n_countries, target_indices):
        return RNNAutoregFull(
            n_features=n_features,
            n_countries=n_countries,
            target_indices=target_indices,
            hidden_size=self.config.get('hidden_size', 64),
            key=key
        )
    
    def _forward(self, model, x_batch, c_idx, horizon):
        # Vectorize inference over the batch dimension
        return jax.vmap(lambda x, c: model(x, c, horizon))(x_batch, c_idx)