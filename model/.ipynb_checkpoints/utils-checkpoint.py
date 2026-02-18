import numpy as np
import jax.numpy as jnp
from typing import Dict, Tuple, List, Iterator


def create_matrices(lags, data, look_ahead=1):
    """
    Crée X (Lags) et Y (Target déplacée de look_ahead).
    Si look_ahead=1, c'est du VAR classique.
    Si look_ahead=h, c'est pour la méthode 'Direct'.
    
    Args:
        data: array (T, K) avec T observations et K variables
        look_ahead: horizon de prédiction
        
    Returns:
        X: (T', p*K) matrice de features (lags)
        Y: (T', K) matrice cible
    """
    T, K = data.shape
    p = lags
    if T <= p + look_ahead - 1:
        return None, None
    
    # Y : cible à t + look_ahead
    # Indices : [p + look_ahead - 1 : T]
    Y = data[p + look_ahead - 1:, :]
    
    # X : lags construits avant la cible
    # Les lags commencent à l'indice 0, se terminent à p-1+look_ahead-1
    lag_list = []
    for l in range(p):
        # Lag l : indices [p-1-l : T-look_ahead-l]
        start_idx = p - 1 - l
        end_idx = T - look_ahead - l
        
        slice_data = data[start_idx:end_idx, :]
        lag_list.append(slice_data)
        
    X = np.hstack(lag_list)
    
    assert X.shape[0] == Y.shape[0], \
        f"Mismatch: X shape {X.shape[0]} != Y shape {Y.shape[0]}"
    
    return X, Y

def ohe(ts, index, num_countries):
    """
    Creat one hot encoded version of the ts.
    """
    if num_countries == 1:
        return ts
    else: 
        ohe_vector = np.zeros(num_countries)
        ohe_vector[index] = 1.0
        ohe_matrix = np.tile(ohe_vector, (ts.shape[0], 1))
        return np.hstack([ts, ohe_matrix])


class JAXDataLoader:
    """Prépare (X, Y, CountryIdx) SANS OHE - features brutes seulement."""
    
    def __init__(self, data: Dict, window_size: int, horizon: int, batch_size: int):
        self.X, self.Y = [], []
        self.country_indices = []  # <--- 1. NOUVELLE LISTE
        
        countries = data['meta']['countries']
        target_cols = data['meta']['target_cols']
        feature_cols = data['meta']['features']
        
        # Indices des colonnes cibles
        target_indices = [feature_cols.index(c) for c in target_cols]
        
        # Boucle sur les pays (i est l'index du pays)
        for i, c in enumerate(countries): 
            ts = data['train'].get(c)
            if ts is None or len(ts) <= window_size + horizon:
                continue
            
            # Découpage en fenêtres glissantes
            for t in range(len(ts) - window_size - horizon + 1):
                x_window = ts[t : t + window_size] 
                y_horizon = ts[t + window_size : t + window_size + horizon]
                
                self.X.append(x_window)
                # On ne garde que les targets pour Y
                self.Y.append(y_horizon[:, target_indices])
                
                # <--- 2. STOCKAGE DE L'INDEX PAYS
                # On associe l'index 'i' à cet échantillon précis
                self.country_indices.append(i) 
        
        self.X = jnp.array(self.X)  # (N, L, D)
        self.Y = jnp.array(self.Y)  # (N, H, n_targets)
        
        self.country_indices = jnp.array(self.country_indices, dtype=jnp.int32) # (N,)
        
        self.batch_size = batch_size
        self.n_features = self.X.shape[-1]
        self.n_targets = self.Y.shape[-1]
        self.n_countries = len(countries) # Utile pour l'initialisation du modèle
    
    def __iter__(self):
        indices = np.arange(len(self.X))
        np.random.shuffle(indices)
        for i in range(0, len(self.X), self.batch_size):
            idx = indices[i : i + self.batch_size]
            yield self.X[idx], self.Y[idx], self.country_indices[idx]