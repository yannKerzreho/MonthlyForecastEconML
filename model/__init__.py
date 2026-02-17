from model.hyperopt_lagged_model import HyperoptLaggedModel
from model.lagged_model import LaggedModel
from model.minnesota_scaler import MinnesotaLagScaler
from model.esn import ESNModel
from model.dfm import DFMModel
from model.fvar import FVAR
from model.latent_ode import LatentNODEModel
from model.rnn import RNNModel
from model.ncde import NCDEModel

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import copy


def make_builder(model_cls, base_config, hook=None, **fixed_overrides):
    def builder(seed, config=None):
        run_config = copy.deepcopy(base_config)
        run_config.update(fixed_overrides)
        if config:
            run_config.update(config)

        run_config["seed"] = seed

        if hook:
            run_config = hook(run_config, seed)

        return model_cls(run_config)

    return builder


# VAR 
ridge_param = np.exp([(i-15)*0.5 for i in range(30)]).tolist()

config_var = {
        'lags': 2,
        'structure': 'panel',
        'strategy': 'direct',
        'trend': 'c',
        'base_estimator': LinearRegression(fit_intercept=True),
    }
def build_var(seed, config=config_var):
    return LaggedModel(config)

# RIDGE
config_ridge = {
        'lags': 12,
        'structure': 'panel',
        'strategy': 'direct',
        'trend': 'c',
        'base_estimator': Ridge(fit_intercept=True),
        'alpha_per_target': 'tow-stage',
        'param_grid': {
            'alpha': ridge_param
        },
        'cv_ratio': 0.2,
        'search_type': 'grid'
    }

def build_ridge_cv(seed, config=config_ridge):
    return HyperoptLaggedModel(config)

# RIDGE + MINNESOTA PRIOR
config_minnesota_ridge = {
        'lags': 12,
        'structure': 'panel',
        'strategy': 'direct',
        'trend': 'c',
                
        'param_grid': {
            'ridge__alpha': ridge_param,
            'minnesota__power': [0.0, 1.0, 2.0]
        },
        
        'cv_ratio': 0.2,
        'cv_splits': 5
    }

def build_minnesota_ridge(seed, config=config_minnesota_ridge):
    n_lags = config.get('lags', 8)
    pipeline = Pipeline([
        ('minnesota', MinnesotaLagScaler(n_lags=n_lags)),
        ('ridge', Ridge(fit_intercept=True))
    ])
    config['base_estimator'] = pipeline
    return HyperoptLaggedModel(config)

# RANDOM FOREST
config_forest = {
    'lags': 12,
    'strategy': 'direct',
    'trend': 'c',
    'max_depth': 5,
    'n_estimators': 100,
    }

def build_rf_pooled(seed, config=None):
    if config is None: 
        config = config_forest.copy()
    else: 
        config = config.copy()
    max_depth = config.pop('max_depth', 5)
    n_estimators = config.pop('n_estimators', 100)
    config['structure'] = 'pooled'
    config['ohe'] = True
    config['base_estimator'] = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, n_jobs=-1, random_state=seed)
    return LaggedModel(config)

def build_rf_panel(seed, config=None):
    if config is None: 
        config = config_forest.copy()
    else: 
        config = config.copy()
    max_depth = config.pop('max_depth', 5)
    n_estimators = config.pop('n_estimators', 100)
    config['structure'] = 'panel'
    config['ohe'] = False
    config['base_estimator'] = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, n_jobs=-1, random_state=seed)
    return LaggedModel(config)


# FACTRO VAR
config_fvar = {
    'n_factors': 5,
    'lags': 12,
    'structure': 'panel',
    'verbose': False,
}

def build_fvar(seed, config=None):
    if config is None: 
        config = config_fvar.copy()
    else: 
        config = config.copy()
    
    config["structure"] = "panel"
    config["seed"] = seed
    return FVAR(config)


# ECHO STATE NETWORK
config_esn = {
    'ridge_alphas': ridge_param,
    'units': 200,
    'structure': 'panel',
    'ohe': False,
    'lr': 0.6,
}

def build_esn_panel(seed, config=None):
    if config is None: 
        config = config_esn.copy()
    else: 
        config = config.copy()
    config['ridge_alphas'] = config.get('ridge_alphas', ridge_param)
    config['structure'] = 'panel'
    config['ohe'] = False
    config['seed'] = seed
    return ESNModel(config)

def build_esn_pooled(seed, config=None):
    if config is None: 
        config = config_esn.copy()
    else: 
        config = config.copy()
    config['ridge_alphas'] = config.get('ridge_alphas', ridge_param)
    config['structure'] = 'pooled'
    config['ohe'] = True
    config['seed'] = seed
    return ESNModel(config)

# DFM
config_dfm = {
        'k_factors': 1,
        'factor_order': 2,
        'error_order': 1,
        'structure': 'panel',
        'verbose': False,
    }

def build_dfm(seed, config=None):
    if config is None: 
        config = config_dfm.copy()
    else: 
        config = config.copy()
    config["structure"] = "panel"
    config["seed"] = seed
    return DFMModel(config)


# RNN
config_rnn = {
    'hidden_size': 24,
    'rho': 1,
    'window_size': 12,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'epochs': 200,
    'batch_size': 64,
    'verbose': False
}

def build_rnn(seed, config=None):
    if config is None: 
        config = config_rnn.copy()
    else: 
        config = config.copy()
    config["structure"] = "panel"
    config["seed"] = seed
    return RNNModel(config)


# LATENT ODE
config_node = {
    'n_latent': 8,
    'hidden_size': 24,
    'depth': 2,
    'rho': 1,                
    'window_size': 12,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'epochs': 200,
    'batch_size': 64,
    'verbose': False
}

def build_node(seed, config=None):
    if config is None: 
        config = config_node.copy()
    else: 
        config = config.copy()
    config["structure"] = "panel"
    config["seed"] = seed
    return LatentNODEModel(config)

# NCDE
config_ncde = {
    'n_latent': 8,
    'hidden_size': 24,
    'depth': 2,
    'rho': 1,
    'window_size': 12,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'epochs': 200,
    'batch_size': 64,
    'verbose': False
}

def build_ncde(seed, config=None):
    if config is None: 
        config = config_node.copy()
    else: 
        config = config.copy()
    config["structure"] = "panel"
    config["seed"] = seed
    return NCDEModel(config)