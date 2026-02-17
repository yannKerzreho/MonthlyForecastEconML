# Macroeconomic Forecasting Benchmark: Deep Learning and Econometrics

This repository contains the source code for a research project evaluating the efficacy of deep learning methods (Neural ODEs and RNN) against standard econometric benchmarks for macroeconomic forecasting.

The project addresses two distinct multivariate forecasting scenarios: the US data (FRED-MD) and a custom panel based on international data (OECD MEI).

## Project Structure

* **`main_fred.ipynb`**: Experiments on US data. Focuses on time series dynamics and handling "ragged edges" (mixed publication delays).
* **`main_mei.ipynb`**: Experiments on OECD data. Focuses on panel structures and cross-country heterogeneity.
* **`model/`**: Model implementations.
* **JAX/Diffrax**: Latent Neural ODEs, Neural CDEs, RNNs.
* **Scikit-Learn/Statsmodels**: Dynamic Factor Models (DFM), VAR, Ridge Regression, Random Forest.


* **`evaluator.py`**: The evaluation engine handling Walk-Forward Validation (backtesting) and statistical hypothesis testing (Diebold-Mariano, Friedman).
* **`dataloader.py`**: Data management pipeline ensuring strict temporal splitting to prevent look-ahead bias.

## Implemented Models

The framework compares state-of-the-art neural differential equations against robust linear baselines:

1. **Benchmarks**: Random Walk, Autoregressive (AR) models.
2. **Econometrics**: Vector Autoregression (VAR), Dynamic Factor Models (DFM).
3. **Machine Learning**: Ridge Regression and Random Forest (implemented with Minnesota Prior scaling).
4. **Deep Learning (JAX)**:
* **Latent ODE**: Continuous-time modeling of latent dynamics.
* **Neural CDE**: Neural Controlled Differential Equations designed to handle irregular time series.
* **ESN**: Echo State Networks (Reservoir Computing).



## Methodology

The evaluation framework is designed to mimic real-time forecasting conditions:

* **Walk-Forward Validation**: Models are iteratively re-trained on an expanding window basis.
* **Metrics**: RMSE and MAE are computed per horizon and per target.
* **Statistical Significance**:
* **Friedman Test**: Non-parametric test to detect differences in ranking across multiple models.
* **Diebold-Mariano Test (1995)**: Pairwise test for predictive accuracy, correcting for autocorrelation in forecast errors.



## Installation

This project requires Python 3.9+ and relies heavily on the JAX ecosystem.

```bash
pip install -r requirements.txt

```

## Usage Example

To run a comparative backtest on panel data:

```python
from dataloader import MacroDataLoader
from evaluator import UniFreqPanelMacroEvaluator
from model.lagged_model import LaggedModel
from sklearn.linear_model import Ridge

# 1. Load and process data
loader = MacroDataLoader(df, config={'target_col': 'GDP', 'date_col': 'Date', 'features': [...]})

# 2. Define model builders (seed -> model)
models = {
    'Ridge': lambda seed: LaggedModel({'base_estimator': Ridge(), 'lags': 4}),
    'Benchmark': lambda seed: LaggedModel({'base_estimator': Ridge(), 'lags': 1})
}

# 3. Initialize Evaluator
evaluator = UniFreqPanelMacroEvaluator(loader, models, horizon=12, metric='rmse')

# 4. Run Walk-Forward Validation
results = evaluator.run_backtesting(end_train_date_list=backtest_dates)

# 5. Statistical Analysis
evaluator.print_significance_tests(results)

```

# Note for latter project

- Create an orchestrator that only returns prediction series
- Create a class to manage significance tests, calculate errors, etc.
- Create a class for plots
- Request non-standardized variables from the model
