# Macroeconomic Forecasting Benchmark: Deep Learning and Econometrics

This repository contains the source code for a research project evaluating the efficacy of deep learning methods (Neural ODEs and RNN) against standard econometric benchmarks for macroeconomic forecasting.

The project addresses two distinct multivariate forecasting scenarios: the US data (FRED-MD) and a custom panel based on international data (OECD MEI).

## Project Structure

* **`main_fred.ipynb`**: Experiments on US data (FRED-MD) with large number of varaibles.
* **`main_mei.ipynb`**: Experiments on OECD data. Focuses on panel structures and cross-country heterogeneity.
* **`model/`**: Model implementations (Latent Neural ODEs, Neural CDEs, RNNs, DFM, VAR, Ridge Regression, Random Forest).
* **`evaluator.py`**: The evaluation engine handling Walk-Forward Validation (backtesting) and statistical hypothesis testing (Diebold-Mariano, Friedman).
* **`dataloader.py`**: Data management pipeline ensuring strict temporal splitting to prevent look-ahead bias.

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
