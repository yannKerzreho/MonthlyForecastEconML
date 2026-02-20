import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Callable, Optional, Tuple, Union, Any

class ContinuousSplitEvaluator:
    """
    Evaluator for Continuous Monitoring of Learning Curves across Temporal Splits.

    This class implements a **Sequential Block Cross-Validation** strategy specifically 
    designed to diagnose model stability and overfitting.

    **Methodology:**
    1.  **Temporal Partitioning:** The backtest period is divided into $N$ contiguous blocks (Splits).
    2.  **Fresh Training:** For *each* split $k$:
        - A new model is instantiated (reset weights).
        - Training Data: All history $t < \text{Start}(Split_k)$.
        - Validation Bundle: All $(X, Y)$ pairs occurring *within* $Split_k$.
    3.  **Monitoring:** The model is trained on the history while periodically evaluating 
        its loss on the "future" Validation Bundle.

    **Goal:** Unlike standard backtesting (which gives a single error metric), this approach 
    visualizes the *trajectory* of learning. It answers: 
    *"Did the model converge effectively in 2020? Did it overfit in 2022?"*
    """

    def __init__(
        self,
        dataloader,
        models_builders: Dict[str, Callable[[int], Any]],
        horizon: int,
        seeds: Optional[List[int]] = None,
        verbose: bool = True,
        eval_frequency: int = 10
    ):
        """
        Args:
            dataloader: Instance of `MacroDataLoader`.
            models_builders: Dictionary mapping model names to builder functions 
                             (signature: `seed -> model`).
            horizon: Forecasting horizon $H$.
            seeds: List of random seeds (typically use seeds[0] for deterministic monitoring).
            verbose: Enable logging.
            eval_frequency: Evaluate validation loss every $N$ training epochs.
        """
        self.dataloader = dataloader
        self.models_builders = models_builders
        self.horizon = horizon
        self.seeds = seeds if seeds is not None else [42]
        self.verbose = verbose
        self.eval_frequency = eval_frequency

    def _create_validation_bundle(
        self, 
        val_dates: pd.DatetimeIndex,
        window_size: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Pre-computes a array bundle containing all test cases in the split.
        This allows for efficient, single-pass validation during the training loop.

        Args:
            val_dates: List of dates comprising the split.
            window_size: Input sequence length required by the model.

        Returns:
            Tuple (X_batch, Y_batch, Country_Indices_batch) or None if empty.
        """
        X_list, Y_list, C_list = [], [], []
        
        # Iterate through every time step in the validation block
        for date in val_dates:
            # Request data state at this specific date
            # 'test' contains the realization of the future (Target)
            # 'train' contains the history up to date (Input)
            req = self.dataloader.get_request(end_train_date=date, horizon=self.horizon)
            
            countries = req['meta']['countries']
            test_data = req['test']  # Shape: (n_countries, horizon, n_targets)
            
            for i, country in enumerate(countries):
                # Retrieve History
                ts = req['train'].get(country)
                
                # Safety Checks
                if ts is None or len(ts) < window_size:
                    continue
                
                # Retrieve Target (Ground Truth)
                y_true = test_data[i]
                
                # Skip if NaNs present in target (missing future) or input (missing history)
                if np.isnan(y_true).any() or np.isnan(ts[-window_size:]).any():
                    continue
                
                # Append to lists
                X_list.append(ts[-window_size:]) 
                Y_list.append(y_true)
                C_list.append(i)
        
        if not X_list:
            return None
            
        # Stack into Float32 Tensors (Optimized for JAX/Torch)
        return (
            np.stack(X_list).astype(np.float32),
            np.stack(Y_list).astype(np.float32),
            np.array(C_list, dtype=np.int32)
        )

    def run(self, backtest_dates: pd.DatetimeIndex, n_splits: int = 4) -> Dict:
        """
        Executes the sequential evaluation protocol.

        Args:
            backtest_dates: The full range of dates to cover.
            n_splits: Number of temporal blocks.

        Returns:
            Dictionary containing loss history for each model and split.
        """
        # 1. Partition the timeline
        splits = np.array_split(backtest_dates, n_splits)
        results = {}

        for name, builder in self.models_builders.items():
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"MONITORING MODEL: {name}")
                print(f"{'='*60}")
            
            model_splits_history = []
            
            for i, split in enumerate(splits):
                if len(split) == 0: continue
                
                start_date = split[0]
                end_date = split[-1]
                
                if self.verbose:
                    print(f"\n>> Split {i+1}/{n_splits} : {start_date.date()} -> {end_date.date()} ({len(split)} obs)")
                
                # 2. Instantiate Fresh Model (Tabula Rasa)
                seed = self.seeds[0]
                model = builder(seed)
                
                # Check Interface Compliance
                if not hasattr(model, 'train_with_monitoring'):
                    print(f"Model {name} missing 'train_with_monitoring' method. Skipping.")
                    continue

                # 3. Data Preparation
                # A. Training Set: All history available BEFORE the split starts
                # Note: We use start_date as cutoff.
                train_req = self.dataloader.get_request(end_train_date=start_date, horizon=self.horizon)
                
                # B. Validation Bundle: Data points WITHIN the split range
                # Dynamically retrieve window_size from the model instance
                window_size = getattr(model, 'window_size', 12)
                val_bundle = self._create_validation_bundle(split, window_size)
                
                if val_bundle is None:
                    print("No validation data available for this split (insufficient history?).")
                    continue

                # 4. Training with Live Monitoring
                history = model.train_with_monitoring(
                    train_data=train_req,
                    val_bundle=val_bundle,
                    horizon=self.horizon,
                    eval_frequency=self.eval_frequency
                )
                
                model_splits_history.append({
                    'split_id': i,
                    'range': (start_date, end_date),
                    'history': history,
                    'n_val_samples': len(val_bundle[0])
                })
            
            results[name] = model_splits_history
            
        return results

    def plot_learning_curves(self, results: Dict, metric_name='MSE Loss', log_scale=True):
        """
        Visualizes Comparative Learning Curves.
        
        Plots Training (dashed) vs Validation (solid) loss over epochs for all models,
        separated by temporal split.
        """
        if not results:
            print("No results to plot.")
            return

        # Determine grid dimensions based on first model
        first_model_res = next(iter(results.values()))
        n_splits = len(first_model_res)
        
        # Setup Figure
        fig, axes = plt.subplots(1, n_splits, figsize=(6 * n_splits, 6), sharey=True)
        if n_splits == 1: axes = [axes]
        
        # Assign distinct colors to models
        # Uses tab10 colormap for clear distinction
        unique_models = list(results.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(unique_models))))
        model_colors = {name: col for name, col in zip(unique_models, colors)}

        # Iterate over Splits (Columns)
        for split_idx in range(n_splits):
            ax = axes[split_idx]
            
            # Iterate over Models (Curves)
            for model_name, splits_data in results.items():
                # Safety check for missing splits
                if split_idx >= len(splits_data): continue
                
                data = splits_data[split_idx]
                hist = data['history']
                dates = data['range']
                col = model_colors[model_name]
                
                # --- A. Train Loss (Light Dashed Line) ---
                train_loss = hist.get('train_loss', [])
                if train_loss:
                    epochs = range(len(train_loss))
                    ax.plot(epochs, train_loss, 
                            linestyle='--', alpha=0.4, color=col, linewidth=1.5,
                            label=f"{model_name} (Train)" if split_idx == 0 else "")
                
                # --- B. Validation Loss (Solid Marked Line) ---
                val_loss = hist.get('val_loss', [])
                val_epochs = hist.get('epochs', [])
                
                if val_loss and len(val_loss) == len(val_epochs):
                    ax.plot(val_epochs, val_loss, 
                            linestyle='-', alpha=1.0, color=col, linewidth=2.5,
                            marker='o', markersize=4,
                            label=f"{model_name} (Val)")

            # --- C. Formatting ---
            start_str = dates[0].date()
            end_str = dates[1].date()
            ax.set_title(f"Split {split_idx+1}\n{start_str} → {end_str}", fontsize=12, fontweight='bold')
            ax.set_xlabel("Epochs")
            
            if split_idx == 0: 
                ax.set_ylabel(metric_name)
            
            if log_scale: 
                ax.set_yscale('log')
            
            ax.grid(True, which='both', linestyle='--', alpha=0.3)
            
            # Add Legend only to the last plot to reduce clutter
            if split_idx == n_splits - 1:
                ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), 
                          fontsize=10, framealpha=0.9, shadow=True)

        plt.tight_layout()
        plt.show()
