import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from scipy import stats
from typing import Optional, Literal, Dict, Tuple, List, Callable, Union
from itertools import combinations

from dataloader import MacroDataLoader

_QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
_METRIC_FUNCS = {
    'rmse': lambda p, t: (p - t) ** 2,  # Root applied after temporal aggregation
    'mae':  lambda p, t: np.abs(p - t),
}


class UniFreqPanelMacroEvaluator:
    """
    Orchestrator for Panel Macroeconomic Forecasting Evaluation.

    Handles the end-to-end backtesting pipeline:
        1. Walk-Forward Validation  – rolling training window simulation.
        2. Error Aggregation        – RMSE/MAE across Horizons, Targets, Entities.
        3. Statistical Inference    – Friedman + Diebold-Mariano tests.

    Statistical Tests:
        - Diebold-Mariano (1995): Equal predictive accuracy between two models,
          corrected for autocorrelation via Newey-West estimator.
        - Friedman (1937): Non-parametric k-model comparison across time blocks.
    """

    def __init__(
        self,
        dataloader: MacroDataLoader,
        models_builders: Dict[str, Callable[[int], any]],
        horizon: int,
        seeds: Optional[Union[int, List[int], List[List[int]]]] = None,
        verbose: bool = True,
        metric: Literal['rmse', 'mae'] = 'rmse',
    ):
        """
        Args:
            dataloader:       MacroDataLoader instance.
            models_builders:  {model_name: builder_func(seed) -> model}.
            horizon:          Forecasting horizon H.
            seeds:            int, List[int], or List[List[int]] (one list per model).
            verbose:          Logging flag.
            metric:           'rmse' or 'mae'.
        """
        if metric not in _METRIC_FUNCS:
            raise ValueError(f"metric must be one of {list(_METRIC_FUNCS)}, got '{metric}'")

        self.dataloader      = dataloader
        self.models_builders = models_builders
        self.horizon         = horizon
        self.verbose         = verbose
        self.metric          = metric
        self.quantiles       = _QUANTILES
        self.compute_error   = _METRIC_FUNCS[metric]
        self.seed_map        = self._normalize_seeds(seeds, list(models_builders.keys()))


    @staticmethod
    def _normalize_seeds(
        seeds: Optional[Union[int, List[int], List[List[int]]]],
        model_names: List[str],
    ) -> Dict[str, List[int]]:
        """
        Normalizes seed input into a {model_name: [seed, ...]} mapping.

        Accepted formats:
            None        -> all models use [42]
            42          -> all models use [42]
            [42, 7]     -> all models share the same seed list
            [[42], [7]] -> one seed list per model (must match model count)
        """
        if seeds is None:
            return {name: [42] for name in model_names}
        if isinstance(seeds, int):
            return {name: [seeds] for name in model_names}
        if isinstance(seeds[0], list):
            if len(seeds) != len(model_names):
                raise ValueError(
                    f"Number of seed lists ({len(seeds)}) != "
                    f"number of models ({len(model_names)})"
                )
            return dict(zip(model_names, seeds))
        # Flat list shared across all models
        return {name: seeds for name in model_names}


    def run_backtesting(
        self,
        end_train_date_list: List[Union[str, pd.Timestamp]],
        num_exp_by_fit: int = 1,
    ) -> Dict[str, Dict]:
        """
        Executes walk-forward backtesting for all models.

        Args:
            end_train_date_list: Chronological list of training cutoff dates.
            num_exp_by_fit:      Retrain every N steps (1 = every step).

        Returns:
            {model_name: {
                'stats':      aggregated statistics dict,
                'raw_errors': np.ndarray (n_seeds, n_dates, horizon, n_targets)
            }}
        """
        results = {}
        for name, builder in self.models_builders.items():
            if self.verbose:
                print(f"\n[Backtest] Model: {name}")

            stats_data, raw_errors = self._model_backtesting(
                builder, end_train_date_list, num_exp_by_fit, self.seed_map[name]
            )
            results[name] = {'stats': stats_data, 'raw_errors': raw_errors}

        return results

    def _model_backtesting(
        self,
        builder: Callable,
        dates: List,
        step_freq: int,
        seeds: List[int],
    ) -> Tuple[Dict, np.ndarray]:
        """Loops over seeds and dates, returns (stats, raw_errors)."""
        all_seeds_errors = []
        for seed in seeds:
            model = builder(seed)
            seed_errors = [
                self._step_model_backtesting(model, date, train=(i % step_freq == 0))
                for i, date in enumerate(dates)
            ]
            all_seeds_errors.append(seed_errors)

        raw_errors = np.array(all_seeds_errors)  # (n_seeds, n_dates, horizon, n_targets)
        return self._compute_stats(raw_errors), raw_errors

    def _step_model_backtesting(
        self,
        model,
        end_train_date: Union[str, pd.Timestamp],
        train: bool = True,
    ) -> np.ndarray:
        """
        Single walk-forward step.

        Returns:
            errors averaged over countries: (horizon, n_targets)
        """
        data = self.dataloader.get_request(end_train_date=end_train_date, horizon=self.horizon)

        y_pred, _ = model.train_and_predict(data, self.horizon, train)
        y_test    = data['test']  # (n_countries, horizon, n_targets)

        if y_pred.ndim == 2:
            y_pred = np.expand_dims(y_pred, axis=-1)

        assert y_pred.shape[0] == y_test.shape[0], "n_countries mismatch"
        assert y_pred.shape[1] == y_test.shape[1], "horizon mismatch"

        # (n_countries, horizon, n_targets) -> (horizon, n_targets)
        return np.nanmean(self.compute_error(y_pred, y_test), axis=0)


    def _compute_stats(self, res: np.ndarray) -> Dict:
        """
        Aggregates raw error tensor into statistical summaries.

        Input:  (n_seeds, n_dates, horizon, n_targets)
        Output: {
            'mean_mean':  (horizon, n_targets),
            'mean_std':   (horizon, n_targets),
            'mean_q{q}':  (horizon, n_targets),   for q in self.quantiles
            'std_q{q}':   (horizon, n_targets),   for q in self.quantiles
        }
        """
        multi_seed = res.shape[0] > 1

        # Temporal aggregation (over Dates)
        qs   = np.nanquantile(res, self.quantiles, axis=1)  # (n_q, n_seeds, H, T)
        mean = np.nanmean(res, axis=1)                       # (n_seeds, H, T)

        if self.metric == 'rmse':
            qs   = np.sqrt(qs)
            mean = np.sqrt(mean)

        # Seed aggregation
        out = {
            'mean_mean': np.nanmean(mean, axis=0),
            'mean_std':  np.nanstd(mean, axis=0) if multi_seed else np.zeros_like(mean[0]),
        }
        for i, q in enumerate(self.quantiles):
            out[f'mean_q{q}'] = np.nanmean(qs[i], axis=0)
            out[f'std_q{q}']  = np.nanstd(qs[i],  axis=0) if multi_seed else np.zeros_like(qs[i, 0])

        return out


    def compare_models(
        self,
        results: Dict[str, Dict],
        target_idx: Optional[int] = 0,
        metric_func: Optional[Callable] = None,
    ) -> pd.DataFrame:
        """
        Builds a comparison DataFrame with per-horizon columns and a global ranking.

        Args:
            results:     Output of run_backtesting.
            target_idx:  Target index, or None to sum all targets.
            metric_func: Optional callable(stats) -> scalar for custom ranking.

        Returns:
            DataFrame with columns [Model, H1, H1_std, ..., Mean, Ranking].
        """
        rows = []
        for name, data in results.items():
            s = data['stats']
            mean_mean = self._select_target(s['mean_mean'], target_idx)
            mean_std  = self._select_target(s['mean_std'],  target_idx)

            row = {'Model': name}
            for h in range(self.horizon):
                row[f'H{h+1}']     = mean_mean[h]
                row[f'H{h+1}_std'] = mean_std[h]
            row['Mean'] = metric_func(s) if metric_func else float(np.nanmean(mean_mean))
            rows.append(row)

        df = pd.DataFrame(rows).sort_values('Mean').reset_index(drop=True)
        df['Ranking'] = np.arange(1, len(df) + 1)
        return df

    def print_comparison_table(
        self,
        results: Dict[str, Dict],
        target_idx: Optional[int] = 0,
        decimals: int = 4,
    ) -> None:
        """Prints a formatted comparison table."""
        df    = self.compare_models(results, target_idx=target_idx)
        label = "All Targets (sum)" if target_idx is None else f"Target {target_idx}"

        print(f"\n{'='*100}")
        print(f"MODEL COMPARISON — {self.metric.upper()} @ {label}")
        print('='*100)
        print(df.to_string(index=False, float_format=lambda x: f"{x:.{decimals}f}"))
        print('='*100)
        print(f"Lower is better (metric: {self.metric})")


    def _select_target(self, arr: np.ndarray, target_idx: Optional[int]) -> np.ndarray:
        """Selects one target slice or sums all targets along the last axis."""
        return np.sum(arr, axis=-1) if target_idx is None else arr[..., target_idx]

    def _plot_trace(
        self,
        ax,
        stats_data: Dict,
        target_idx: Optional[int],
        color: str,
        label: str,
        show_mean: bool = True,
    ) -> None:
        """
        Draws a single model trace:
            - Outer fill : [Q_min, Q_max] band
            - Dashed line: Median ± seed std
            - Solid line : Mean  ± seed std  (if show_mean)
        """
        x     = np.arange(self.horizon)
        qs    = sorted(self.quantiles)
        q_min, q_max = qs[0], qs[-1]
        q_med = 0.5 if 0.5 in qs else qs[len(qs) // 2]

        def get(key):
            return self._select_target(stats_data[key], target_idx)

        # Quantile envelope
        ax.fill_between(
            x, get(f'mean_q{q_min}'), get(f'mean_q{q_max}'),
            color=color, alpha=0.10, label=f"{label} (Q{q_min}–Q{q_max})"
        )
        # Median
        med, med_std = get(f'mean_q{q_med}'), get(f'std_q{q_med}')
        ax.plot(x, med, color=color, ls='--', lw=2, label=f"{label} (Median)")
        ax.fill_between(x, med - med_std, med + med_std, color=color, alpha=0.15)

        # Mean (optional)
        if show_mean:
            m, m_std = get('mean_mean'), get('mean_std')
            ax.plot(x, m, color=color, lw=2.5, label=f"{label} (Mean)")
            ax.fill_between(x, m - m_std, m + m_std, color=color, alpha=0.25)


    def plot_compare_models(
        self,
        results: Dict[str, Dict],
        target_idx: Optional[int] = 0,
        figsize: Tuple[int, int] = (14, 7),
        show_mean: bool = True,
        title: Optional[str] = None,
    ) -> None:
        """
        Comparative plot with Mean, Median, and Quantile bands per model.

        Args:
            results:    Output of run_backtesting.
            target_idx: Target index, or None to sum all targets.
            figsize:    Figure size.
            show_mean:  Whether to overlay the mean curve.
            title:      Custom title.
        """
        fig, ax = plt.subplots(figsize=figsize)
        colors  = plt.cm.tab10(np.linspace(0, 1, len(results)))

        for (name, data), color in zip(results.items(), colors):
            if 'stats' not in data:
                print(f"⚠️  Model '{name}' has no stats — skipping.")
                continue
            self._plot_trace(ax, data['stats'], target_idx, color, name, show_mean)

        label = "All Targets (sum)" if target_idx is None else f"Target {target_idx}"
        ax.set_title(title or f"Model Comparison — {self.metric.upper()} @ {label}",
                     fontsize=14, fontweight='bold')
        ax.set_xlabel("Horizon (Steps)", fontsize=12)
        ax.set_ylabel(self.metric.upper(), fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, which='both', ls='-', alpha=0.2)
        plt.tight_layout()
        plt.show()


    def diebold_mariano_test(
        self,
        errors_model1: np.ndarray,
        errors_model2: np.ndarray,
        h: int = 1,
        alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided',
    ) -> Tuple[float, float]:
        """
        Diebold-Mariano test for equal predictive accuracy.

        Args:
            errors_model1: Loss series for model 1 — shape (n_dates,).
            errors_model2: Loss series for model 2 — shape (n_dates,).
            h:             Forecast horizon (for Newey-West lag correction).
            alternative:   'two-sided' | 'less' (M1 better) | 'greater' (M2 better).

        Returns:
            (dm_statistic, p_value)
        """
        d             = errors_model1 - errors_model2
        d_mean, n     = np.nanmean(d), len(d)
        gamma0        = np.var(d, ddof=1)
        gamma_sum     = sum(
            np.sum((d[k:] - d_mean) * (d[:-k] - d_mean)) / n
            for k in range(1, h) if k < n
        )
        var_d = (gamma0 + 2 * gamma_sum) / n

        if var_d <= 1e-12:
            warnings.warn("Near-zero variance in DM test — returning trivial result.")
            return 0.0, 1.0

        dm_stat = d_mean / np.sqrt(var_d)
        p_value = {
            'two-sided': 2 * (1 - stats.norm.cdf(abs(dm_stat))),
            'less':      stats.norm.cdf(dm_stat),
            'greater':   1 - stats.norm.cdf(dm_stat),
        }[alternative]

        return float(dm_stat), float(p_value)


    def pairwise_dm_tests(
        self,
        results: Dict[str, Dict],
        target_idx: Optional[int] = 0,
        horizon_idx: Optional[int] = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Runs DM tests for all model pairs.

        Args:
            results:     Output of run_backtesting.
            target_idx:  Target index, or None to sum all targets.
            horizon_idx: Specific horizon index, or None to average all horizons.
            alpha:       Significance level.

        Returns:
            Sorted DataFrame with DM statistics, p-values, and winners.
        """
        model_names = list(results.keys())
        rows        = []

        for name1, name2 in combinations(model_names, 2):
            e1 = self._prepare_error_series(results[name1]['raw_errors'], target_idx, horizon_idx)
            e2 = self._prepare_error_series(results[name2]['raw_errors'], target_idx, horizon_idx)

            h_for_dm = (horizon_idx + 1) if horizon_idx is not None else self.horizon
            h_label  = f"H{horizon_idx + 1}" if horizon_idx is not None else "All H"
            dm_stat, p = self.diebold_mariano_test(e1, e2, h=h_for_dm)

            avg  = lambda x: np.nanmean(x) if self.metric == 'mae' else np.sqrt(np.nanmean(x))
            m1, m2 = avg(e1), avg(e2)
            sig    = p < alpha

            rows.append({
                'Model 1':                   name1,
                'Model 2':                   name2,
                'Horizon':                   h_label,
                f'{self.metric.upper()} M1': m1,
                f'{self.metric.upper()} M2': m2,
                'DM Statistic':              dm_stat,
                'p-value':                   p,
                'Significant':               "Yes" if sig else "No",
                'Winner':                    (name1 if m1 < m2 else name2) if sig else "Tie",
            })

        return pd.DataFrame(rows).sort_values('p-value')


    def friedman_test(
        self,
        results: Dict[str, Dict],
        target_idx: Optional[int] = 0,
        horizon_idx: Optional[int] = None,
    ) -> Tuple[float, float, pd.DataFrame]:
        """
        Friedman non-parametric test across multiple models.

        Returns:
            (statistic, p_value, ranks_df)
        """
        model_names  = list(results.keys())
        error_series = [
            self._prepare_error_series(
                results[n]['raw_errors'], target_idx, horizon_idx, apply_root=True
            )
            for n in model_names
        ]
        errors_mat         = np.column_stack(error_series)  # (n_dates, n_models)
        statistic, p_value = stats.friedmanchisquare(*error_series)
        mean_ranks         = np.nanmean(
            np.apply_along_axis(stats.rankdata, 1, errors_mat), axis=0
        )
        ranks_df = pd.DataFrame({
            'Model':                       model_names,
            'Mean Rank':                   mean_ranks,
            f'Mean {self.metric.upper()}': np.nanmean(errors_mat, axis=0),
        }).sort_values('Mean Rank')

        return statistic, p_value, ranks_df

    def print_significance_tests(
        self,
        results: Dict[str, Dict],
        target_idx: Optional[int] = 0,
        horizon_idx: Optional[int] = None,
        alpha: float = 0.05,
    ) -> None:
        """
        Prints a full significance test report: Friedman + pairwise DM.

        Args:
            results:     Output of run_backtesting.
            target_idx:  Target index, or None to sum all targets.
            horizon_idx: Specific horizon index, or None to average all horizons.
            alpha:       Significance level.
        """
        target_label  = "All Targets (sum)" if target_idx is None else f"Target {target_idx}"
        horizon_label = f"H{horizon_idx + 1}" if horizon_idx is not None else "Average over all horizons"

        print(f"\n{'='*100}")
        print("STATISTICAL SIGNIFICANCE TESTS")
        print(f"  Target  : {target_label}")
        print(f"  Horizon : {horizon_label}")
        print('='*100)

        # 1. Friedman Test
        print("\n1. FRIEDMAN TEST  (Non-parametric, multiple models)")
        print('-'*100)
        f_stat, p_val, ranks_df = self.friedman_test(results, target_idx, horizon_idx)
        sig_label = "✓ Significant" if p_val < alpha else "✗ Not significant"
        print(f"Statistic : {f_stat:.4f}")
        print(f"p-value   : {p_val:.4f}  →  {sig_label} (α={alpha})")
        print("\nMean Ranks (lower = better):")
        print(ranks_df.to_string(index=False))

        # 2. Pairwise DM Tests
        print("\n\n2. DIEBOLD-MARIANO PAIRWISE TESTS")
        print('-'*100)
        dm_df = self.pairwise_dm_tests(results, target_idx, horizon_idx, alpha)
        print(dm_df.to_string(index=False))

        print(f"\n{'='*100}")
        print("Interpretation:")
        print("  • Friedman    : Tests whether ANY models differ significantly.")
        print("  • DM tests    : Identifies which specific pairs differ.")
        print("  • DM stat < 0 : Model 1 better;  DM stat > 0 : Model 2 better.")
        if target_idx is None:
            print("  • Errors summed across all targets.")
        print('='*100)


    def _prepare_error_series(
        self,
        raw_errors: np.ndarray,
        target_idx: Optional[int],
        horizon_idx: Optional[int],
        apply_root: bool = False,
    ) -> np.ndarray:
        """
        Reduces raw_errors (n_seeds, n_dates, horizon, n_targets) to a 1-D
        time series (n_dates,) suitable for statistical tests.

        Steps:
            1. Select / sum targets    -> (..., horizon)
            2. Average over seeds      -> (n_dates, horizon)
            3. Select / avg horizons   -> (n_dates,)
            4. Optional sqrt for RMSE
        """
        e = self._select_target(raw_errors, target_idx)      # (seeds, dates, H)
        e = np.nanmean(e, axis=0)                             # (dates, H)
        e = e[:, horizon_idx] if horizon_idx is not None else np.nanmean(e, axis=1)
        if apply_root and self.metric == 'rmse':
            e = np.sqrt(e)
        return e