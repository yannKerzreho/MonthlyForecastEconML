import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Union, Optional

class MacroDataLoader:
    """
    Data handler for panel macroeconomic time series.
    
    This class manages the ingestion, alignment, and splitting of cross-sectional 
    time series data (Country $\times$ Time $\times$ Features). It ensures strict 
    temporal separation between training and testing sets to prevent data leakage.
    
    **Attributes:**
    - `raw_df` (pd.DataFrame): The complete historical dataset.
    - `countries` (List[str]): List of unique entities/countries in the panel.
    - `target_cols` (List[str]): Subset of features designated as prediction targets.
    """

    def __init__(self, df: pd.DataFrame, config: dict):
        """
        Args:
            df: Raw DataFrame containing columns [date_col, country_col, features...].
            config: Configuration dictionary with keys:
                - 'date_col': Name of the time index column.
                - 'country_col': Name of the entity identifier column.
                - 'features': List of all feature names.
                - 'target_col': Name(s) of the target variable(s).
        """
        # 1. Validation
        required_keys = ['date_col', 'features']
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Config missing required key: {key}")
        
        self.raw_df = df.copy()
        self.cfg = config
        self.country_col = self.cfg.get('country_col')
        self.date_col = self.cfg['date_col']
        
        # 2. Index Management
        if self.raw_df.index.name != self.date_col:
            if self.date_col in self.raw_df.columns:
                self.raw_df.set_index(self.date_col, inplace=True)
            else:
                raise ValueError(f"date_col '{self.date_col}' not found.")
        
        try:
            self.raw_df.index = pd.to_datetime(self.raw_df.index)
        except Exception as e:
            raise ValueError(f"Index conversion to datetime failed: {e}")
                
        # 3. Sorting (Crucial for time series operations)
        if self.country_col:
            if self.country_col not in self.raw_df.columns:
                raise ValueError(f"country_col '{self.country_col}' missing.")
            self.raw_df = self.raw_df.sort_values([self.country_col, self.date_col])
        else:
            self.raw_df = self.raw_df.sort_index()

        # 4. Target Definition
        target_cfg = self.cfg.get('target_col')
        if target_cfg is None:
            self.target_cols = self.cfg['features']
        elif isinstance(target_cfg, str):
            self.target_cols = [target_cfg]
        elif isinstance(target_cfg, list):
            self.target_cols = target_cfg
        else:
            raise ValueError("target_col must be str, list, or None")
        
        # Validate targets exist in features
        if not set(self.target_cols).issubset(set(self.cfg['features'])):
            raise ValueError("All target_cols must be present in 'features' list.")

        # 5. Entity Identification
        if self.country_col:
            self.countries = sorted(self.raw_df[self.country_col].unique())
        else:
            self.countries = ['single_country']
        self.num_countries = len(self.countries)
        
        print(f"[MacroDataLoader] Initialized: {self.num_countries} entities, "
              f"{len(self.raw_df)} obs, {len(self.cfg['features'])} features.")

    def get_request(
        self, 
        end_train_date: Union[str, pd.Timestamp], 
        horizon: int
    ) -> Dict:
        """
        Generates a training/testing split for a specific temporal cutoff.
        
        **Methodology:**
        1.  **Strict Temporal Split:** Data $\le$ `cutoff` is Train; Data $>$ `cutoff` is Test.
        2.  **Look-Ahead Bias Prevention:** Standardization parameters ($\mu, \sigma$) are 
            estimated *solely* on the training set.
        3.  **Output Structure:** - Train: Dictionary of 2D arrays (Time $\times$ Feat) per country.
            - Test: 3D array (Country $\times$ Horizon $\times$ Targets) for vectorized evaluation.

        

        Args:
            end_train_date: The last inclusive date for the training set.
            horizon: Forecasting horizon $H$ (number of future steps).
            
        Returns:
            Dict containing:
                - 'train': {country: np.ndarray (T, F)}
                - 'test': np.ndarray (N_countries, H, N_targets) (Scaled)
                - 'scalers': {country: sklearn.StandardScaler}
                - 'meta': Metadata dictionary.
        """
        cutoff = pd.to_datetime(end_train_date)
        
        train_dict = {}
        scalers = {}
        # Pre-allocate Test tensor: (N, H, Targets)
        y_test = np.full(
            (self.num_countries, horizon, len(self.target_cols)),
            np.nan
        )

        for i, country in enumerate(self.countries):
            # Isolate Entity Data
            if self.country_col:
                mask = self.raw_df[self.country_col] == country
                df_c = self.raw_df[mask].drop(columns=[self.country_col])
            else:
                df_c = self.raw_df.copy()
    
            # 1. Temporal Split
            df_train_raw = df_c[df_c.index <= cutoff].copy()
            
            if df_train_raw.empty:
                print(f"Warning: No training data for {country} before {cutoff}")
                continue
            
            # 2. Fit Scaler (Train Set Only)
            # z = (x - mean) / std
            scaler = StandardScaler()
            scaled_vals = scaler.fit_transform(df_train_raw[self.cfg['features']])
            
            # Store Scaled Train Data
            # We reconstruct DataFrame to keep alignment but store .values for performance
            train_dict[country] = scaled_vals
            scalers[country] = scaler

            # 3. Construct Test Set (Evaluation)
            df_future = df_c[df_c.index > cutoff].sort_index().head(horizon)

            if not df_future.empty:
                # Apply Train parameters to Test data (transform only)
                # Note: Test targets are returned SCALED to match model output distribution
                df_future[self.cfg['features']] = scaler.transform(df_future[self.cfg['features']])
                
                # Extract Targets
                vals = df_future[self.target_cols].values
                valid_len = min(len(vals), horizon)
                y_test[i, :valid_len, :] = vals[:valid_len, :]

        # Final Validations
        if not train_dict:
            raise RuntimeError("No training data generated for any country.")
            
        return {
            "train": train_dict,
            "test": y_test,
            "scalers": scalers,
            "meta": {
                "countries": self.countries,
                "target_cols": self.target_cols,
                "features": list(self.cfg['features']),
            }
        }