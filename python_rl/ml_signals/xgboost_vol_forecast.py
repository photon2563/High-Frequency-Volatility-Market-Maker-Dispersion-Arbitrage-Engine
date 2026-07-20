"""
xgboost_vol_forecast.py — Near-term realized volatility prediction using XGBoost.

Features: lagged realized vol (1d, 5d, 20d), IV-RV spread, VIX term structure slope,
          intraday high-low range, log return autocorrelation.
Target: next-day realized volatility.
Regularization: max_depth=4, learning_rate=0.05, subsample=0.8.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple


@dataclass
class VolForecastConfig:
    """Configuration for the XGBoost vol forecaster."""
    # Feature engineering
    rv_lags: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    iv_rv_spread_lag: int = 1
    lookback_window: int = 60

    # XGBoost hyperparameters
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1       # L1 regularization
    reg_lambda: float = 1.0      # L2 regularization
    min_child_weight: int = 5

    # Evaluation
    test_size: float = 0.2
    n_cv_folds: int = 5


class FeatureBuilder:
    """Constructs features for realized vol prediction from price data."""

    def __init__(self, config: Optional[VolForecastConfig] = None):
        self.config = config or VolForecastConfig()

    def compute_realized_vol(
        self, prices: np.ndarray, window: int = 20
    ) -> np.ndarray:
        """
        Compute rolling realized volatility (annualized).

        RV = √(252 · Σ r²_t / n)  where r_t = log(P_t / P_{t-1})
        """
        log_returns = np.diff(np.log(prices))
        rv = np.full(len(log_returns), np.nan)

        for i in range(window - 1, len(log_returns)):
            window_returns = log_returns[i - window + 1 : i + 1]
            rv[i] = np.sqrt(252 * np.mean(window_returns**2))

        return rv

    def compute_features(
        self,
        prices: np.ndarray,
        implied_vols: Optional[np.ndarray] = None,
        high_prices: Optional[np.ndarray] = None,
        low_prices: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build feature matrix and target vector.

        @param prices: Daily close prices
        @param implied_vols: Daily ATM implied volatility (optional)
        @param high_prices: Daily highs for Parkinson vol (optional)
        @param low_prices: Daily lows for Parkinson vol (optional)
        @returns (X, y, feature_names)
        """
        n = len(prices)
        log_returns = np.diff(np.log(prices))

        features = {}
        feature_names = []

        # ── Lagged realized volatilities ──
        for lag in self.config.rv_lags:
            rv = self.compute_realized_vol(prices, window=lag)
            key = f"rv_{lag}d"
            features[key] = rv[:-1]  # Align with target (next-day)
            feature_names.append(key)

        # ── IV-RV spread ──
        if implied_vols is not None:
            rv_20 = self.compute_realized_vol(prices, window=20)
            iv_rv_spread = implied_vols[1:] - rv_20[:-1]  # Align
            features["iv_rv_spread"] = iv_rv_spread[: len(log_returns) - 1]
            feature_names.append("iv_rv_spread")

        # ── Log return features ──
        features["return_1d"] = log_returns[:-1]
        feature_names.append("return_1d")

        features["abs_return_1d"] = np.abs(log_returns[:-1])
        feature_names.append("abs_return_1d")

        # ── Squared return (proxy for instantaneous variance) ──
        features["squared_return"] = log_returns[:-1] ** 2
        feature_names.append("squared_return")

        # ── Return autocorrelation (5-day rolling) ──
        autocorr = np.full(len(log_returns), np.nan)
        for i in range(5, len(log_returns)):
            r = log_returns[i - 5 : i]
            r_lag = log_returns[i - 6 : i - 1]
            if np.std(r) > 1e-15 and np.std(r_lag) > 1e-15:
                autocorr[i] = np.corrcoef(r, r_lag)[0, 1]
            else:
                autocorr[i] = 0.0
        features["return_autocorr_5d"] = autocorr[:-1]
        feature_names.append("return_autocorr_5d")

        # ── Parkinson volatility (if high/low available) ──
        if high_prices is not None and low_prices is not None:
            hl_ratio = np.log(high_prices / low_prices)
            parkinson = np.full(n, np.nan)
            for i in range(19, n):
                window = hl_ratio[i - 19 : i + 1]
                parkinson[i] = np.sqrt(252 / (4 * np.log(2)) * np.mean(window**2))
            features["parkinson_vol_20d"] = parkinson[1:-1]
            feature_names.append("parkinson_vol_20d")

        # ── Target: next-day realized vol ──
        # Using 1-day squared return as proxy for realized variance
        target = np.abs(log_returns[1:]) * np.sqrt(252)  # Annualized

        # ── Align all features ──
        min_len = min(len(v) for v in features.values())
        min_len = min(min_len, len(target))

        X = np.column_stack([v[-min_len:] for v in features.values()])
        y = target[-min_len:]

        # Remove NaN rows
        valid_mask = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        return X, y, feature_names


class XGBoostVolForecaster:
    """XGBoost-based near-term realized volatility forecaster."""

    def __init__(self, config: Optional[VolForecastConfig] = None):
        self.config = config or VolForecastConfig()
        self.model = None
        self.feature_builder = FeatureBuilder(self.config)
        self.feature_names: List[str] = []

    def fit(
        self,
        prices: np.ndarray,
        implied_vols: Optional[np.ndarray] = None,
        high_prices: Optional[np.ndarray] = None,
        low_prices: Optional[np.ndarray] = None,
    ) -> Dict:
        """Train the XGBoost model on historical data."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost required. Install: pip install xgboost")

        X, y, self.feature_names = self.feature_builder.compute_features(
            prices, implied_vols, high_prices, low_prices
        )

        # Train/test split (time-series aware — no shuffle)
        split_idx = int(len(X) * (1 - self.config.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        self.model = xgb.XGBRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            min_child_weight=self.config.min_child_weight,
            objective="reg:squarederror",
            random_state=42,
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Performance metrics
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        from sklearn.metrics import mean_squared_error, r2_score
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        return {
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict realized volatility from features."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self.model.predict(features)

    def feature_importance(self) -> Dict[str, float]:
        """Get feature importance ranking."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))
