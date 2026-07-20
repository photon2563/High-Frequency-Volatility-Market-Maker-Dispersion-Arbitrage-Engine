"""
svm_regime_classifier.py — RBF kernel SVM for market regime classification.

Classifies market into regimes: {low_vol, normal, high_vol, crisis}.
Uses macro/microstructure features for regime identification.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import IntEnum


class MarketRegime(IntEnum):
    """Market regime labels."""
    LOW_VOL = 0     # VIX < 15, calm markets
    NORMAL = 1      # VIX 15-25, typical conditions
    HIGH_VOL = 2    # VIX 25-35, elevated uncertainty
    CRISIS = 3      # VIX > 35, extreme stress


REGIME_NAMES = {
    MarketRegime.LOW_VOL: "Low Volatility",
    MarketRegime.NORMAL: "Normal",
    MarketRegime.HIGH_VOL: "High Volatility",
    MarketRegime.CRISIS: "Crisis",
}


@dataclass
class RegimeConfig:
    """Configuration for the SVM regime classifier."""
    # SVM hyperparameters
    kernel: str = "rbf"
    C: float = 10.0              # Regularization (higher = less regularization)
    gamma: str = "scale"         # RBF kernel bandwidth
    class_weight: str = "balanced"  # Handle imbalanced classes

    # Feature engineering
    vix_thresholds: List[float] = field(
        default_factory=lambda: [15.0, 25.0, 35.0]
    )
    lookback_windows: List[int] = field(
        default_factory=lambda: [5, 10, 20, 60]
    )

    # Cross-validation
    n_cv_folds: int = 5
    test_size: float = 0.2


class RegimeFeatureBuilder:
    """Builds macro/microstructure features for regime classification."""

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()

    def label_regimes(
        self, vix_series: np.ndarray,
        thresholds: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Assign regime labels based on VIX levels.

        VIX < 15     → LOW_VOL
        15 ≤ VIX < 25 → NORMAL
        25 ≤ VIX < 35 → HIGH_VOL
        VIX ≥ 35     → CRISIS
        """
        t = thresholds or self.config.vix_thresholds
        labels = np.full(len(vix_series), MarketRegime.NORMAL, dtype=int)

        labels[vix_series < t[0]] = MarketRegime.LOW_VOL
        labels[(vix_series >= t[0]) & (vix_series < t[1])] = MarketRegime.NORMAL
        labels[(vix_series >= t[1]) & (vix_series < t[2])] = MarketRegime.HIGH_VOL
        labels[vix_series >= t[2]] = MarketRegime.CRISIS

        return labels

    def build_features(
        self,
        prices: np.ndarray,
        vix: Optional[np.ndarray] = None,
        implied_vols: Optional[np.ndarray] = None,
        volumes: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build feature matrix from market data.

        Features:
          - VIX level and rate of change
          - VIX term structure slope (if available)
          - Realized vol at multiple windows
          - Realized-implied vol gap
          - Price momentum (multi-scale)
          - Return skewness and kurtosis
          - Volume ratio (if available)
        """
        n = len(prices)
        log_returns = np.diff(np.log(prices))
        features = {}
        names = []

        # ── VIX features (if available) ──
        if vix is not None:
            features["vix_level"] = vix[1:]
            names.append("vix_level")

            vix_change = np.diff(vix)
            features["vix_1d_change"] = vix_change
            names.append("vix_1d_change")

            # 5-day VIX rate of change
            vix_roc = np.full(len(vix), np.nan)
            for i in range(5, len(vix)):
                vix_roc[i] = (vix[i] - vix[i - 5]) / vix[i - 5]
            features["vix_5d_roc"] = vix_roc[1:]
            names.append("vix_5d_roc")

        # ── Realized vol at multiple windows ──
        for window in self.config.lookback_windows:
            rv = np.full(len(log_returns), np.nan)
            for i in range(window - 1, len(log_returns)):
                rv[i] = np.sqrt(252 * np.mean(log_returns[i - window + 1 : i + 1] ** 2))
            features[f"rv_{window}d"] = rv
            names.append(f"rv_{window}d")

        # ── Realized-implied vol gap ──
        if implied_vols is not None:
            rv_20 = features.get("rv_20d", np.full(len(log_returns), np.nan))
            iv_rv_gap = implied_vols[1:] - rv_20
            min_len = min(len(iv_rv_gap), len(log_returns))
            features["iv_rv_gap"] = iv_rv_gap[:min_len]
            names.append("iv_rv_gap")

        # ── Return momentum (multi-scale) ──
        for window in [5, 10, 20]:
            mom = np.full(len(log_returns), np.nan)
            for i in range(window, len(log_returns)):
                mom[i] = np.sum(log_returns[i - window : i])
            features[f"momentum_{window}d"] = mom
            names.append(f"momentum_{window}d")

        # ── Return distribution features ──
        for window in [20, 60]:
            skew = np.full(len(log_returns), np.nan)
            kurt = np.full(len(log_returns), np.nan)
            for i in range(window, len(log_returns)):
                r = log_returns[i - window : i]
                mu = np.mean(r)
                sig = np.std(r, ddof=1)
                if sig > 1e-15:
                    skew[i] = np.mean(((r - mu) / sig) ** 3)
                    kurt[i] = np.mean(((r - mu) / sig) ** 4) - 3  # Excess kurtosis
            features[f"skewness_{window}d"] = skew
            features[f"kurtosis_{window}d"] = kurt
            names.extend([f"skewness_{window}d", f"kurtosis_{window}d"])

        # ── Volume features ──
        if volumes is not None:
            vol_ratio = np.full(len(volumes), np.nan)
            for i in range(20, len(volumes)):
                vol_ratio[i] = volumes[i] / np.mean(volumes[i - 20 : i])
            features["volume_ratio_20d"] = vol_ratio[1:]
            names.append("volume_ratio_20d")

        # ── Align all features ──
        min_len = min(len(v) for v in features.values())
        X = np.column_stack([v[-min_len:] for v in features.values()])

        return X, names


class SVMRegimeClassifier:
    """SVM-based market regime classifier."""

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self.model = None
        self.scaler = None
        self.feature_builder = RegimeFeatureBuilder(self.config)
        self.feature_names: List[str] = []

    def fit(
        self,
        prices: np.ndarray,
        vix: np.ndarray,
        implied_vols: Optional[np.ndarray] = None,
        volumes: Optional[np.ndarray] = None,
    ) -> Dict:
        """Train the SVM classifier."""
        try:
            from sklearn.svm import SVC
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import classification_report, accuracy_score
        except ImportError:
            raise ImportError("scikit-learn required. Install: pip install scikit-learn")

        # Build features and labels
        X, self.feature_names = self.feature_builder.build_features(
            prices, vix, implied_vols, volumes
        )
        labels = self.feature_builder.label_regimes(vix)

        # Align labels with features
        min_len = min(len(X), len(labels))
        X = X[-min_len:]
        y = labels[-min_len:]

        # Remove NaN rows
        valid_mask = ~np.any(np.isnan(X), axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) == 0:
            raise ValueError("No valid samples after NaN removal")

        # Time-series split
        split_idx = int(len(X) * (1 - self.config.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Standardize features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train SVM
        self.model = SVC(
            kernel=self.config.kernel,
            C=self.config.C,
            gamma=self.config.gamma,
            class_weight=self.config.class_weight,
            probability=True,  # Enable predict_proba
            random_state=42,
        )
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)

        return {
            "train_accuracy": float(accuracy_score(y_train, y_pred_train)),
            "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "class_distribution": {
                REGIME_NAMES[MarketRegime(i)]: int(np.sum(y == i))
                for i in range(4) if np.sum(y == i) > 0
            },
        }

    def predict(self, features: np.ndarray) -> MarketRegime:
        """Predict market regime from features."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        features_scaled = self.scaler.transform(features.reshape(1, -1))
        pred = self.model.predict(features_scaled)
        return MarketRegime(pred[0])

    def predict_proba(self, features: np.ndarray) -> Dict[str, float]:
        """Predict regime probabilities."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not trained.")

        features_scaled = self.scaler.transform(features.reshape(1, -1))
        proba = self.model.predict_proba(features_scaled)[0]

        return {
            REGIME_NAMES[MarketRegime(i)]: float(proba[i])
            for i in range(len(proba))
        }
