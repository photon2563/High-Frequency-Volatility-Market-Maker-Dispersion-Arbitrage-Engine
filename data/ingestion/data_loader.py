"""
options_loader.py — Options chain data loader (CSV/Parquet).
underlying_loader.py — Spot/futures price loader.

Data ingestion stubs for the backtest pipeline.
In production, these would connect to market data APIs or data warehouses.
For development, they generate synthetic data for testing.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from pathlib import Path


@dataclass
class OptionsChain:
    """Container for an options chain snapshot."""
    timestamp: float
    underlying_price: float
    strikes: np.ndarray
    expiries: np.ndarray       # Years to expiry
    call_bids: np.ndarray      # n_strikes x n_expiries
    call_asks: np.ndarray
    put_bids: np.ndarray
    put_asks: np.ndarray
    call_volumes: np.ndarray
    put_volumes: np.ndarray
    implied_vols: np.ndarray   # Mid IV surface


@dataclass
class UnderlyingData:
    """Container for underlying asset price history."""
    dates: np.ndarray          # Unix timestamps or ordinal dates
    open_prices: np.ndarray
    high_prices: np.ndarray
    low_prices: np.ndarray
    close_prices: np.ndarray
    volumes: np.ndarray
    vix: Optional[np.ndarray] = None  # VIX index if available


class SyntheticDataGenerator:
    """
    Generate synthetic market data for backtest development.

    Simulates:
      - GBM price paths with stochastic volatility (Heston-like)
      - Options chains with realistic IV surface (skew + term structure)
      - VIX derived from ATM implied vol
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_prices(
        self,
        n_days: int = 252,
        S0: float = 100.0,
        mu: float = 0.08,
        sigma: float = 0.20,
        dt: float = 1 / 252,
    ) -> UnderlyingData:
        """Generate synthetic daily price data via GBM."""
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * self.rng.standard_normal(n_days)
        prices = S0 * np.exp(np.cumsum(log_returns))
        prices = np.insert(prices, 0, S0)

        # Simulate realistic OHLC
        close = prices
        noise = sigma * np.sqrt(dt) * 0.3
        high = close * (1 + np.abs(self.rng.normal(0, noise, len(close))))
        low = close * (1 - np.abs(self.rng.normal(0, noise, len(close))))
        open_prices = np.roll(close, 1)
        open_prices[0] = S0

        # Volume (log-normal)
        volumes = np.exp(self.rng.normal(np.log(1e6), 0.3, len(close)))

        # VIX: ATM implied vol * 100, with some noise
        base_vix = sigma * 100
        vix = base_vix + self.rng.normal(0, 2, len(close))
        vix = np.clip(vix, 10, 80)

        return UnderlyingData(
            dates=np.arange(len(close)),
            open_prices=open_prices,
            high_prices=high,
            low_prices=low,
            close_prices=close,
            volumes=volumes,
            vix=vix,
        )

    def generate_options_chain(
        self,
        S: float,
        r: float = 0.05,
        sigma_base: float = 0.20,
        n_strikes: int = 11,
        n_expiries: int = 4,
    ) -> OptionsChain:
        """
        Generate a synthetic options chain with realistic IV surface.

        IV surface features:
          - Skew: OTM puts have higher IV (protective put demand)
          - Term structure: longer maturities → slightly higher IV
          - Random noise: ±1 vol point
        """
        # Strike range: 85% to 115% of spot
        strikes = np.linspace(0.85 * S, 1.15 * S, n_strikes)
        expiries = np.array([30, 60, 90, 180][:n_expiries]) / 365.0

        # IV surface with skew
        iv_surface = np.zeros((n_strikes, n_expiries))
        for i, K in enumerate(strikes):
            for j, T in enumerate(expiries):
                moneyness = np.log(K / S) / (sigma_base * np.sqrt(T))
                # Skew: IV increases for low strikes (OTM puts)
                skew = -0.15 * moneyness  # negative moneyness → higher IV
                # Term structure: slight upward slope
                term = 0.02 * np.sqrt(T)
                # Noise
                noise = self.rng.normal(0, 0.005)
                iv_surface[i, j] = sigma_base + skew + term + noise

        iv_surface = np.clip(iv_surface, 0.05, 1.0)

        # Generate bid/ask from IV (spread ≈ 1-2 vol points)
        spread_vol = 0.01 + 0.005 * self.rng.random((n_strikes, n_expiries))

        # Black-Scholes prices (simplified — would use C++ engine in production)
        from scipy.stats import norm

        def bs_price(S, K, r, T, sigma, is_call):
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if is_call:
                return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        call_mids = np.zeros((n_strikes, n_expiries))
        put_mids = np.zeros((n_strikes, n_expiries))
        for i, K in enumerate(strikes):
            for j, T in enumerate(expiries):
                call_mids[i, j] = bs_price(S, K, r, T, iv_surface[i, j], True)
                put_mids[i, j] = bs_price(S, K, r, T, iv_surface[i, j], False)

        # Bid/ask
        call_half_spread = call_mids * spread_vol / iv_surface
        put_half_spread = put_mids * spread_vol / iv_surface

        call_bids = np.maximum(call_mids - call_half_spread, 0.01)
        call_asks = call_mids + call_half_spread
        put_bids = np.maximum(put_mids - put_half_spread, 0.01)
        put_asks = put_mids + put_half_spread

        # Volumes
        call_volumes = np.exp(self.rng.normal(5, 1, (n_strikes, n_expiries)))
        put_volumes = np.exp(self.rng.normal(5, 1, (n_strikes, n_expiries)))

        return OptionsChain(
            timestamp=0.0,
            underlying_price=S,
            strikes=strikes,
            expiries=expiries,
            call_bids=call_bids,
            call_asks=call_asks,
            put_bids=put_bids,
            put_asks=put_asks,
            call_volumes=call_volumes,
            put_volumes=put_volumes,
            implied_vols=iv_surface,
        )


def load_csv(filepath: str) -> UnderlyingData:
    """Load underlying price data from CSV. Expects columns: date,open,high,low,close,volume"""
    import csv
    data = {"open": [], "high": [], "low": [], "close": [], "volume": []}
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in data:
                data[key].append(float(row.get(key, 0)))

    return UnderlyingData(
        dates=np.arange(len(data["close"])),
        open_prices=np.array(data["open"]),
        high_prices=np.array(data["high"]),
        low_prices=np.array(data["low"]),
        close_prices=np.array(data["close"]),
        volumes=np.array(data["volume"]),
    )
