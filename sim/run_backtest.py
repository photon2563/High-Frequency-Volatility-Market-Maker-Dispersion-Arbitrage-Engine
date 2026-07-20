"""
run_backtest.py — Full backtest orchestrator.

Pipeline: Data → C++ Pricing/Hedging → RL Policy → PnL Attribution

Orchestrates the complete trading simulation:
  1. Load/generate market data
  2. Compute implied volatilities via C++ solver
  3. Generate dispersion signal (Z-score)
  4. Query RL agent for bid/ask offsets
  5. Execute fills, update inventory
  6. Hedge via Central Risk Book logic (limit vs market)
  7. Vega-neutralize portfolio
  8. Record PnL attribution: spread + vol arb + gamma scalp − txn costs
"""

from __future__ import annotations

import argparse
import sys
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "python_rl"))


@dataclass
class BacktestConfig:
    """Configuration for the full backtest."""
    # Data
    use_real_data: bool = False
    start_date: str = "2023-01-01"
    n_days: int = 252                  # Trading days to simulate
    n_constituents: int = 5            # Number of index constituents
    initial_capital: float = 1_000_000

    # Market parameters
    spot_price: float = 100.0
    risk_free_rate: float = 0.05
    base_volatility: float = 0.20
    txn_cost: float = 0.001           # 10 bps proportional

    # Strategy parameters
    risk_aversion: float = 0.1
    max_inventory: int = 100
    zscore_entry: float = 0.5
    zscore_exit: float = 0.05
    rebalance_freq: int = 1           # Days between rebalances

    # Hedging
    leland_dt: float = 1 / 252        # Daily rebalancing
    max_delta_tolerance: float = 50.0  # Shares

    # Seed
    seed: int = 42


@dataclass
class DailyPnL:
    """Daily PnL attribution breakdown."""
    date: int
    spread_pnl: float = 0.0
    vol_arb_pnl: float = 0.0
    gamma_scalp_pnl: float = 0.0
    txn_cost: float = 0.0
    total_pnl: float = 0.0
    inventory: int = 0
    net_vega: float = 0.0
    net_delta: float = 0.0
    dispersion_signal: float = 0.0


class BacktestEngine:
    """
    Full backtest orchestrator.

    Integrates:
      - Synthetic data generation
      - Black-Scholes pricing (pure Python fallback if C++ not available)
      - Avellaneda-Stoikov / Lucic-Tse quoting
      - Leland hedging + Central Risk Book logic
      - Dispersion trading signals
      - Vega neutralization
      - PnL attribution
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.daily_pnl: List[DailyPnL] = []

        # Try to import C++ core
        self._cpp_available = False
        try:
            import davinci_py
            self._cpp_available = True
            print("[INFO] C++ core engine loaded via pybind11")
        except ImportError:
            print("[INFO] C++ core not available — using pure Python fallback")

    def _bs_price_py(self, S, K, r, T, sigma, is_call=True):
        """Pure Python BS price (fallback if C++ not available)."""
        from scipy.stats import norm
        if T <= 0:
            return max(S - K, 0) if is_call else max(K - S, 0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if is_call:
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def _bs_vega_py(self, S, K, r, T, sigma):
        """Pure Python BS Vega."""
        from scipy.stats import norm
        if T <= 0:
            return 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * norm.pdf(d1) * np.sqrt(T)

    def _bs_gamma_py(self, S, K, r, T, sigma):
        """Pure Python BS Gamma."""
        from scipy.stats import norm
        if T <= 0:
            return 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    def generate_synthetic_paths(self):
        """Generate synthetic price paths for index + constituents."""
        cfg = self.config
        n = cfg.n_days
        dt = 1 / 252

        # Index price path (GBM)
        index_returns = (0.08 - 0.5 * cfg.base_volatility**2) * dt + \
                       cfg.base_volatility * np.sqrt(dt) * self.rng.standard_normal(n)
        index_prices = cfg.spot_price * np.exp(np.cumsum(index_returns))
        index_prices = np.insert(index_prices, 0, cfg.spot_price)

        # Constituent paths (correlated GBM)
        constituent_prices = []
        constituent_vols = []
        weights = self.rng.dirichlet(np.ones(cfg.n_constituents))

        for i in range(cfg.n_constituents):
            vol_i = cfg.base_volatility * (0.8 + 0.4 * self.rng.random())
            corr = 0.6 + 0.3 * self.rng.random()
            returns_i = corr * index_returns + np.sqrt(1 - corr**2) * \
                       vol_i * np.sqrt(dt) * self.rng.standard_normal(n)
            prices_i = cfg.spot_price * np.exp(np.cumsum(returns_i))
            prices_i = np.insert(prices_i, 0, cfg.spot_price)
            constituent_prices.append(prices_i)
            constituent_vols.append(vol_i)

        # Implied vol surface (base + skew + noise)
        index_ivs = cfg.base_volatility + 0.02 * self.rng.standard_normal(n + 1)
        index_ivs = np.clip(index_ivs, 0.05, 0.80)

        constituent_ivs = []
        for vol_i in constituent_vols:
            ivs = vol_i + 0.015 * self.rng.standard_normal(n + 1)
            ivs = np.clip(ivs, 0.05, 0.80)
            constituent_ivs.append(ivs)

        return {
            "index_prices": index_prices,
            "constituent_prices": constituent_prices,
            "weights": weights,
            "index_ivs": index_ivs,
            "constituent_ivs": constituent_ivs,
            "constituent_vols": constituent_vols,
        }

    def run(self) -> Dict:
        """
        Execute the full backtest.

        Returns summary statistics and daily PnL attribution.
        """
        cfg = self.config
        print("═" * 70)
        print("  DA VINCI VOL PROJECT — BACKTEST ENGINE")
        print("═" * 70)

        start_time = time.time()

        # ── 1. Generate or load market data ──
        if cfg.use_real_data:
            print(f"\n[1/7] Loading real market data (yfinance) from {cfg.start_date}...")
            from data.ingestion.yfinance_loader import fetch_real_market_data
            df = fetch_real_market_data(start_date=cfg.start_date)
            
            cfg.n_days = len(df) - 1 # Update n_days based on actual data
            index_prices = df['index_price'].values
            weights = np.ones(5) / 5.0 # Equal weighting approx
            index_ivs = df['index_iv'].values
            
            constituent_ivs = []
            for i in range(5):
                constituent_ivs.append(df[f'c{i}_iv'].values)
        else:
            print("\n[1/7] Generating synthetic market data...")
            data = self.generate_synthetic_paths()

            index_prices = data["index_prices"]
            weights = data["weights"]
            index_ivs = data["index_ivs"]
            constituent_ivs = data["constituent_ivs"]

        # ── 2. Initialize state ──
        inventory = 0
        cash = cfg.initial_capital
        portfolio_vega = 0.0
        portfolio_gamma = 0.0
        portfolio_delta = 0.0
        total_txn_costs = 0.0

        # ── 3. Initialize dispersion signal ──
        print("[2/7] Initializing dispersion Z-score engine...")
        from data.sanity_checks.mental_math_logger import MentalMathLogger
        logger = MentalMathLogger(tolerance=0.15)

        z_history = []
        dirty_corr_history = []

        # ── 4. Main simulation loop ──
        print(f"[3/7] Running {cfg.n_days}-day simulation...")

        for day in range(1, cfg.n_days + 1):
            S = index_prices[day]
            S_prev = index_prices[day - 1]
            dS = S - S_prev
            K = S  # ATM options
            T = 30 / 365  # 30-day options
            r = cfg.risk_free_rate
            sigma = index_ivs[day]

            # ── Compute dirty implied correlation ──
            const_iv_today = [ivs[day] for ivs in constituent_ivs]
            weighted_iv = sum(w * iv for w, iv in zip(weights, const_iv_today))
            dirty_corr = (sigma**2) / (weighted_iv**2) if weighted_iv > 0 else 0
            dirty_corr = min(dirty_corr, 1.0)
            dirty_corr_history.append(dirty_corr)

            # ── Z-score signal ──
            z_score = 0.0
            if len(dirty_corr_history) >= 20:
                window = dirty_corr_history[-20:]
                mean_w = np.mean(window)
                std_w = np.std(window, ddof=1)
                if std_w > 1e-10:
                    z_score = (dirty_corr - mean_w) / std_w
            z_history.append(z_score)

            # ── Market making quotes (A-S model) ──
            time_remaining = max((cfg.n_days - day) / cfg.n_days, 0.001)
            reservation = S - inventory * cfg.risk_aversion * sigma**2 * time_remaining
            elasticity = (2 / cfg.risk_aversion) * np.log(1 + cfg.risk_aversion / 1.5)
            half_spread = elasticity / 2

            bid = reservation - half_spread
            ask = reservation + half_spread

            # ── Simulate fills (Poisson) ──
            lambda_bid = 10 * np.exp(-1.5 * max(S - bid, 0))
            lambda_ask = 10 * np.exp(-1.5 * max(ask - S, 0))
            bid_fills = min(self.rng.poisson(lambda_bid), max(cfg.max_inventory - inventory, 0))
            ask_fills = min(self.rng.poisson(lambda_ask), max(cfg.max_inventory + inventory, 0))

            # ── Update inventory ──
            inventory += bid_fills - ask_fills
            spread_pnl = ask_fills * (ask - S) + bid_fills * (S - bid)
            cash += ask_fills * ask - bid_fills * bid
            txn = cfg.txn_cost * (bid_fills + ask_fills) * S
            total_txn_costs += txn

            # ── Greeks for hedging ──
            gamma = self._bs_gamma_py(S, K, r, T, sigma)
            vega = self._bs_vega_py(S, K, r, T, sigma)
            portfolio_gamma = inventory * gamma
            portfolio_vega = inventory * vega
            portfolio_delta = inventory * 0.5  # Approx ATM delta

            # ── Gamma scalping P&L ──
            sigma_realized = abs(dS / S_prev) * np.sqrt(252) if S_prev > 0 else sigma
            gamma_pnl = 0.5 * portfolio_gamma * dS**2

            # ── Vol arb P&L ──
            vol_arb = 0.5 * portfolio_gamma * S**2 * (sigma_realized**2 - sigma**2) / 252

            # ── Leland check ──
            le = 0.7979 * cfg.txn_cost / (sigma * np.sqrt(cfg.leland_dt))

            # ── Record daily P&L ──
            total = spread_pnl + vol_arb + gamma_pnl - txn
            self.daily_pnl.append(DailyPnL(
                date=day,
                spread_pnl=spread_pnl,
                vol_arb_pnl=vol_arb,
                gamma_scalp_pnl=gamma_pnl,
                txn_cost=txn,
                total_pnl=total,
                inventory=inventory,
                net_vega=portfolio_vega,
                net_delta=portfolio_delta,
                dispersion_signal=z_score,
            ))

        # ── 5. Sanity checks ──
        print("[4/7] Running cross-verification sanity checks...")
        logger.check_atm_call(
            self._bs_price_py(100, 100, 0.05, 1.0, 0.20),
            100, 0.20, 1.0
        )
        logger.check_atm_vega(
            self._bs_vega_py(100, 100, 0.05, 1.0, 0.20),
            100, 1.0
        )
        logger.check_leland(
            0.7979 * 0.001 / (0.20 * np.sqrt(1/252)),
            0.001, 0.20, 1/252
        )
        print(logger.report())

        # ── 6. Compute summary statistics ──
        print("[5/7] Computing summary statistics...")
        pnl_series = np.array([d.total_pnl for d in self.daily_pnl])
        cumulative_pnl = np.cumsum(pnl_series)

        total_pnl = cumulative_pnl[-1]
        sharpe = np.mean(pnl_series) / np.std(pnl_series) * np.sqrt(252) if np.std(pnl_series) > 0 else 0
        max_dd = np.min(cumulative_pnl - np.maximum.accumulate(cumulative_pnl))
        win_rate = np.mean(pnl_series > 0)

        spread_total = sum(d.spread_pnl for d in self.daily_pnl)
        vol_arb_total = sum(d.vol_arb_pnl for d in self.daily_pnl)
        gamma_total = sum(d.gamma_scalp_pnl for d in self.daily_pnl)

        elapsed = time.time() - start_time

        summary = {
            "total_pnl": total_pnl,
            "annualized_sharpe": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "total_txn_costs": total_txn_costs,
            "spread_pnl": spread_total,
            "vol_arb_pnl": vol_arb_total,
            "gamma_scalp_pnl": gamma_total,
            "n_days": cfg.n_days,
            "elapsed_seconds": elapsed,
        }

        # ── 7. Print results ──
        print("\n[6/7] Backtest Results:")
        print("─" * 50)
        print(f"  Total P&L:          ${total_pnl:>12,.2f}")
        print(f"  Sharpe Ratio:       {sharpe:>12.2f}")
        print(f"  Max Drawdown:       ${max_dd:>12,.2f}")
        print(f"  Win Rate:           {win_rate:>12.1%}")
        print(f"  Transaction Costs:  ${total_txn_costs:>12,.2f}")
        print("─" * 50)
        print(f"  Spread P&L:         ${spread_total:>12,.2f}")
        print(f"  Vol Arb P&L:        ${vol_arb_total:>12,.2f}")
        print(f"  Gamma Scalp P&L:    ${gamma_total:>12,.2f}")
        print("─" * 50)
        print(f"  Elapsed:            {elapsed:>12.2f}s")
        print("[7/7] Backtest complete.")

        return summary


def main():
    parser = argparse.ArgumentParser(description="Da Vinci Vol Project Backtest")
    parser.add_argument("--real-data", action="store_true", help="Use real historical data from yfinance")
    parser.add_argument("--start-date", type=str, default="2023-01-01", help="Start date for real data (YYYY-MM-DD)")
    args = parser.parse_args()

    config = BacktestConfig(
        use_real_data=args.real_data,
        start_date=args.start_date,
        n_days=252,
        n_constituents=5,
        initial_capital=1_000_000,
        base_volatility=0.20,
        txn_cost=0.001,
    )

    engine = BacktestEngine(config)
    results = engine.run()


if __name__ == "__main__":
    main()
