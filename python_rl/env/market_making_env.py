"""
market_making_env.py — Gymnasium-compatible environment for options market making.

Wraps the C++ core pricing/hedging engine via pybind11.
Implements a continuous-action MDP where the agent controls bid/ask offsets.

State:  [mid_price, inventory, time_to_horizon, sigma, spread, order_imbalance]
Action: [bid_offset, ask_offset] ∈ ℝ² (offsets from reservation price)
Reward: spread_captured - γ · q² · σ²  (mean-variance objective)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class MarketConfig:
    """Configuration for the market-making environment."""
    # Asset parameters
    initial_price: float = 100.0
    volatility: float = 0.02           # Per-step volatility (≈ 20% annualized if daily)
    drift: float = 0.0                 # Price drift per step

    # Market microstructure
    base_arrival_rate: float = 10.0    # Base order arrival intensity (orders/step)
    arrival_decay: float = 1.5         # Exponential decay with spread (κ)

    # Inventory & risk
    max_inventory: int = 50            # Maximum absolute inventory
    risk_aversion: float = 0.1         # γ — inventory risk penalty weight
    max_steps: int = 1000              # Steps per episode (≈ 1 trading day)

    # Reward shaping
    spread_reward_scale: float = 1.0
    inventory_penalty_scale: float = 1.0
    terminal_penalty_scale: float = 10.0  # End-of-day unwinding cost

    # Execution
    lot_size: int = 1                  # Minimum trade size
    tick_size: float = 0.01            # Minimum price increment


class MarketMakingEnv(gym.Env):
    """
    Options Market Making Environment.

    The agent posts bid/ask quotes and earns the spread on fills.
    Inventory accumulates and must be managed to avoid directional risk.
    Reward penalizes large inventory positions (mean-variance objective).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Optional[MarketConfig] = None, render_mode=None):
        super().__init__()
        self.config = config or MarketConfig()
        self.render_mode = render_mode

        # ── Action space: [bid_offset, ask_offset] ──
        # Offsets from the reservation price. Both must be positive (distance from mid).
        self.action_space = spaces.Box(
            low=np.array([0.001, 0.001], dtype=np.float32),
            high=np.array([0.5, 0.5], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        # ── Observation space ──
        # [mid_price_normalized, inventory_normalized, time_remaining_frac,
        #  sigma, recent_spread, order_imbalance]
        self.observation_space = spaces.Box(
            low=np.array([-5.0, -1.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([5.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            shape=(6,),
            dtype=np.float32,
        )

        # State variables
        self._reset_state()

    def _reset_state(self):
        self.mid_price = self.config.initial_price
        self.inventory = 0
        self.cash = 0.0
        self.step_count = 0
        self.pnl_history = []
        self.trade_history = []
        self.realized_spread_history = []
        self._recent_bid_fills = 0
        self._recent_ask_fills = 0

    def _get_obs(self) -> np.ndarray:
        """Construct observation vector."""
        # Normalize mid_price: (price - initial) / initial
        price_norm = (self.mid_price - self.config.initial_price) / self.config.initial_price

        # Normalize inventory: q / max_inventory
        inv_norm = self.inventory / self.config.max_inventory

        # Time remaining fraction
        time_frac = 1.0 - (self.step_count / self.config.max_steps)

        # Sigma (already in reasonable range)
        sigma = self.config.volatility

        # Recent spread (normalized)
        spread = np.mean(self.realized_spread_history[-10:]) if self.realized_spread_history else 0.0
        spread_norm = min(spread / self.mid_price, 1.0)

        # Order imbalance
        total_fills = self._recent_bid_fills + self._recent_ask_fills + 1e-10
        order_imbalance = (self._recent_ask_fills - self._recent_bid_fills) / total_fills

        return np.array([
            np.clip(price_norm, -5.0, 5.0),
            np.clip(inv_norm, -1.0, 1.0),
            time_frac,
            sigma,
            spread_norm,
            np.clip(order_imbalance, -1.0, 1.0),
        ], dtype=np.float32)

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step of the market-making simulation.

        1. Agent posts bid/ask quotes (action = [bid_offset, ask_offset])
        2. Mid-price evolves (GBM)
        3. Fills occur probabilistically (Poisson with intensity decaying in spread)
        4. Inventory and cash update
        5. Reward = spread_captured - risk_penalty
        """
        bid_offset = float(action[0])
        ask_offset = float(action[1])

        # ── Compute reservation price (Avellaneda-Stoikov) ──
        time_remaining = max(
            (self.config.max_steps - self.step_count) / self.config.max_steps, 1e-6
        )
        gamma = self.config.risk_aversion
        sigma = self.config.volatility
        reservation_price = self.mid_price - self.inventory * gamma * sigma**2 * time_remaining

        # ── Post quotes ──
        bid_price = reservation_price - bid_offset
        ask_price = reservation_price + ask_offset
        spread = ask_price - bid_price

        # ── Mid-price evolution (GBM) ──
        dW = self.np_random.standard_normal()
        self.mid_price *= np.exp(
            (self.config.drift - 0.5 * sigma**2) + sigma * dW
        )

        # ── Order arrival (Poisson with exponential decay) ──
        bid_distance = max(self.mid_price - bid_price, 0)
        ask_distance = max(ask_price - self.mid_price, 0)

        kappa = self.config.arrival_decay
        A = self.config.base_arrival_rate

        lambda_bid = A * np.exp(-kappa * bid_distance)
        lambda_ask = A * np.exp(-kappa * ask_distance)

        # Poisson draws for fills
        bid_fills = self.np_random.poisson(lambda_bid)
        ask_fills = self.np_random.poisson(lambda_ask)

        # Inventory limits
        max_buy = self.config.max_inventory - self.inventory
        max_sell = self.config.max_inventory + self.inventory

        bid_fills = min(bid_fills, max(max_buy, 0)) * self.config.lot_size
        ask_fills = min(ask_fills, max(max_sell, 0)) * self.config.lot_size

        # ── Update inventory and cash ──
        # Bid fills: we BUY from the market → inventory increases, cash decreases
        self.inventory += bid_fills
        self.cash -= bid_fills * bid_price

        # Ask fills: we SELL to the market → inventory decreases, cash increases
        self.inventory -= ask_fills
        self.cash += ask_fills * ask_price

        # ── Compute reward ──
        # Spread capture
        spread_captured = ask_fills * ask_price - bid_fills * bid_price
        if bid_fills + ask_fills > 0:
            self.realized_spread_history.append(spread)

        # Inventory risk penalty: γ · q² · σ²
        inventory_penalty = gamma * self.inventory**2 * sigma**2

        reward = (
            self.config.spread_reward_scale * spread_captured
            - self.config.inventory_penalty_scale * inventory_penalty
        )

        # Track fills for order imbalance
        self._recent_bid_fills = bid_fills
        self._recent_ask_fills = ask_fills

        # ── Record ──
        self.step_count += 1
        mark_to_market = self.cash + self.inventory * self.mid_price
        self.pnl_history.append(mark_to_market)

        if bid_fills > 0 or ask_fills > 0:
            self.trade_history.append({
                "step": self.step_count,
                "bid_fills": bid_fills,
                "ask_fills": ask_fills,
                "bid_price": bid_price,
                "ask_price": ask_price,
                "inventory": self.inventory,
            })

        # ── Termination ──
        terminated = self.step_count >= self.config.max_steps
        truncated = abs(self.inventory) > self.config.max_inventory * 1.5

        # Terminal penalty: cost of unwinding remaining inventory
        if terminated and self.inventory != 0:
            unwinding_cost = (
                abs(self.inventory) * sigma * self.mid_price
                * self.config.terminal_penalty_scale
            )
            reward -= unwinding_cost

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def _get_info(self) -> Dict[str, Any]:
        return {
            "mid_price": self.mid_price,
            "inventory": self.inventory,
            "cash": self.cash,
            "mark_to_market": self.cash + self.inventory * self.mid_price,
            "step": self.step_count,
            "n_trades": len(self.trade_history),
        }

    def render(self):
        if self.render_mode == "human":
            info = self._get_info()
            print(
                f"Step {info['step']:4d} | "
                f"Mid: {info['mid_price']:.2f} | "
                f"Inv: {info['inventory']:+4d} | "
                f"MtM: {info['mark_to_market']:.2f}"
            )
