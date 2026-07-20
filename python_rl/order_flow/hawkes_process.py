"""
hawkes_process.py — Self-exciting Hawkes process for order flow simulation.

Replaces naive Poisson order arrivals with realistic clustering behavior.
λ(t) = μ + Σ_{t_i < t} α · exp(-β(t - t_i))

Key properties:
  - Self-exciting: each event increases the probability of future events
  - Mean-reverting intensity: decays back to baseline μ
  - Captures order flow clustering observed in real LOBs
  - Stationarity condition: α/β < 1
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class HawkesParams:
    """Parameters for a univariate Hawkes process."""
    mu: float = 1.0     # Baseline intensity
    alpha: float = 0.5   # Excitation magnitude
    beta: float = 1.0    # Decay rate

    @property
    def branching_ratio(self) -> float:
        """α/β — must be < 1 for stationarity."""
        return self.alpha / self.beta

    @property
    def is_stationary(self) -> bool:
        return self.branching_ratio < 1.0

    @property
    def stationary_mean_intensity(self) -> float:
        """E[λ] = μ / (1 - α/β)"""
        if not self.is_stationary:
            return float("inf")
        return self.mu / (1.0 - self.branching_ratio)

    def validate(self):
        if self.mu <= 0:
            raise ValueError("Baseline intensity μ must be positive")
        if self.alpha < 0:
            raise ValueError("Excitation magnitude α must be non-negative")
        if self.beta <= 0:
            raise ValueError("Decay rate β must be positive")
        if not self.is_stationary:
            raise ValueError(
                f"Process is non-stationary: α/β = {self.branching_ratio:.3f} ≥ 1"
            )


class HawkesProcess:
    """
    Univariate self-exciting Hawkes process.

    Simulates order arrival times where each arrival temporarily
    increases the probability of subsequent arrivals, creating
    the clustering behavior observed in real limit order books.

    Uses Ogata's modified thinning algorithm for exact simulation.
    """

    def __init__(self, params: Optional[HawkesParams] = None, seed: Optional[int] = None):
        self.params = params or HawkesParams()
        self.params.validate()
        self.rng = np.random.default_rng(seed)

        # Event history
        self.event_times: List[float] = []
        self.intensity_history: List[Tuple[float, float]] = []

    def intensity(self, t: float) -> float:
        """
        Compute the intensity λ(t) at time t.

        λ(t) = μ + Σ_{t_i < t} α · exp(-β(t - t_i))
        """
        lam = self.params.mu
        for ti in self.event_times:
            if ti < t:
                lam += self.params.alpha * np.exp(-self.params.beta * (t - ti))
        return lam

    def intensity_upper_bound(self, t: float) -> float:
        """
        Compute an upper bound λ*(t) for the thinning algorithm.

        At time t, the maximum possible intensity before the next event
        is λ(t) itself (since without new events, intensity only decays).
        """
        return self.intensity(t)

    def simulate(self, T: float, max_events: int = 100000) -> np.ndarray:
        """
        Simulate event times on [0, T] using Ogata's thinning algorithm.

        The algorithm:
          1. Compute upper bound λ* ≥ λ(t) for all t in the current interval
          2. Generate candidate event at t + Exp(1/λ*)
          3. Accept with probability λ(t_candidate) / λ*
          4. If accepted, record event and update intensity

        This produces exact samples from the Hawkes process.

        @param T: End time of simulation
        @param max_events: Safety cap on number of events
        @returns Array of event times
        """
        self.event_times = []
        self.intensity_history = []
        t = 0.0
        n_events = 0

        while t < T and n_events < max_events:
            # Upper bound on intensity
            lam_star = self.intensity(t)

            if lam_star <= 0:
                break

            # Generate candidate inter-arrival time
            u1 = self.rng.random()
            dt = -np.log(u1) / lam_star
            t_candidate = t + dt

            if t_candidate >= T:
                break

            # Thinning: accept/reject
            lam_candidate = self.intensity(t_candidate)
            u2 = self.rng.random()

            if u2 <= lam_candidate / lam_star:
                # Accept
                self.event_times.append(t_candidate)
                self.intensity_history.append((t_candidate, lam_candidate))
                n_events += 1

            t = t_candidate

        return np.array(self.event_times)

    def simulate_with_intensities(
        self, T: float, dt: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate events AND record the intensity path at regular intervals.

        @param T: End time
        @param dt: Grid spacing for intensity recording
        @returns (event_times, time_grid, intensity_on_grid)
        """
        events = self.simulate(T)

        # Record intensity on a regular grid
        time_grid = np.arange(0, T, dt)
        intensities = np.array([self.intensity(t) for t in time_grid])

        return events, time_grid, intensities

    def reset(self):
        """Clear event history."""
        self.event_times.clear()
        self.intensity_history.clear()


class BidAskHawkes:
    """
    Bivariate Hawkes process for bid and ask order flow.

    Models both self-excitation (buy triggers more buys) and
    cross-excitation (buy triggers sells, capturing market makers
    adjusting quotes after order flow).

    λ_bid(t)  = μ_bid  + Σ α_bb exp(-β_bb Δt) [bid→bid]  + Σ α_ab exp(-β_ab Δt) [ask→bid]
    λ_ask(t)  = μ_ask  + Σ α_aa exp(-β_aa Δt) [ask→ask]  + Σ α_ba exp(-β_ba Δt) [bid→ask]
    """

    @dataclass
    class BiParams:
        mu_bid: float = 1.0
        mu_ask: float = 1.0
        alpha_self: float = 0.3    # Self-excitation
        alpha_cross: float = 0.1   # Cross-excitation
        beta_self: float = 1.0     # Self-decay
        beta_cross: float = 0.5    # Cross-decay

    def __init__(self, params: Optional["BidAskHawkes.BiParams"] = None,
                 seed: Optional[int] = None):
        self.params = params or self.BiParams()
        self.rng = np.random.default_rng(seed)
        self.bid_times: List[float] = []
        self.ask_times: List[float] = []

    def intensity_bid(self, t: float) -> float:
        lam = self.params.mu_bid
        for ti in self.bid_times:
            if ti < t:
                lam += self.params.alpha_self * np.exp(-self.params.beta_self * (t - ti))
        for ti in self.ask_times:
            if ti < t:
                lam += self.params.alpha_cross * np.exp(-self.params.beta_cross * (t - ti))
        return lam

    def intensity_ask(self, t: float) -> float:
        lam = self.params.mu_ask
        for ti in self.ask_times:
            if ti < t:
                lam += self.params.alpha_self * np.exp(-self.params.beta_self * (t - ti))
        for ti in self.bid_times:
            if ti < t:
                lam += self.params.alpha_cross * np.exp(-self.params.beta_cross * (t - ti))
        return lam

    def simulate(self, T: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate bid and ask order arrivals on [0, T].

        Uses superposition thinning: treat both streams as a single
        process and assign each event to bid or ask based on relative intensity.

        @returns (bid_times, ask_times)
        """
        self.bid_times = []
        self.ask_times = []
        t = 0.0

        while t < T:
            lam_bid = self.intensity_bid(t)
            lam_ask = self.intensity_ask(t)
            lam_total = lam_bid + lam_ask

            if lam_total <= 0:
                break

            # Inter-arrival from combined process
            dt = -np.log(self.rng.random()) / lam_total
            t += dt

            if t >= T:
                break

            # Thinning
            lam_bid_new = self.intensity_bid(t)
            lam_ask_new = self.intensity_ask(t)
            lam_total_new = lam_bid_new + lam_ask_new

            if self.rng.random() * lam_total <= lam_total_new:
                # Accept — assign to bid or ask
                if self.rng.random() * lam_total_new <= lam_bid_new:
                    self.bid_times.append(t)
                else:
                    self.ask_times.append(t)

        return np.array(self.bid_times), np.array(self.ask_times)

    def reset(self):
        self.bid_times.clear()
        self.ask_times.clear()
