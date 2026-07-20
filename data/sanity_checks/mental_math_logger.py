"""
mental_math_logger.py — Side-by-side comparison of algorithmic vs mental-math approximations.

Module 6 cross-verification: ensures outputs are sanity-checked against
rapid mental approximations that a prop trader would use on-the-fly.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class SanityCheckResult:
    """Result of a single sanity check."""
    label: str           # What we're checking
    exact_value: float   # Algorithm output
    mental_math: float   # Quick approximation
    abs_error: float     # |exact - mental_math|
    rel_error: float     # |exact - mental_math| / |exact| (if exact != 0)
    passed: bool         # Within acceptable tolerance


class MentalMathLogger:
    """
    Cross-verification logger: prints side-by-side comparisons of
    exact algorithmic outputs vs. rapid mental-math approximations.

    Key approximations:
      - ATM call ≈ 0.4 · S · σ · √T  (Brenner-Subrahmanyam)
      - ATM Vega ≈ S · √T / √(2π)
      - ATM Gamma ≈ φ(0) / (S · σ · √T) = 1/(S · σ · √T · √(2π))
      - Leland Le ≈ 0.8 · C / (σ · √Δt)
      - Variance swap Vega ≈ 2 · σ_K · T
    """

    INV_SQRT_2PI = 0.3989422804014327

    def __init__(self, tolerance: float = 0.10):
        """
        @param tolerance: Maximum acceptable relative error for "pass" (default 10%)
        """
        self.tolerance = tolerance
        self.checks: List[SanityCheckResult] = []

    def _record(self, label: str, exact: float, approx: float) -> SanityCheckResult:
        abs_err = abs(exact - approx)
        rel_err = abs_err / abs(exact) if abs(exact) > 1e-15 else float("inf")
        passed = rel_err < self.tolerance
        result = SanityCheckResult(label, exact, approx, abs_err, rel_err, passed)
        self.checks.append(result)
        return result

    # ── ATM call price ──
    def check_atm_call(self, exact_price: float, S: float, sigma: float, T: float) -> SanityCheckResult:
        """ATM call ≈ 0.4 · S · σ · √T"""
        approx = 0.4 * S * sigma * np.sqrt(T)
        return self._record("ATM Call Price", exact_price, approx)

    # ── ATM Vega ──
    def check_atm_vega(self, exact_vega: float, S: float, T: float) -> SanityCheckResult:
        """ATM Vega ≈ S · √T / √(2π)"""
        approx = S * np.sqrt(T) * self.INV_SQRT_2PI
        return self._record("ATM Vega", exact_vega, approx)

    # ── ATM Gamma ──
    def check_atm_gamma(self, exact_gamma: float, S: float, sigma: float, T: float) -> SanityCheckResult:
        """ATM Gamma ≈ 1 / (S · σ · √T · √(2π))"""
        approx = 1.0 / (S * sigma * np.sqrt(T) * np.sqrt(2 * np.pi))
        return self._record("ATM Gamma", exact_gamma, approx)

    # ── Leland number ──
    def check_leland(self, exact_le: float, C: float, sigma: float, dt: float) -> SanityCheckResult:
        """Le ≈ 0.8 · C / (σ · √Δt)"""
        approx = 0.8 * C / (sigma * np.sqrt(dt))
        return self._record("Leland Number", exact_le, approx)

    # ── Variance swap Vega ──
    def check_varswap_vega(self, exact_vega: float, sigma_K: float, T: float) -> SanityCheckResult:
        """Var swap Vega ≈ 2 · σ_K · T"""
        approx = 2.0 * sigma_K * T
        return self._record("VarSwap Vega", exact_vega, approx)

    # ── Variance swap units for neutralization ──
    def check_varswap_units(self, exact_units: float, portfolio_vega: float,
                            sigma_K: float, T: float) -> SanityCheckResult:
        """N_var ≈ -portfolio_vega / (2 · σ_K · T)"""
        approx = -portfolio_vega / (2.0 * sigma_K * T)
        return self._record("VarSwap Hedge Units", exact_units, approx)

    # ── Dirty implied correlation ──
    def check_dirty_correlation(self, exact_rho: float, index_iv: float,
                                 weighted_constituent_iv: float) -> SanityCheckResult:
        """ρ_dirty ≈ (σ_idx / Σw_iσ_i)²"""
        approx = (index_iv / weighted_constituent_iv) ** 2 if weighted_constituent_iv > 0 else 0
        return self._record("Dirty Correlation", exact_rho, approx)

    # ── Gamma scalping P&L ──
    def check_gamma_pnl(self, exact_pnl: float, gamma: float, S: float,
                         sigma_r: float, sigma_i: float, dt: float = 1/252) -> SanityCheckResult:
        """PnL ≈ 0.5 · Γ · S² · (σ²_r - σ²_i) · Δt"""
        approx = 0.5 * gamma * S**2 * (sigma_r**2 - sigma_i**2) * dt
        return self._record("Gamma Scalping P&L", exact_pnl, approx)

    def report(self) -> str:
        """Generate a formatted report of all sanity checks."""
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║          MENTAL MATH CROSS-VERIFICATION REPORT             ║",
            "╠══════════════════════════════════════════════════════════════╣",
        ]
        for c in self.checks:
            status = "✓ PASS" if c.passed else "✗ FAIL"
            lines.append(
                f"║ {c.label:25s} │ Exact: {c.exact_value:12.6f} │ "
                f"Approx: {c.mental_math:12.6f} │ Err: {c.rel_error:6.2%} │ {status} ║"
            )
        lines.append("╚══════════════════════════════════════════════════════════════╝")

        n_pass = sum(1 for c in self.checks if c.passed)
        lines.append(f"\n  {n_pass}/{len(self.checks)} checks passed "
                     f"(tolerance: {self.tolerance:.0%})")

        return "\n".join(lines)

    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def reset(self):
        self.checks.clear()
