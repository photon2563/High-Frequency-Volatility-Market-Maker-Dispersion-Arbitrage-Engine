#pragma once
/// @file vega_neutralizer.hpp
/// @brief Vega neutralization via variance swap / ATM straddle layering.
///
/// Isolates Gamma scalping profits by neutralizing Vega exposure.
/// Ensures that P&L reflects purely realized-vs-implied vol spread,
/// immune to mark-to-market losses from IV fluctuations.

#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace davinci {
namespace hedging {

// ─────────────────────────────────────────────────────────────────────────────
// Result structs
// ─────────────────────────────────────────────────────────────────────────────

struct NeutralizeResult {
    double portfolio_vega_before;   // Net Vega before neutralization
    double portfolio_vega_after;    // Net Vega after (should be ≈ 0)
    double units_required;          // Hedge instrument units needed
    double residual_vanna;          // Remaining Vanna exposure (diagnostic)
    double residual_volga;          // Remaining Volga exposure (diagnostic)
    bool   is_neutralized;          // True if |vega_after| < tolerance
};

struct PortfolioGreeks {
    double delta;
    double gamma;
    double vega;
    double theta;
    double vanna;
    double volga;
};

// ─────────────────────────────────────────────────────────────────────────────
// Variance swap Vega
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Vega of a unit-notional variance swap.
///
/// A variance swap has constant Vega per unit notional:
///   ν_varswap = 2 · σ_strike · T
///
/// where σ_strike is the variance swap strike (in vol terms).
///
/// @param sigma_strike Variance swap strike (annualized vol, e.g., 0.20)
/// @param T Time to maturity (years)
/// @returns Vega per unit notional of variance swap
inline double variance_swap_vega(double sigma_strike, double T) {
    return 2.0 * sigma_strike * T;
}

/// Compute units of variance swap needed to neutralize portfolio Vega.
///
/// N_var = -ν_portfolio / ν_unit_varswap
///
/// Negative portfolio Vega → buy variance (go long vol)
/// Positive portfolio Vega → sell variance (go short vol)
///
/// @param portfolio_vega Net portfolio Vega
/// @param sigma_strike Variance swap strike
/// @param T Time to maturity
/// @returns Number of variance swap units (positive = long, negative = short)
inline double varswap_units_for_neutralization(
    double portfolio_vega, double sigma_strike, double T)
{
    double unit_vega = variance_swap_vega(sigma_strike, T);
    if (std::abs(unit_vega) < 1e-15) {
        throw std::invalid_argument("VegaNeutralizer: variance swap vega too small");
    }
    return -portfolio_vega / unit_vega;
}

// ─────────────────────────────────────────────────────────────────────────────
// ATM straddle Vega
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Vega of an ATM straddle (call + put).
///
/// ATM straddle Vega = 2 · ATM call Vega = 2 · S · √T · φ(0) = 2 · S · √T / √(2π)
///
/// At-the-money: d1 ≈ σ√T/2, but for ATM d1 ≈ 0 → φ(d1) ≈ 1/√(2π)
///
/// @param S Spot price
/// @param T Time to maturity
/// @param sigma Implied vol (used for more precise ATM Vega)
/// @returns Vega per unit straddle
inline double atm_straddle_vega(double S, double T, double sigma = 0.0) {
    static constexpr double inv_sqrt_2pi = 0.3989422804014327;

    if (sigma > 0.0) {
        // More precise: use actual d1 for ATM (K = S · e^(rT) ≈ S for short T)
        double sqrt_T = std::sqrt(T);
        double d1 = 0.5 * sigma * sqrt_T; // ATM approximation
        double nd1 = inv_sqrt_2pi * std::exp(-0.5 * d1 * d1);
        return 2.0 * S * nd1 * sqrt_T;
    }

    // Rough ATM: φ(0) = 1/√(2π)
    return 2.0 * S * std::sqrt(T) * inv_sqrt_2pi;
}

/// Compute units of ATM straddles needed to neutralize portfolio Vega.
///
/// N_straddle = -ν_portfolio / ν_ATM_straddle
///
/// @param portfolio_vega Net portfolio Vega
/// @param S Spot price
/// @param T Time to maturity
/// @param sigma Implied vol (for precise ATM Vega)
/// @returns Number of straddle units
inline double straddle_units_for_neutralization(
    double portfolio_vega, double S, double T, double sigma = 0.0)
{
    double unit_vega = atm_straddle_vega(S, T, sigma);
    if (std::abs(unit_vega) < 1e-15) {
        throw std::invalid_argument("VegaNeutralizer: straddle vega too small");
    }
    return -portfolio_vega / unit_vega;
}

// ─────────────────────────────────────────────────────────────────────────────
// Full neutralization engine
// ─────────────────────────────────────────────────────────────────────────────

/// Neutralize portfolio Vega using variance swaps.
///
/// @param greeks Current portfolio Greeks
/// @param sigma_strike Variance swap strike
/// @param T Time to maturity
/// @param tolerance Max acceptable residual Vega (default 1e-6)
/// @returns NeutralizeResult with hedge sizing and diagnostics
inline NeutralizeResult neutralize_with_varswap(
    const PortfolioGreeks& greeks,
    double sigma_strike, double T,
    double tolerance = 1e-6)
{
    double n_var = varswap_units_for_neutralization(greeks.vega, sigma_strike, T);
    double residual_vega = greeks.vega + n_var * variance_swap_vega(sigma_strike, T);

    // Variance swap has ~zero Vanna/Volga (constant vega), so residuals remain
    return NeutralizeResult{
        greeks.vega,
        residual_vega,
        n_var,
        greeks.vanna,       // Unchanged — varswap doesn't hedge Vanna
        greeks.volga,       // Unchanged — varswap doesn't hedge Volga
        std::abs(residual_vega) < tolerance
    };
}

/// Neutralize portfolio Vega using ATM straddles.
///
/// Note: straddles also affect Gamma, Delta, Vanna, and Volga — unlike variance swaps.
/// This is a simpler but less clean hedging instrument.
///
/// @param greeks Current portfolio Greeks
/// @param S Spot price
/// @param T Time to maturity
/// @param sigma Implied vol
/// @param straddle_gamma Gamma per unit straddle (for impact assessment)
/// @param straddle_vanna Vanna per unit straddle
/// @param straddle_volga Volga per unit straddle
/// @param tolerance Max acceptable residual Vega
/// @returns NeutralizeResult with hedge sizing and diagnostics
inline NeutralizeResult neutralize_with_straddle(
    const PortfolioGreeks& greeks,
    double S, double T, double sigma,
    double straddle_gamma = 0.0,
    double straddle_vanna = 0.0,
    double straddle_volga = 0.0,
    double tolerance = 1e-6)
{
    double n_straddle = straddle_units_for_neutralization(greeks.vega, S, T, sigma);
    double unit_vega = atm_straddle_vega(S, T, sigma);
    double residual_vega = greeks.vega + n_straddle * unit_vega;

    // Straddle impacts on higher-order Greeks
    double residual_vanna = greeks.vanna + n_straddle * straddle_vanna;
    double residual_volga = greeks.volga + n_straddle * straddle_volga;

    return NeutralizeResult{
        greeks.vega,
        residual_vega,
        n_straddle,
        residual_vanna,
        residual_volga,
        std::abs(residual_vega) < tolerance
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// Gamma scalping P&L estimation
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate daily Gamma scalping P&L.
///
/// When Vega is neutralized, the daily P&L is approximately:
///   PnL ≈ 0.5 · Γ · S² · (σ²_realized_daily - σ²_implied_daily)
///
/// This is the "pure convexity" profit isolated from IV fluctuations.
///
/// @param gamma Portfolio Gamma
/// @param S Spot price
/// @param sigma_realized Annualized realized vol
/// @param sigma_implied Annualized implied vol
/// @param dt Time step (1/252 for daily)
/// @returns Estimated P&L per time step
inline double gamma_scalping_pnl(
    double gamma, double S,
    double sigma_realized, double sigma_implied,
    double dt = 1.0 / 252.0)
{
    double var_spread = sigma_realized * sigma_realized - sigma_implied * sigma_implied;
    return 0.5 * gamma * S * S * var_spread * dt;
}

} // namespace hedging
} // namespace davinci
