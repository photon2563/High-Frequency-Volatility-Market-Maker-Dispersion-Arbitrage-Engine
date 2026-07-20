#pragma once
/// @file leland.hpp
/// @brief Leland model for delta hedging under transaction costs.
///
/// Modifies Black-Scholes effective volatility to absorb frictional costs.
/// Key output: Leland number Le = √(2/π) · C / (σ√Δt)
/// This determines the no-trade band around the theoretical delta.

#include <cmath>
#include <stdexcept>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace davinci {
namespace hedging {

// ─────────────────────────────────────────────────────────────────────────────
// Leland model core
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Leland number.
///
/// Le = √(2/π) · C / (σ · √Δt)
///
/// Relates proportional transaction cost C to the option's volatility
/// and discrete rebalancing interval Δt.
///
/// Le < 1: hedging cost manageable, BS delta hedge approximately valid
/// Le ≈ 1: transaction costs dominate; significant modification needed
/// Le > 1: hedging extremely costly; consider wider bands or proxy hedging
///
/// @param txn_cost Proportional transaction cost (e.g., 0.001 = 10 bps)
/// @param sigma Asset volatility (annualized)
/// @param dt Rebalancing interval (in years, e.g., 1/252 for daily)
/// @returns Leland number
inline double leland_number(double txn_cost, double sigma, double dt) {
    if (sigma <= 0.0) throw std::invalid_argument("Leland: sigma must be positive");
    if (dt <= 0.0)    throw std::invalid_argument("Leland: dt must be positive");
    if (txn_cost < 0.0) throw std::invalid_argument("Leland: txn_cost must be non-negative");

    static constexpr double sqrt_2_over_pi = 0.7978845608028654; // √(2/π)
    return sqrt_2_over_pi * txn_cost / (sigma * std::sqrt(dt));
}

/// Compute the effective (modified) volatility under the Leland framework.
///
/// σ̂² = σ² · (1 + Le · sign(Γ))
///
/// - Long Gamma (Γ > 0): σ̂ > σ — absorbs hedging costs into higher effective vol
/// - Short Gamma (Γ < 0): σ̂ < σ — lower effective vol (hedging adds value)
///
/// @param sigma True asset volatility
/// @param le Leland number
/// @param gamma_sign Sign of portfolio Gamma (+1 for long, -1 for short)
/// @returns Modified effective volatility
inline double effective_volatility(double sigma, double le, double gamma_sign) {
    double sign = (gamma_sign >= 0.0) ? 1.0 : -1.0;
    double effective_var = sigma * sigma * (1.0 + le * sign);

    // Safeguard: effective variance must be positive
    if (effective_var <= 0.0) {
        // Short gamma with Le > 1 → hedging is catastrophically expensive
        // Return minimal positive vol
        return 0.001;
    }
    return std::sqrt(effective_var);
}

// ─────────────────────────────────────────────────────────────────────────────
// No-trade band computation
// ─────────────────────────────────────────────────────────────────────────────

struct HedgeBand {
    double delta_target;   // Theoretical Black-Scholes delta
    double band_lower;     // Lower bound of no-trade zone
    double band_upper;     // Upper bound of no-trade zone
    double band_width;     // Width of the no-trade zone
    double leland_number;  // Le for diagnostics
    double effective_vol;  // Modified σ̂
};

/// Compute the no-trade hedging band around the theoretical delta.
///
/// The band width scales with the Leland number and Gamma:
///   ε = √(Le · Δt · |Γ| · S²) · scaling_factor
///
/// Simplified practical band:
///   ε ≈ (3·C / σ)^(1/3) · |Γ|^(1/3) · S^(2/3) · Δt^(1/6)
///
/// @param delta_target BS theoretical delta
/// @param gamma_value Portfolio Gamma (absolute)
/// @param S Spot price
/// @param txn_cost Proportional transaction cost
/// @param sigma Volatility
/// @param dt Rebalancing interval
/// @returns HedgeBand struct
inline HedgeBand compute_hedge_band(
    double delta_target, double gamma_value,
    double S, double txn_cost, double sigma, double dt)
{
    double le = leland_number(txn_cost, sigma, dt);
    double gamma_sign = (gamma_value >= 0.0) ? 1.0 : -1.0;
    double eff_vol = effective_volatility(sigma, le, gamma_sign);

    // Whalley-Wilmott asymptotic band width (proportional transaction cost model)
    // ε ≈ (3 · C · Δt · S² · |Γ| / 2)^(1/3)  [Zakamouline & Koekebakker]
    double abs_gamma = std::abs(gamma_value);
    double band_half_width = 0.0;

    if (abs_gamma > 1e-15 && S > 0.0) {
        double inner = 1.5 * txn_cost * dt * S * S * abs_gamma;
        band_half_width = std::cbrt(inner);
    }

    return HedgeBand{
        delta_target,
        delta_target - band_half_width,
        delta_target + band_half_width,
        2.0 * band_half_width,
        le,
        eff_vol
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// Hedge decision logic
// ─────────────────────────────────────────────────────────────────────────────

enum class HedgeDecision {
    HOLD,          // Delta within band — no action
    REBALANCE,     // Delta outside band — rebalance to target
    PARTIAL        // Delta near band edge — partial rebalance
};

struct HedgeOrder {
    HedgeDecision decision;
    double shares_to_trade;   // Positive = buy, negative = sell
    double estimated_cost;    // txn_cost * |shares| * S
};

/// Evaluate whether to rebalance based on current delta vs. hedge band.
///
/// @param current_delta Current portfolio delta (shares held)
/// @param band Computed hedge band
/// @param S Spot price
/// @param txn_cost Transaction cost for cost estimation
/// @returns HedgeOrder with decision and required trade
inline HedgeOrder evaluate_hedge(
    double current_delta, const HedgeBand& band,
    double S, double txn_cost)
{
    HedgeOrder order{};

    if (current_delta >= band.band_lower && current_delta <= band.band_upper) {
        // Within band — hold
        order.decision = HedgeDecision::HOLD;
        order.shares_to_trade = 0.0;
        order.estimated_cost = 0.0;
    } else {
        // Outside band — rebalance to target
        order.decision = HedgeDecision::REBALANCE;
        order.shares_to_trade = band.delta_target - current_delta;
        order.estimated_cost = txn_cost * std::abs(order.shares_to_trade) * S;
    }

    return order;
}

// ─────────────────────────────────────────────────────────────────────────────
// Proxy hedge risk-adjusted value
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate the risk-adjusted value of proxy hedging.
///
/// When the underlying is illiquid, we can hedge with a correlated liquid substitute.
/// The trade-off is basis risk (ρ < 1) vs. transaction cost savings.
///
/// Risk-adjusted benefit = (1 - ρ²) · σ² · Γ² · S² · Δt  [basis risk cost]
///                       vs. C_direct - C_proxy                [txn cost saving]
///
/// @param rho Correlation between asset and proxy
/// @param sigma_asset Volatility of the hedged asset
/// @param sigma_proxy Volatility of the proxy instrument
/// @param gamma_value Portfolio Gamma
/// @param S Spot price
/// @param txn_cost_direct Transaction cost for direct hedge
/// @param txn_cost_proxy Transaction cost for proxy
/// @param dt Rebalancing interval
/// @returns > 0 means proxy hedge is preferable
inline double proxy_hedge_benefit(
    double rho, double sigma_asset, double sigma_proxy,
    double gamma_value, double S,
    double txn_cost_direct, double txn_cost_proxy, double dt)
{
    // Basis risk cost (variance of residual)
    double basis_risk = (1.0 - rho * rho) * sigma_asset * sigma_asset
                       * gamma_value * gamma_value * S * S * dt;

    // Transaction cost saving
    double le_direct = leland_number(txn_cost_direct, sigma_asset, dt);
    double le_proxy  = leland_number(txn_cost_proxy, sigma_proxy, dt);
    double txn_saving = le_direct - le_proxy; // Lower Le = cheaper hedging

    // Positive = proxy is better (savings outweigh basis risk)
    return txn_saving - basis_risk;
}

} // namespace hedging
} // namespace davinci
