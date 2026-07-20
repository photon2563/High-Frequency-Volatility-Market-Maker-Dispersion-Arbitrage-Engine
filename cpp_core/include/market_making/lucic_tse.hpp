#pragma once
/// @file lucic_tse.hpp
/// @brief Volatility-arbitrage-adjusted market making (Lucic & Tse framework).
///
/// Extends Avellaneda-Stoikov by decomposing optimal quotes into three components:
///   1. Expected volatility arbitrage profit (realized vs implied)
///   2. Demand elasticity adjustment
///   3. Inventory risk penalty (scaled by Greeks)
///
/// The market maker simultaneously acts as liquidity provider AND volatility arbitrageur.

#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace davinci {
namespace market_making {

// ─────────────────────────────────────────────────────────────────────────────
// Result struct
// ─────────────────────────────────────────────────────────────────────────────

struct LTQuote {
    double bid;
    double ask;
    double spread;

    // Decomposition for attribution
    double vol_arb_component;     // Component 1: expected vol arb profit
    double elasticity_component;  // Component 2: demand elasticity
    double inventory_component;   // Component 3: inventory risk penalty
};

// ─────────────────────────────────────────────────────────────────────────────
// Model parameters
// ─────────────────────────────────────────────────────────────────────────────

struct LTParams {
    double gamma;          // Risk aversion coefficient
    double kappa;          // Order arrival intensity decay
    double sigma_realized; // Expected realized volatility (trader's view)
    double sigma_implied;  // Current market implied volatility
    double xi;             // Vol-of-vol parameter (stochastic vol amplitude)

    void validate() const {
        if (gamma <= 0.0)          throw std::invalid_argument("LT: gamma must be positive");
        if (kappa <= 0.0)          throw std::invalid_argument("LT: kappa must be positive");
        if (sigma_realized <= 0.0) throw std::invalid_argument("LT: sigma_realized must be positive");
        if (sigma_implied <= 0.0)  throw std::invalid_argument("LT: sigma_implied must be positive");
        if (xi < 0.0)             throw std::invalid_argument("LT: xi must be non-negative");
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Component computations
// ─────────────────────────────────────────────────────────────────────────────

/// Component 1: Expected volatility arbitrage profit.
///
/// V_arb = 0.5 · Γ · S² · (σ²_realized - σ²_implied) · Δt
///
/// When σ_realized > σ_implied: positive profit expectation from Gamma scalping.
/// When σ_realized < σ_implied: negative (options are overpriced by market).
///
/// @param gamma_greek Option's Gamma (∂²V/∂S²)
/// @param S Spot price
/// @param params LT model parameters
/// @param dt Time step (in years)
/// @returns Expected vol arb profit per time step
inline double vol_arb_profit(
    double gamma_greek, double S,
    const LTParams& params, double dt)
{
    double var_spread = params.sigma_realized * params.sigma_realized
                      - params.sigma_implied  * params.sigma_implied;
    return 0.5 * gamma_greek * S * S * var_spread * dt;
}

/// Component 2: Demand elasticity adjustment.
///
/// D_elast = (1/κ) · ln(1 + κ/γ)
///
/// Captures how sensitive order flow is to quoted spread width.
/// Higher κ → orders are more sensitive to spread → narrower quotes optimal.
///
/// @param params LT model parameters
/// @returns Elasticity-adjusted half-spread
inline double demand_elasticity(const LTParams& params) {
    return (1.0 / params.kappa) * std::log(1.0 + params.kappa / params.gamma);
}

/// Component 3: Inventory risk penalty.
///
/// R_inv = γ · q · [Γ · S² · σ² + ν² · ξ²] · (T - t)
///
/// Penalizes holding inventory proportional to:
///   - Gamma risk: second-order price sensitivity
///   - Vega risk: vol-of-vol uncertainty (ν² · ξ²)
///
/// @param inventory Current inventory position
/// @param gamma_greek Option's Gamma
/// @param vega_greek Option's Vega
/// @param S Spot price
/// @param params LT model parameters
/// @param time_remaining Time to horizon
/// @returns Inventory risk penalty (positive for long, negative for short inventory)
inline double inventory_risk_penalty(
    double inventory, double gamma_greek, double vega_greek,
    double S, const LTParams& params, double time_remaining)
{
    double gamma_risk = gamma_greek * S * S * params.sigma_implied * params.sigma_implied;
    double vega_risk  = vega_greek * vega_greek * params.xi * params.xi;
    return params.gamma * inventory * (gamma_risk + vega_risk) * time_remaining;
}

// ─────────────────────────────────────────────────────────────────────────────
// Full Lucic-Tse quoting
// ─────────────────────────────────────────────────────────────────────────────

/// Compute Lucic-Tse optimal bid and ask quotes with full component decomposition.
///
/// bid = s + V_arb - D_elast - R_inv
/// ask = s + V_arb + D_elast - R_inv
///
/// When σ_realized = σ_implied (no vol view), this reduces approximately to
/// Avellaneda-Stoikov with the elasticity term dominating.
///
/// @param mid_price Current mid-price
/// @param inventory Current inventory
/// @param gamma_greek Option's Gamma
/// @param vega_greek Option's Vega
/// @param S Spot price
/// @param params LT model parameters
/// @param time_remaining Time to horizon (in years)
/// @param dt Time step for vol arb calculation
/// @returns LTQuote with bid, ask, and component breakdown
inline LTQuote compute_quotes(
    double mid_price, double inventory,
    double gamma_greek, double vega_greek,
    double S, const LTParams& params,
    double time_remaining, double dt)
{
    params.validate();

    double v_arb   = vol_arb_profit(gamma_greek, S, params, dt);
    double d_elast = demand_elasticity(params);
    double r_inv   = inventory_risk_penalty(
                        inventory, gamma_greek, vega_greek,
                        S, params, time_remaining);

    // Reservation price: mid + vol_arb - inventory_penalty
    double reservation = mid_price + v_arb - r_inv;

    double bid = reservation - d_elast;
    double ask = reservation + d_elast;

    // Ensure bid < ask (sanity — should always hold if d_elast > 0)
    if (bid >= ask) {
        double mid = 0.5 * (bid + ask);
        bid = mid - 0.01;  // Minimum 1 cent spread
        ask = mid + 0.01;
    }

    return LTQuote{
        bid,
        ask,
        ask - bid,
        v_arb,
        d_elast,
        r_inv
    };
}

/// Check if the model predicts the current market is in a "vol arb" regime.
///
/// @returns true if |σ_realized - σ_implied| > threshold
inline bool is_vol_arb_opportunity(const LTParams& params, double threshold = 0.02) {
    return std::abs(params.sigma_realized - params.sigma_implied) > threshold;
}

/// Compute the variance risk premium (VRP).
/// VRP = σ²_implied - σ²_realized (typically positive due to risk premium).
inline double variance_risk_premium(const LTParams& params) {
    return params.sigma_implied * params.sigma_implied
         - params.sigma_realized * params.sigma_realized;
}

} // namespace market_making
} // namespace davinci
