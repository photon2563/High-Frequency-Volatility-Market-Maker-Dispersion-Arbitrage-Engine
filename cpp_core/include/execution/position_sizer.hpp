#pragma once
/// @file position_sizer.hpp
/// @brief Lot-size rounding, futures proxy hedge, and P&L approximation.
///
/// Handles the practical microstructure constraints of execution:
///   - Round theoretical positions to official exchange lot sizes
///   - Compute residual delta from rounding mismatch
///   - Generate futures proxy hedge orders to neutralize residuals
///   - First-order Vega expansion for P&L approximation

#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace davinci {
namespace execution {

// ─────────────────────────────────────────────────────────────────────────────
// Position sizing result
// ─────────────────────────────────────────────────────────────────────────────

struct SizedPosition {
    double theoretical_contracts;  // Exact theoretical position
    int    rounded_contracts;      // Lot-size-rounded position
    double residual_delta;         // Delta mismatch from rounding
    int    lot_size;               // Exchange lot size used
    double rounding_error;         // theoretical - rounded (in contracts)
};

struct FuturesHedge {
    int    contracts;              // Number of futures contracts
    double notional;               // Dollar notional of the hedge
    double residual_delta_after;   // Remaining delta after futures hedge
    double contract_multiplier;    // Futures contract multiplier
};

struct PnLApproximation {
    double vega_pnl;       // P&L from vol move: ν · Δσ
    double delta_pnl;      // P&L from spot move: Δ · ΔS
    double gamma_pnl;      // P&L from convexity: 0.5 · Γ · ΔS²
    double theta_pnl;      // P&L from time decay: Θ · Δt
    double total_pnl;      // Sum of all components
};

// ─────────────────────────────────────────────────────────────────────────────
// Lot-size rounding
// ─────────────────────────────────────────────────────────────────────────────

/// Round a theoretical position to the nearest valid lot size.
///
/// @param theoretical_contracts Exact theoretical position (fractional allowed)
/// @param lot_size Exchange lot size (e.g., 100 for US equity options)
/// @param delta_per_contract Delta per option contract (for residual calc)
/// @returns SizedPosition with rounded contracts and residual delta
inline SizedPosition round_to_lots(
    double theoretical_contracts,
    int lot_size = 100,
    double delta_per_contract = 1.0)
{
    if (lot_size <= 0) throw std::invalid_argument("PositionSizer: lot_size must be positive");

    // Round to nearest lot
    int rounded = static_cast<int>(std::round(theoretical_contracts / lot_size)) * lot_size;
    double rounding_error = theoretical_contracts - static_cast<double>(rounded);
    double residual_delta = rounding_error * delta_per_contract;

    return SizedPosition{
        theoretical_contracts,
        rounded,
        residual_delta,
        lot_size,
        rounding_error
    };
}

/// Conservative rounding: always round towards zero (less aggressive positioning).
inline SizedPosition round_to_lots_conservative(
    double theoretical_contracts,
    int lot_size = 100,
    double delta_per_contract = 1.0)
{
    if (lot_size <= 0) throw std::invalid_argument("PositionSizer: lot_size must be positive");

    int rounded;
    if (theoretical_contracts >= 0) {
        rounded = static_cast<int>(std::floor(theoretical_contracts / lot_size)) * lot_size;
    } else {
        rounded = static_cast<int>(std::ceil(theoretical_contracts / lot_size)) * lot_size;
    }

    double rounding_error = theoretical_contracts - static_cast<double>(rounded);
    double residual_delta = rounding_error * delta_per_contract;

    return SizedPosition{
        theoretical_contracts,
        rounded,
        residual_delta,
        lot_size,
        rounding_error
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// Futures proxy hedge
// ─────────────────────────────────────────────────────────────────────────────

/// Compute futures contracts needed to hedge residual delta from lot-size rounding.
///
/// @param residual_delta Delta to hedge (shares equivalent)
/// @param futures_multiplier Contract multiplier (e.g., 50 for E-mini S&P)
/// @param futures_delta Delta per futures contract (usually ≈ 1.0 × multiplier)
/// @param S Current spot price (for notional calculation)
/// @returns FuturesHedge with contract count and remaining exposure
inline FuturesHedge compute_futures_hedge(
    double residual_delta,
    double futures_multiplier = 50.0,
    double futures_delta = 1.0,
    double S = 0.0)
{
    double delta_per_contract = futures_delta * futures_multiplier;
    if (std::abs(delta_per_contract) < 1e-15) {
        throw std::invalid_argument("PositionSizer: futures delta per contract too small");
    }

    double exact_contracts = -residual_delta / delta_per_contract;
    int rounded_contracts = static_cast<int>(std::round(exact_contracts));

    double hedged_delta = residual_delta + rounded_contracts * delta_per_contract;
    double notional = std::abs(rounded_contracts) * futures_multiplier * S;

    return FuturesHedge{
        rounded_contracts,
        notional,
        hedged_delta,
        futures_multiplier
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// P&L approximation (first-order Taylor expansion)
// ─────────────────────────────────────────────────────────────────────────────

/// Approximate P&L using first-order Greek expansion.
///
/// PnL ≈ Δ·ΔS + ½·Γ·ΔS² + ν·Δσ + Θ·Δt
///
/// This is the workhorse approximation for rapid cross-verification.
/// Must match the full BS reprice to within reasonable tolerance.
///
/// @param delta Portfolio delta
/// @param gamma Portfolio gamma
/// @param vega Portfolio vega
/// @param theta Portfolio theta
/// @param dS Spot price change
/// @param d_sigma Vol change (in absolute terms, e.g., 0.01 = 1 vol point)
/// @param dt Time elapsed (in years)
/// @returns PnLApproximation with component breakdown
inline PnLApproximation approximate_pnl(
    double delta, double gamma, double vega, double theta,
    double dS, double d_sigma = 0.0, double dt = 0.0)
{
    double delta_pnl = delta * dS;
    double gamma_pnl = 0.5 * gamma * dS * dS;
    double vega_pnl  = vega * d_sigma;
    double theta_pnl = theta * dt;

    return PnLApproximation{
        vega_pnl,
        delta_pnl,
        gamma_pnl,
        theta_pnl,
        delta_pnl + gamma_pnl + vega_pnl + theta_pnl
    };
}

/// ATM straddle Vega for quick P&L cross-verification.
///
/// ν_ATM ≈ S · √T / √(2π)
///
/// This is the "mental math" approximation for Vega.
inline double quick_atm_vega(double S, double T) {
    static constexpr double inv_sqrt_2pi = 0.3989422804014327;
    return S * std::sqrt(T) * inv_sqrt_2pi;
}

/// ATM call price approximation (Brenner-Subrahmanyam).
///
/// C_ATM ≈ 0.4 · S · σ · √T
///
/// Accurate to within ~2% for ATM options. Used for sanity checks.
inline double quick_atm_call(double S, double sigma, double T) {
    return 0.4 * S * sigma * std::sqrt(T);
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispersion position sizing
// ─────────────────────────────────────────────────────────────────────────────

/// Compute position sizes for a dispersion trade (index vs constituents).
///
/// The key constraint: Vega-weighted position so that the trade is
/// net Vega-neutral between index and constituents.
///
/// @param index_vega Vega per unit of index straddle
/// @param constituent_vegas Vector of Vega per unit of each constituent straddle
/// @param constituent_weights Market-cap weights
/// @param target_index_notional Target notional for the index leg
/// @param lot_size Exchange lot size
/// @returns Vector of sized positions (index first, then constituents)
inline std::vector<SizedPosition> size_dispersion_trade(
    double index_vega,
    const std::vector<double>& constituent_vegas,
    const std::vector<double>& constituent_weights,
    double target_index_notional,
    int lot_size = 100)
{
    size_t n = constituent_vegas.size();
    if (n != constituent_weights.size()) {
        throw std::invalid_argument("PositionSizer: vegas and weights size mismatch");
    }

    std::vector<SizedPosition> positions;

    // Index leg: sell straddles
    double index_contracts = -target_index_notional / (index_vega > 0 ? index_vega : 1.0);
    positions.push_back(round_to_lots(index_contracts, lot_size));

    // Constituent legs: buy straddles, weighted by market cap
    // Total constituent Vega must offset index Vega
    double total_vega_target = -positions[0].rounded_contracts * index_vega;

    for (size_t i = 0; i < n; ++i) {
        if (constituent_vegas[i] > 1e-15) {
            double target = (constituent_weights[i] * total_vega_target) / constituent_vegas[i];
            positions.push_back(round_to_lots(target, lot_size));
        } else {
            positions.push_back(SizedPosition{0.0, 0, 0.0, lot_size, 0.0});
        }
    }

    return positions;
}

} // namespace execution
} // namespace davinci
