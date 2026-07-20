#pragma once
/// @file avellaneda_stoikov.hpp
/// @brief Inventory-aware optimal market making model (Avellaneda & Stoikov, 2008).
///
/// Computes reservation price and optimal spread based on:
///   - Current inventory position
///   - Asset volatility
///   - Risk aversion parameter
///   - Order arrival intensity
///   - Time to horizon

#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace davinci {
namespace market_making {

// ─────────────────────────────────────────────────────────────────────────────
// Result struct
// ─────────────────────────────────────────────────────────────────────────────

struct ASQuote {
    double bid;               // Optimal bid price
    double ask;               // Optimal ask price
    double reservation_price; // Risk-adjusted mid-price
    double optimal_spread;    // Total bid-ask spread
    double skew;              // Inventory-driven price adjustment
};

// ─────────────────────────────────────────────────────────────────────────────
// Model parameters
// ─────────────────────────────────────────────────────────────────────────────

struct ASParams {
    double gamma;   // Risk aversion coefficient (higher → tighter inventory control)
    double kappa;   // Order arrival intensity decay parameter
    double sigma;   // Asset volatility (annualized)
    double T;       // Time horizon (fraction of day, e.g., 1.0 = full day)

    void validate() const {
        if (gamma <= 0.0) throw std::invalid_argument("AS: gamma must be positive");
        if (kappa <= 0.0) throw std::invalid_argument("AS: kappa must be positive");
        if (sigma <= 0.0) throw std::invalid_argument("AS: sigma must be positive");
        if (T < 0.0)      throw std::invalid_argument("AS: T must be non-negative");
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Core quoting functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the reservation price.
///
/// r(s, q, t) = s - q · γ · σ² · (T - t)
///
/// @param mid_price Current mid-price of the asset
/// @param inventory Current inventory (positive = long, negative = short)
/// @param params Model parameters
/// @param time_elapsed Fraction of horizon elapsed (0 = start, T = end)
/// @returns Risk-adjusted reservation price
inline double reservation_price(
    double mid_price, double inventory,
    const ASParams& params, double time_elapsed = 0.0)
{
    double time_remaining = std::max(params.T - time_elapsed, 0.0);
    return mid_price - inventory * params.gamma * params.sigma * params.sigma * time_remaining;
}

/// Compute the optimal spread.
///
/// δ* = γ · σ² · (T - t) + (2/γ) · ln(1 + γ/κ)
///
/// @param params Model parameters
/// @param time_elapsed Fraction of horizon elapsed
/// @returns Optimal total spread
inline double optimal_spread(const ASParams& params, double time_elapsed = 0.0)
{
    double time_remaining = std::max(params.T - time_elapsed, 0.0);
    double inventory_component = params.gamma * params.sigma * params.sigma * time_remaining;
    double elasticity_component = (2.0 / params.gamma) * std::log(1.0 + params.gamma / params.kappa);
    return inventory_component + elasticity_component;
}

/// Compute optimal bid and ask quotes.
///
/// bid = reservation_price - δ*/2
/// ask = reservation_price + δ*/2
///
/// @param mid_price Current mid-price
/// @param inventory Current inventory
/// @param params Model parameters
/// @param time_elapsed Fraction of horizon elapsed
/// @returns ASQuote with bid, ask, reservation price, and spread
inline ASQuote compute_quotes(
    double mid_price, double inventory,
    const ASParams& params, double time_elapsed = 0.0)
{
    params.validate();

    double r = reservation_price(mid_price, inventory, params, time_elapsed);
    double delta = optimal_spread(params, time_elapsed);
    double half_delta = 0.5 * delta;

    // Inventory skew: how much the reservation price deviates from mid
    double skew = mid_price - r;

    return ASQuote{
        r - half_delta,    // bid
        r + half_delta,    // ask
        r,                 // reservation_price
        delta,             // optimal_spread
        skew               // skew
    };
}

/// Compute the probability of order arrival at a given distance from mid.
///
/// λ(δ) = A · exp(-κ · δ)
///
/// @param distance Distance from mid-price (half-spread)
/// @param A Base arrival intensity
/// @param kappa Decay parameter
/// @returns Arrival rate
inline double arrival_intensity(double distance, double A, double kappa) {
    return A * std::exp(-kappa * distance);
}

/// Compute expected fill probabilities for the current quotes.
///
/// @param quote Current AS quote
/// @param mid_price Mid-price
/// @param A Base arrival intensity
/// @param kappa Decay parameter
/// @param dt Time step
/// @returns {bid_fill_prob, ask_fill_prob}
inline std::pair<double, double> fill_probabilities(
    const ASQuote& quote, double mid_price,
    double A, double kappa, double dt)
{
    double bid_dist = mid_price - quote.bid;
    double ask_dist = quote.ask - mid_price;
    double lambda_bid = arrival_intensity(bid_dist, A, kappa);
    double lambda_ask = arrival_intensity(ask_dist, A, kappa);

    // Poisson probability of at least one fill in dt
    double p_bid = 1.0 - std::exp(-lambda_bid * dt);
    double p_ask = 1.0 - std::exp(-lambda_ask * dt);

    return {p_bid, p_ask};
}

} // namespace market_making
} // namespace davinci
