#pragma once
/// @file central_risk_book.hpp
/// @brief Central Risk Book: HJB-QVI-based limit-vs-market order hedging logic.
///
/// Implements the Muhle-Karbe, Wang & Webster framework where the delta hedge
/// is executed optimally via a mix of passive limit orders and aggressive market orders.
///
/// Three regions around the target delta:
///   Inner zone:  |Δ_drift| < ε_inner  →  HOLD (no action)
///   Middle zone: ε_inner ≤ |Δ_drift| < ε_outer  →  POST limit order
///   Outer zone:  |Δ_drift| ≥ ε_outer  →  AGGRESSIVE market order

#include <cmath>
#include <stdexcept>
#include <string>
#include <algorithm>

namespace davinci {
namespace hedging {

// ─────────────────────────────────────────────────────────────────────────────
// Enums and result structs
// ─────────────────────────────────────────────────────────────────────────────

enum class OrderType {
    HOLD,          // No action needed
    LIMIT_ORDER,   // Post passive limit order at theoretical delta
    MARKET_ORDER   // Aggressively cross the spread
};

struct CRBOrder {
    OrderType type;
    double shares;           // Shares to trade (positive = buy, negative = sell)
    double limit_price;      // Only meaningful for LIMIT_ORDER
    double urgency;          // 0.0 = patient, 1.0 = immediate
    std::string reason;      // Human-readable decision rationale
};

// ─────────────────────────────────────────────────────────────────────────────
// CRB parameters
// ─────────────────────────────────────────────────────────────────────────────

struct CRBParams {
    double adverse_selection;    // Expected adverse selection per limit order ($ per share)
    double half_spread;          // Half of the bid-ask spread in the underlying
    double sigma;                // Underlying asset volatility
    double gamma_portfolio;      // Net portfolio Gamma
    double max_delta_tolerance;  // Maximum allowable delta deviation (shares)

    // Derived thresholds
    double inner_threshold() const {
        // Inner zone: hold if delta drift < adverse_selection / (2 * sigma)
        // This is where the cost of posting a limit order (adverse selection)
        // exceeds the benefit of hedging
        if (sigma <= 0.0) return 0.01;
        return adverse_selection / (2.0 * sigma);
    }

    double outer_threshold() const {
        // Outer zone: market order when delta drift exceeds the point where
        // the spread cost is justified by the gamma risk avoided
        // ε_outer = min(max_delta_tolerance, half_spread / (gamma * sigma²))
        double risk_threshold = max_delta_tolerance;
        if (std::abs(gamma_portfolio) > 1e-15 && sigma > 0.0) {
            double gamma_risk = half_spread / (std::abs(gamma_portfolio) * sigma * sigma);
            risk_threshold = std::min(risk_threshold, gamma_risk);
        }
        return std::max(risk_threshold, inner_threshold() * 1.5); // Ensure outer > inner
    }

    void validate() const {
        if (adverse_selection < 0.0)
            throw std::invalid_argument("CRB: adverse_selection must be non-negative");
        if (half_spread < 0.0)
            throw std::invalid_argument("CRB: half_spread must be non-negative");
        if (sigma <= 0.0)
            throw std::invalid_argument("CRB: sigma must be positive");
        if (max_delta_tolerance <= 0.0)
            throw std::invalid_argument("CRB: max_delta_tolerance must be positive");
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Core decision logic
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate the optimal hedge action based on current delta deviation.
///
/// Uses the HJB-QVI framework to determine:
///   1. Whether to hold (delta within inner band)
///   2. Whether to post a limit order (delta in middle band)
///   3. Whether to cross the spread with a market order (delta beyond outer band)
///
/// @param delta_actual Current portfolio delta (shares equivalent)
/// @param delta_target Theoretical (BS) target delta
/// @param S Current spot price
/// @param params CRB model parameters
/// @returns CRBOrder with type, shares, and rationale
inline CRBOrder evaluate_hedge_action(
    double delta_actual, double delta_target,
    double S, const CRBParams& params)
{
    params.validate();

    double drift = delta_actual - delta_target;
    double abs_drift = std::abs(drift);
    double inner = params.inner_threshold();
    double outer = params.outer_threshold();

    CRBOrder order{};

    if (abs_drift < inner) {
        // ── Inner zone: HOLD ──
        // Cost of any action exceeds risk of holding
        order.type = OrderType::HOLD;
        order.shares = 0.0;
        order.limit_price = 0.0;
        order.urgency = 0.0;
        order.reason = "Delta drift (" + std::to_string(abs_drift)
                     + ") within inner threshold (" + std::to_string(inner) + ")";
    }
    else if (abs_drift < outer) {
        // ── Middle zone: LIMIT ORDER ──
        // Post passive limit order to avoid paying the spread
        double shares_needed = delta_target - delta_actual;
        double limit_offset = params.half_spread * 0.5; // Post inside the spread

        // Limit price: if buying (shares > 0), post below mid; if selling, above mid
        double limit = (shares_needed > 0)
            ? S - limit_offset  // Buy limit below mid
            : S + limit_offset; // Sell limit above mid

        order.type = OrderType::LIMIT_ORDER;
        order.shares = shares_needed;
        order.limit_price = limit;
        order.urgency = (abs_drift - inner) / (outer - inner); // 0→1 linearity
        order.reason = "Delta drift (" + std::to_string(abs_drift)
                     + ") in limit order zone [" + std::to_string(inner)
                     + ", " + std::to_string(outer) + ")";
    }
    else {
        // ── Outer zone: MARKET ORDER ──
        // Delta has drifted too far — pay the spread to rebalance immediately
        double shares_needed = delta_target - delta_actual;

        order.type = OrderType::MARKET_ORDER;
        order.shares = shares_needed;
        order.limit_price = 0.0; // Market order — no limit
        order.urgency = 1.0;
        order.reason = "Delta drift (" + std::to_string(abs_drift)
                     + ") exceeds outer threshold (" + std::to_string(outer)
                     + ") — EMERGENCY rebalance";
    }

    return order;
}

/// Compute the transaction cost savings from using the CRB approach
/// versus naive market-order-only hedging.
///
/// @param limit_fill_rate Expected fill rate for limit orders (0 to 1)
/// @param half_spread Half bid-ask spread
/// @param avg_trades_per_day Average number of hedge trades per day
/// @param fraction_limit Fraction of trades done via limit orders
/// @returns Estimated daily cost savings per share
inline double estimated_crb_savings(
    double limit_fill_rate, double half_spread,
    double avg_trades_per_day, double fraction_limit)
{
    // Naive cost: all market orders, pay full spread every time
    double naive_cost = avg_trades_per_day * 2.0 * half_spread;

    // CRB cost: fraction via limit (pay adverse selection only), rest via market
    double limit_cost = fraction_limit * avg_trades_per_day * limit_fill_rate * half_spread * 0.5;
    double market_cost = (1.0 - fraction_limit) * avg_trades_per_day * 2.0 * half_spread;

    return naive_cost - (limit_cost + market_cost);
}

} // namespace hedging
} // namespace davinci
