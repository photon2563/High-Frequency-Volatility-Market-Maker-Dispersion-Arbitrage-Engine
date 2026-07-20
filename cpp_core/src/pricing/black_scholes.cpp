/// @file black_scholes.cpp
/// @brief Implementation file for Black-Scholes pricing engine.
///
/// All core logic is in the header (inline functions for performance).
/// This file exists for:
///   1. Explicit template instantiation (if needed in future)
///   2. Non-inline utility functions
///   3. Validation and diagnostic routines

#include "pricing/black_scholes.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>

namespace davinci {
namespace pricing {

// ─────────────────────────────────────────────────────────────────────────────
// Diagnostic / pretty-print utilities
// ─────────────────────────────────────────────────────────────────────────────

std::string format_greeks(const GreeksResult& g, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision);
    oss << "Price:  " << g.price  << "\n"
        << "Delta:  " << g.delta  << "\n"
        << "Gamma:  " << g.gamma  << "\n"
        << "Vega:   " << g.vega   << "  (per 1%: " << g.vega_1pct() << ")\n"
        << "Theta:  " << g.theta  << "  (per day: " << g.theta_1day() << ")\n"
        << "Rho:    " << g.rho    << "\n"
        << "Vanna:  " << g.vanna  << "\n"
        << "Volga:  " << g.volga  << "\n";
    return oss.str();
}

// ─────────────────────────────────────────────────────────────────────────────
// Cross-verification routines (Module 6)
// ─────────────────────────────────────────────────────────────────────────────

/// Verify put-call parity: C - P = S·e^(-qT) - K·e^(-rT)
/// Returns the absolute violation (should be ~0).
double verify_put_call_parity(
    double S, double K, double r, double T, double sigma, double q)
{
    auto call = price_and_greeks(S, K, r, T, sigma, q, true);
    auto put  = price_and_greeks(S, K, r, T, sigma, q, false);
    double lhs = call.price - put.price;
    double rhs = S * std::exp(-q * T) - K * std::exp(-r * T);
    return std::abs(lhs - rhs);
}

/// Verify Gamma symmetry: Γ_call = Γ_put for same parameters.
double verify_gamma_symmetry(
    double S, double K, double r, double T, double sigma, double q)
{
    auto call = price_and_greeks(S, K, r, T, sigma, q, true);
    auto put  = price_and_greeks(S, K, r, T, sigma, q, false);
    return std::abs(call.gamma - put.gamma);
}

/// Verify Vega symmetry: ν_call = ν_put for same parameters.
double verify_vega_symmetry(
    double S, double K, double r, double T, double sigma, double q)
{
    auto call = price_and_greeks(S, K, r, T, sigma, q, true);
    auto put  = price_and_greeks(S, K, r, T, sigma, q, false);
    return std::abs(call.vega - put.vega);
}

/// Numerical delta check: compare analytical Δ with finite-difference.
double verify_delta_finite_diff(
    double S, double K, double r, double T, double sigma, double q,
    bool is_call, double bump = 0.01)
{
    auto up   = price_and_greeks(S + bump, K, r, T, sigma, q, is_call);
    auto down = price_and_greeks(S - bump, K, r, T, sigma, q, is_call);
    double fd_delta = (up.price - down.price) / (2.0 * bump);
    auto exact = price_and_greeks(S, K, r, T, sigma, q, is_call);
    return std::abs(fd_delta - exact.delta);
}

/// Numerical gamma check: compare analytical Γ with finite-difference.
double verify_gamma_finite_diff(
    double S, double K, double r, double T, double sigma, double q,
    bool is_call, double bump = 0.01)
{
    auto up   = price_and_greeks(S + bump, K, r, T, sigma, q, is_call);
    auto mid  = price_and_greeks(S, K, r, T, sigma, q, is_call);
    auto down = price_and_greeks(S - bump, K, r, T, sigma, q, is_call);
    double fd_gamma = (up.price - 2.0 * mid.price + down.price) / (bump * bump);
    return std::abs(fd_gamma - mid.gamma);
}

/// Numerical vega check: compare analytical ν with finite-difference.
double verify_vega_finite_diff(
    double S, double K, double r, double T, double sigma, double q,
    bool is_call, double bump = 0.0001)
{
    auto up   = price_and_greeks(S, K, r, T, sigma + bump, q, is_call);
    auto down = price_and_greeks(S, K, r, T, sigma - bump, q, is_call);
    double fd_vega = (up.price - down.price) / (2.0 * bump);
    auto exact = price_and_greeks(S, K, r, T, sigma, q, is_call);
    return std::abs(fd_vega - exact.vega);
}

} // namespace pricing
} // namespace davinci
