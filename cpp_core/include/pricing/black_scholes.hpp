#pragma once
/// @file black_scholes.hpp
/// @brief Full Black-Scholes pricing engine with all Greeks (Δ,Γ,ν,Θ,Vanna,Volga).
///
/// Design: Zero external dependencies — norm_cdf via std::erfc, norm_pdf via std::exp.
/// Templated for float/double SIMD friendliness. Batch API for vectorized pricing.

#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

namespace davinci {
namespace pricing {

// ─────────────────────────────────────────────────────────────────────────────
// Result struct: all Greeks computed in a single pass
// ─────────────────────────────────────────────────────────────────────────────

struct GreeksResult {
    double price;     // Option price
    double delta;     // ∂V/∂S
    double gamma;     // ∂²V/∂S²
    double vega;      // ∂V/∂σ  (per 1 unit σ, not per 1% — divide by 100 if needed)
    double theta;     // ∂V/∂t  (per year — divide by 365 for calendar day, 252 for trading day)
    double rho;       // ∂V/∂r
    double vanna;     // ∂Δ/∂σ  = ∂ν/∂S  (cross-gamma)
    double volga;     // ∂²V/∂σ² = ∂ν/∂σ  (vol-of-vol sensitivity)

    // Convenience: per-unit conversions
    double vega_1pct() const { return vega * 0.01; }         // Vega per 1% vol move
    double theta_1day(int trading_days = 252) const {        // Theta per 1 trading day
        return theta / static_cast<double>(trading_days);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Mathematical primitives — no Boost dependency
// ─────────────────────────────────────────────────────────────────────────────

/// Standard normal CDF: Φ(x) = 0.5 * erfc(-x / √2)
inline double norm_cdf(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

/// Standard normal PDF: φ(x) = exp(-x²/2) / √(2π)
inline double norm_pdf(double x) {
    static constexpr double inv_sqrt_2pi = 0.3989422804014327; // 1/√(2π)
    return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

// ─────────────────────────────────────────────────────────────────────────────
// Core pricing functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute d1 and d2 for the Black-Scholes formula.
/// @param S Spot price
/// @param K Strike price
/// @param r Risk-free rate (continuous compounding)
/// @param T Time to expiry (years)
/// @param sigma Volatility (annualized)
/// @param q Continuous dividend yield
/// @returns {d1, d2}
inline std::pair<double, double> compute_d1_d2(
    double S, double K, double r, double T, double sigma, double q = 0.0)
{
    if (T <= 0.0 || sigma <= 0.0 || S <= 0.0 || K <= 0.0) {
        throw std::invalid_argument("BS d1/d2: S, K, T, sigma must all be positive");
    }
    double sqrt_T   = std::sqrt(T);
    double sig_sqrtT = sigma * sqrt_T;
    double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sig_sqrtT;
    double d2 = d1 - sig_sqrtT;
    return {d1, d2};
}

/// Compute option price + all Greeks in a single pass.
/// @param S Spot price
/// @param K Strike price
/// @param r Risk-free rate
/// @param T Time to expiry (years)
/// @param sigma Implied volatility (annualized)
/// @param q Continuous dividend yield (default 0)
/// @param is_call true for call, false for put
/// @returns GreeksResult with price and all sensitivities
inline GreeksResult price_and_greeks(
    double S, double K, double r, double T, double sigma,
    double q = 0.0, bool is_call = true)
{
    // ── Intrinsic value for expired/near-expired options ──
    if (T <= 1e-12) {
        GreeksResult result{};
        double intrinsic = is_call
            ? std::max(S - K, 0.0)
            : std::max(K - S, 0.0);
        result.price = intrinsic;
        result.delta = is_call ? (S > K ? 1.0 : 0.0) : (S < K ? -1.0 : 0.0);
        // All other Greeks zero at expiry
        return result;
    }

    auto [d1, d2] = compute_d1_d2(S, K, r, T, sigma, q);

    double sqrt_T    = std::sqrt(T);
    double sig_sqrtT = sigma * sqrt_T;
    double df_q      = std::exp(-q * T);   // dividend discount factor
    double df_r      = std::exp(-r * T);   // risk-free discount factor
    double Nd1       = norm_cdf(d1);
    double Nd2       = norm_cdf(d2);
    double nd1       = norm_pdf(d1);        // φ(d1)

    GreeksResult g{};

    // ── Price ──
    if (is_call) {
        g.price = S * df_q * Nd1 - K * df_r * Nd2;
    } else {
        g.price = K * df_r * (1.0 - Nd2) - S * df_q * (1.0 - Nd1);
    }

    // ── Delta ──
    if (is_call) {
        g.delta = df_q * Nd1;
    } else {
        g.delta = -df_q * (1.0 - Nd1);
    }

    // ── Gamma (identical for call/put) ──
    g.gamma = df_q * nd1 / (S * sig_sqrtT);

    // ── Vega (identical for call/put, per unit sigma) ──
    g.vega = S * df_q * nd1 * sqrt_T;

    // ── Theta ──
    double term1 = -(S * df_q * nd1 * sigma) / (2.0 * sqrt_T);
    if (is_call) {
        g.theta = term1 + q * S * df_q * Nd1 - r * K * df_r * Nd2;
    } else {
        g.theta = term1 - q * S * df_q * (1.0 - Nd1) + r * K * df_r * (1.0 - Nd2);
    }

    // ── Rho ──
    if (is_call) {
        g.rho = K * T * df_r * Nd2;
    } else {
        g.rho = -K * T * df_r * (1.0 - Nd2);
    }

    // ── Vanna: ∂Δ/∂σ = ∂ν/∂S = -df_q * φ(d1) * d2/σ ──
    g.vanna = -df_q * nd1 * d2 / sigma;

    // ── Volga: ∂²V/∂σ² = ν * d1 * d2 / σ ──
    g.volga = g.vega * d1 * d2 / sigma;

    return g;
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch pricing API — vectorized over option parameters
// ─────────────────────────────────────────────────────────────────────────────

/// Price a batch of options. All vectors must be the same length.
inline std::vector<GreeksResult> price_batch(
    const std::vector<double>& S_vec,
    const std::vector<double>& K_vec,
    const std::vector<double>& r_vec,
    const std::vector<double>& T_vec,
    const std::vector<double>& sigma_vec,
    const std::vector<double>& q_vec,
    const std::vector<bool>&   is_call_vec)
{
    size_t n = S_vec.size();
    if (K_vec.size() != n || r_vec.size() != n || T_vec.size() != n ||
        sigma_vec.size() != n || q_vec.size() != n || is_call_vec.size() != n) {
        throw std::invalid_argument("price_batch: all input vectors must have equal length");
    }

    std::vector<GreeksResult> results(n);
    for (size_t i = 0; i < n; ++i) {
        results[i] = price_and_greeks(
            S_vec[i], K_vec[i], r_vec[i], T_vec[i],
            sigma_vec[i], q_vec[i], is_call_vec[i]);
    }
    return results;
}

/// Convenience: price a single option (call) with default q=0
inline double bs_call_price(double S, double K, double r, double T, double sigma, double q = 0.0) {
    return price_and_greeks(S, K, r, T, sigma, q, true).price;
}

/// Convenience: price a single option (put) with default q=0
inline double bs_put_price(double S, double K, double r, double T, double sigma, double q = 0.0) {
    return price_and_greeks(S, K, r, T, sigma, q, false).price;
}

} // namespace pricing
} // namespace davinci
