#pragma once
/// @file implied_vol.hpp
/// @brief High-speed implied volatility solver.
///        Three-stage: Corrado-Miller init → Newton-Raphson → Brent's method fallback.
///
/// Handles edge cases: deep ITM/OTM, near-expiry, zero vega regions.
/// Never gets stuck in infinite loops — bounded iteration counts + fallback guarantee.

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include "black_scholes.hpp"

namespace davinci {
namespace pricing {

// ─────────────────────────────────────────────────────────────────────────────
// Convergence diagnostics
// ─────────────────────────────────────────────────────────────────────────────

enum class IVMethod { CORRADO_MILLER, NEWTON_RAPHSON, BRENT, INTRINSIC_ONLY };

struct IVResult {
    double sigma;          // Solved implied vol
    int    iterations;     // Number of iterations used
    IVMethod method_used;  // Which method produced the final answer
    bool   converged;      // True if converged within tolerance
    double residual;       // |BS(σ) - market_price| at solution
};

// ─────────────────────────────────────────────────────────────────────────────
// Stage 1: Corrado-Miller initial guess
// ─────────────────────────────────────────────────────────────────────────────

/// Corrado-Miller (2005) closed-form approximation for ATM/near-ATM implied vol.
/// Provides a much better initial guess than naive σ₀ = 0.5.
///
/// σ_CM ≈ √(2π/T) · { [C - (F-K)/2] + √[ (C - (F-K)/2)² - (F-K)²/π ] } / (F+K)
///
/// where F = S·exp((r-q)·T) is the forward price.
inline double corrado_miller_guess(
    double market_price, double S, double K, double r, double T,
    double q = 0.0, bool is_call = true)
{
    double F = S * std::exp((r - q) * T);
    double df = std::exp(-r * T);

    // Convert put to call price via put-call parity if needed
    double C_market = market_price;
    if (!is_call) {
        // P = C - df*(F - K)  →  C = P + df*(F - K)
        C_market = market_price + df * (F - K);
    }

    // Intrinsic value check
    double intrinsic = df * std::max(F - K, 0.0);
    if (C_market <= intrinsic + 1e-12) {
        return 0.01; // Near-zero vol, return small positive
    }

    double half_diff = 0.5 * (C_market - df * std::max(F - K, 0.0));
    // For deep ITM calls, C_market ≈ df*(F-K) and half_diff ≈ 0, so we add time value
    double fk_sum = F + K;
    if (fk_sum < 1e-12) return 0.2; // Safeguard

    double sqrt_2pi_over_T = std::sqrt(2.0 * M_PI / T);

    // Simplified Corrado-Miller for practical use:
    //   σ ≈ √(2π/T) · (C - (F-K)·df/2) / ((F+K)/2 · df)
    double time_value = C_market - intrinsic;
    double avg_fk = 0.5 * fk_sum * df;
    if (avg_fk < 1e-12) return 0.2;

    double sigma_guess = sqrt_2pi_over_T * time_value / avg_fk;

    // Clamp to reasonable range
    return std::clamp(sigma_guess, 0.005, 5.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage 2: Newton-Raphson iteration
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int    NR_MAX_ITER   = 50;
static constexpr double NR_TOL        = 1e-10;
static constexpr double NR_VEGA_FLOOR = 1e-14;  // Below this, vega too small for NR

// ─────────────────────────────────────────────────────────────────────────────
// Stage 3: Brent's method fallback
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int    BRENT_MAX_ITER = 100;
static constexpr double BRENT_TOL     = 1e-10;
static constexpr double BRENT_LO      = 0.001;
static constexpr double BRENT_HI      = 5.0;

/// Brent's method for implied vol — guaranteed convergence on bracketed interval.
/// Used as fallback when Newton-Raphson fails (near-zero vega regions).
inline IVResult brent_iv(
    double market_price, double S, double K, double r, double T,
    double q, bool is_call,
    double lo = BRENT_LO, double hi = BRENT_HI)
{
    auto f = [&](double sigma) -> double {
        return price_and_greeks(S, K, r, T, sigma, q, is_call).price - market_price;
    };

    double fa = f(lo);
    double fb = f(hi);

    // Ensure bracket: if same sign, try to expand
    if (fa * fb > 0.0) {
        // Try tighter bracket
        for (double test_lo : {0.0001, 0.0005, 0.001, 0.005, 0.01}) {
            fa = f(test_lo);
            if (fa * fb <= 0.0) { lo = test_lo; break; }
        }
        if (fa * fb > 0.0) {
            for (double test_hi : {3.0, 5.0, 8.0, 10.0}) {
                fb = f(test_hi);
                if (fa * fb <= 0.0) { hi = test_hi; break; }
            }
        }
        if (fa * fb > 0.0) {
            // Cannot bracket — return best guess
            return {std::abs(fa) < std::abs(fb) ? lo : hi, 0, IVMethod::BRENT, false,
                    std::min(std::abs(fa), std::abs(fb))};
        }
    }

    // Classic Brent's algorithm
    double a = lo, b = hi;
    fa = f(a); fb = f(b);
    double c = a, fc = fa;
    double d = b - a, e = d;

    for (int iter = 0; iter < BRENT_MAX_ITER; ++iter) {
        if (fb * fc > 0.0) {
            c = a; fc = fa;
            d = b - a; e = d;
        }
        if (std::abs(fc) < std::abs(fb)) {
            a = b; b = c; c = a;
            fa = fb; fb = fc; fc = fa;
        }

        double tol1 = 2.0 * std::numeric_limits<double>::epsilon() * std::abs(b) + 0.5 * BRENT_TOL;
        double xm = 0.5 * (c - b);

        if (std::abs(xm) <= tol1 || std::abs(fb) < BRENT_TOL) {
            return {b, iter + 1, IVMethod::BRENT, true, std::abs(fb)};
        }

        if (std::abs(e) >= tol1 && std::abs(fa) > std::abs(fb)) {
            double s_val;
            if (std::abs(a - c) < 1e-15) {
                // Linear interpolation
                s_val = fb / fa;
                double p = 2.0 * xm * s_val;
                double q_val = 1.0 - s_val;
                if (p > 0.0) q_val = -q_val; else p = -p;
                if (2.0 * p < std::min(3.0 * xm * q_val - std::abs(tol1 * q_val),
                                       std::abs(e * q_val))) {
                    e = d;
                    d = p / q_val;
                } else {
                    d = xm; e = d;
                }
            } else {
                // Inverse quadratic interpolation
                double q_val = fa / fc;
                double r_val = fb / fc;
                s_val = fb / fa;
                double p = s_val * (2.0 * xm * q_val * (q_val - r_val)
                           - (b - a) * (r_val - 1.0));
                q_val = (q_val - 1.0) * (r_val - 1.0) * (s_val - 1.0);
                if (p > 0.0) q_val = -q_val; else p = -p;
                if (2.0 * p < std::min(3.0 * xm * q_val - std::abs(tol1 * q_val),
                                       std::abs(e * q_val))) {
                    e = d;
                    d = p / q_val;
                } else {
                    d = xm; e = d;
                }
            }
        } else {
            d = xm; e = d;
        }

        a = b; fa = fb;
        if (std::abs(d) > tol1) {
            b += d;
        } else {
            b += (xm > 0.0 ? tol1 : -tol1);
        }
        fb = f(b);
    }

    return {b, BRENT_MAX_ITER, IVMethod::BRENT, false, std::abs(fb)};
}

// ─────────────────────────────────────────────────────────────────────────────
// Main solver: three-stage pipeline
// ─────────────────────────────────────────────────────────────────────────────

/// Solve for implied volatility given a market price.
///
/// Pipeline:
///   1. Corrado-Miller initial guess
///   2. Newton-Raphson refinement (up to 50 iterations)
///   3. Brent's method fallback if NR fails
///
/// @param market_price Observed market price of the option
/// @param S Spot price
/// @param K Strike price
/// @param r Risk-free rate
/// @param T Time to expiry (years)
/// @param q Continuous dividend yield
/// @param is_call true for call, false for put
/// @returns IVResult with solved sigma and convergence diagnostics
inline IVResult solve_implied_vol(
    double market_price, double S, double K, double r, double T,
    double q = 0.0, bool is_call = true)
{
    // ── Sanity checks ──
    if (T <= 0.0 || S <= 0.0 || K <= 0.0) {
        throw std::invalid_argument("solve_implied_vol: S, K, T must be positive");
    }
    if (market_price <= 0.0) {
        throw std::invalid_argument("solve_implied_vol: market_price must be positive");
    }

    // Check for pure intrinsic value (no time value → σ ≈ 0)
    double df = std::exp(-r * T);
    double F = S * std::exp((r - q) * T);
    double intrinsic_call = df * std::max(F - K, 0.0);
    double intrinsic_put  = df * std::max(K - F, 0.0);
    double intrinsic = is_call ? intrinsic_call : intrinsic_put;

    if (market_price < intrinsic - 1e-10) {
        // Price below intrinsic — arbitrage or bad data
        return {0.0, 0, IVMethod::INTRINSIC_ONLY, false, std::abs(market_price - intrinsic)};
    }

    // ── Stage 1: Corrado-Miller guess ──
    double sigma = corrado_miller_guess(market_price, S, K, r, T, q, is_call);

    // ── Stage 2: Newton-Raphson ──
    int iter = 0;
    bool nr_converged = false;
    for (; iter < NR_MAX_ITER; ++iter) {
        auto greeks = price_and_greeks(S, K, r, T, sigma, q, is_call);
        double diff = greeks.price - market_price;

        if (std::abs(diff) < NR_TOL) {
            nr_converged = true;
            break;
        }

        // Vega guard: if vega is too small, NR step is unreliable
        if (std::abs(greeks.vega) < NR_VEGA_FLOOR) {
            break; // Fall through to Brent
        }

        double step = diff / greeks.vega;
        sigma -= step;

        // Clamp sigma to prevent negative or absurd values
        sigma = std::clamp(sigma, 0.0001, 10.0);
    }

    if (nr_converged) {
        double residual = std::abs(
            price_and_greeks(S, K, r, T, sigma, q, is_call).price - market_price);
        return {sigma, iter + 1, IVMethod::NEWTON_RAPHSON, true, residual};
    }

    // ── Stage 3: Brent's method fallback ──
    return brent_iv(market_price, S, K, r, T, q, is_call);
}

/// Batch implied vol solver.
inline std::vector<IVResult> solve_iv_batch(
    const std::vector<double>& market_prices,
    const std::vector<double>& S_vec,
    const std::vector<double>& K_vec,
    const std::vector<double>& r_vec,
    const std::vector<double>& T_vec,
    const std::vector<double>& q_vec,
    const std::vector<bool>&   is_call_vec)
{
    size_t n = market_prices.size();
    std::vector<IVResult> results(n);
    for (size_t i = 0; i < n; ++i) {
        results[i] = solve_implied_vol(
            market_prices[i], S_vec[i], K_vec[i],
            r_vec[i], T_vec[i], q_vec[i], is_call_vec[i]);
    }
    return results;
}

} // namespace pricing
} // namespace davinci
