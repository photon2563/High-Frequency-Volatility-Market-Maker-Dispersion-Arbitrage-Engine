/// @file implied_vol.cpp
/// @brief Implementation file for implied volatility solver.
///
/// Core solver logic is in the header (inline). This file provides:
///   - Diagnostic and logging utilities
///   - Performance benchmarking helpers

#include "pricing/implied_vol.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <vector>

namespace davinci {
namespace pricing {

// ─────────────────────────────────────────────────────────────────────────────
// Diagnostic utilities
// ─────────────────────────────────────────────────────────────────────────────

const char* method_name(IVMethod m) {
    switch (m) {
        case IVMethod::CORRADO_MILLER:  return "Corrado-Miller";
        case IVMethod::NEWTON_RAPHSON:  return "Newton-Raphson";
        case IVMethod::BRENT:           return "Brent";
        case IVMethod::INTRINSIC_ONLY:  return "Intrinsic-Only";
    }
    return "Unknown";
}

std::string format_iv_result(const IVResult& r) {
    std::ostringstream oss;
    oss << std::fixed;
    oss << "IV:         " << std::setprecision(6) << r.sigma
        << " (" << std::setprecision(2) << r.sigma * 100.0 << "%)\n"
        << "Iterations: " << r.iterations << "\n"
        << "Method:     " << method_name(r.method_used) << "\n"
        << "Converged:  " << (r.converged ? "YES" : "NO") << "\n"
        << "Residual:   " << std::scientific << std::setprecision(4) << r.residual << "\n";
    return oss.str();
}

/// Round-trip verification: price → IV → re-price → compare.
/// Returns the absolute error of the round-trip.
double verify_iv_round_trip(
    double S, double K, double r, double T, double sigma, double q,
    bool is_call)
{
    // Forward: sigma → price
    double price = price_and_greeks(S, K, r, T, sigma, q, is_call).price;

    // Inverse: price → sigma_recovered
    auto iv_result = solve_implied_vol(price, S, K, r, T, q, is_call);

    if (!iv_result.converged) return -1.0; // Failed

    // Forward again: sigma_recovered → price_recovered
    double price_recovered = price_and_greeks(
        S, K, r, T, iv_result.sigma, q, is_call).price;

    return std::abs(price - price_recovered);
}

/// Benchmark the IV solver speed: solve N options and report throughput.
struct BenchmarkResult {
    int    n_options;
    double total_time_ms;
    double options_per_second;
    int    n_converged;
    int    n_newton;
    int    n_brent;
};

BenchmarkResult benchmark_iv_solver(int n_options) {
    // Generate a range of test options
    std::vector<double> prices, S_vec, K_vec, r_vec, T_vec, q_vec;
    std::vector<bool> call_vec;

    double S = 100.0, r = 0.05, q = 0.0;
    for (int i = 0; i < n_options; ++i) {
        double moneyness = 0.8 + 0.4 * (static_cast<double>(i) / n_options);
        double K = S * moneyness;
        double T = 0.1 + 1.9 * (static_cast<double>(i % 100) / 100.0);
        double sigma = 0.1 + 0.5 * (static_cast<double>((i * 7) % 100) / 100.0);
        bool is_call = (i % 2 == 0);

        double price = price_and_greeks(S, K, r, T, sigma, q, is_call).price;
        prices.push_back(price);
        S_vec.push_back(S);
        K_vec.push_back(K);
        r_vec.push_back(r);
        T_vec.push_back(T);
        q_vec.push_back(q);
        call_vec.push_back(is_call);
    }

    // Solve all IVs
    auto start = std::chrono::high_resolution_clock::now();

    int n_converged = 0, n_newton = 0, n_brent = 0;
    for (int i = 0; i < n_options; ++i) {
        auto result = solve_implied_vol(
            prices[i], S_vec[i], K_vec[i], r_vec[i], T_vec[i], q_vec[i], call_vec[i]);
        if (result.converged) ++n_converged;
        if (result.method_used == IVMethod::NEWTON_RAPHSON) ++n_newton;
        if (result.method_used == IVMethod::BRENT) ++n_brent;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return BenchmarkResult{
        n_options,
        elapsed_ms,
        n_options / (elapsed_ms / 1000.0),
        n_converged,
        n_newton,
        n_brent
    };
}

} // namespace pricing
} // namespace davinci
