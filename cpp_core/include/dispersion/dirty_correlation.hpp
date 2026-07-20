#pragma once
/// @file dirty_correlation.hpp
/// @brief Dirty implied correlation: index IV² / weighted constituent IV² ratio.
///
/// Measures the structural premium embedded in index volatility relative to
/// its constituents — the fundamental signal for dispersion trading.

#include <cmath>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <algorithm>

namespace davinci {
namespace dispersion {

// ─────────────────────────────────────────────────────────────────────────────
// Result struct
// ─────────────────────────────────────────────────────────────────────────────

struct CorrelationResult {
    double dirty_correlation;       // ρ_dirty = σ²_index / (Σ w_i σ_i)²
    double index_iv;                // Index implied vol
    double weighted_constituent_iv; // Market-cap weighted average constituent IV
    double correlation_premium;     // How much ρ_dirty exceeds historical average
};

// ─────────────────────────────────────────────────────────────────────────────
// Core computation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the dirty implied correlation.
///
/// ρ_dirty = σ²_index / (Σ w_i · σ_i)²
///
/// This equals the "average" pairwise correlation implied by the index and
/// constituent option markets. When ρ_dirty > realized correlation, the
/// dispersion trade (short index vol, long constituent vol) profits.
///
/// @param index_iv Index implied volatility (annualized, e.g., 0.20)
/// @param constituent_ivs Vector of constituent implied volatilities
/// @param weights Market-cap weights (must sum to ≈ 1.0)
/// @returns dirty implied correlation ∈ [0, 1] (clamped)
inline double compute_dirty_correlation(
    double index_iv,
    const std::vector<double>& constituent_ivs,
    const std::vector<double>& weights)
{
    if (constituent_ivs.size() != weights.size()) {
        throw std::invalid_argument("DirtyCorrelation: IVs and weights must have same length");
    }
    if (constituent_ivs.empty()) {
        throw std::invalid_argument("DirtyCorrelation: need at least one constituent");
    }
    if (index_iv <= 0.0) {
        throw std::invalid_argument("DirtyCorrelation: index_iv must be positive");
    }

    // Compute weighted average constituent IV: Σ w_i · σ_i
    double weighted_iv = 0.0;
    for (size_t i = 0; i < constituent_ivs.size(); ++i) {
        if (constituent_ivs[i] <= 0.0 || weights[i] < 0.0) {
            throw std::invalid_argument("DirtyCorrelation: constituent IVs and weights must be non-negative");
        }
        weighted_iv += weights[i] * constituent_ivs[i];
    }

    if (weighted_iv <= 1e-15) {
        throw std::invalid_argument("DirtyCorrelation: weighted constituent IV is effectively zero");
    }

    // ρ_dirty = σ²_index / (Σ w_i σ_i)²
    double rho = (index_iv * index_iv) / (weighted_iv * weighted_iv);

    // Clamp to [0, 1] — dirty correlation > 1 indicates estimation error
    return std::clamp(rho, 0.0, 1.0);
}

/// Full computation with diagnostics.
inline CorrelationResult compute_dirty_correlation_full(
    double index_iv,
    const std::vector<double>& constituent_ivs,
    const std::vector<double>& weights,
    double historical_avg_correlation = 0.0)
{
    double weighted_iv = 0.0;
    for (size_t i = 0; i < constituent_ivs.size(); ++i) {
        weighted_iv += weights[i] * constituent_ivs[i];
    }

    double rho = compute_dirty_correlation(index_iv, constituent_ivs, weights);

    return CorrelationResult{
        rho,
        index_iv,
        weighted_iv,
        rho - historical_avg_correlation
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// Alternative: variance-weighted dirty correlation
// ─────────────────────────────────────────────────────────────────────────────

/// Variance-weighted dirty correlation (alternative formulation).
///
/// ρ_var = [σ²_index - Σ w²_i · σ²_i] / [Σ_{i≠j} w_i · w_j · σ_i · σ_j]
///
/// This decomposes the index variance into:
///   σ²_index = Σ w²_i σ²_i + Σ_{i≠j} w_i w_j ρ_{ij} σ_i σ_j
///
/// Solving for the "implied equicorrelation" ρ:
///   ρ = (σ²_index - Σ w²_i σ²_i) / (Σ_{i≠j} w_i w_j σ_i σ_j)
///
/// @returns Implied equicorrelation (can be < 0 in extreme dislocations)
inline double compute_variance_weighted_correlation(
    double index_iv,
    const std::vector<double>& constituent_ivs,
    const std::vector<double>& weights)
{
    size_t n = constituent_ivs.size();
    if (n != weights.size() || n == 0) {
        throw std::invalid_argument("VarWeightedCorrelation: size mismatch");
    }

    // Σ w²_i · σ²_i (individual variance contribution)
    double individual_var = 0.0;
    for (size_t i = 0; i < n; ++i) {
        individual_var += weights[i] * weights[i] * constituent_ivs[i] * constituent_ivs[i];
    }

    // Σ_{i≠j} w_i w_j σ_i σ_j (cross-variance contribution)
    double cross_var = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            cross_var += 2.0 * weights[i] * weights[j]
                       * constituent_ivs[i] * constituent_ivs[j];
        }
    }

    if (std::abs(cross_var) < 1e-15) return 0.0;

    double index_var = index_iv * index_iv;
    return (index_var - individual_var) / cross_var;
}

} // namespace dispersion
} // namespace davinci
