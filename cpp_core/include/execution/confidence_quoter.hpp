#pragma once
/// @file confidence_quoter.hpp
/// @brief Confidence-interval-based bid-ask band generation.
///
/// Generates 100%, 75%, and 50% confidence bands around a theoretical
/// fair value. Mimics advanced market-making games used at prop trading
/// firms to test probabilistic reasoning and numerical intuition.

#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace davinci {
namespace execution {

// ─────────────────────────────────────────────────────────────────────────────
// Result structs
// ─────────────────────────────────────────────────────────────────────────────

struct ConfidenceBand {
    double confidence_level;  // e.g., 1.0, 0.75, 0.50
    double bid;               // Lower bound
    double ask;               // Upper bound
    double spread;            // ask - bid
    double z_score;           // Z-score used for this confidence level
};

struct ConfidenceQuotes {
    double fair_value;             // Theoretical fair value
    double uncertainty;            // σ_V (standard deviation of fair value estimate)
    std::vector<ConfidenceBand> bands;  // Confidence bands (100%, 75%, 50%)
};

// ─────────────────────────────────────────────────────────────────────────────
// Z-scores for standard confidence levels
// ─────────────────────────────────────────────────────────────────────────────

/// Z-score for a given two-sided confidence level.
///
/// For a two-sided interval: CI = [μ - z·σ, μ + z·σ]
/// Probability P(|X - μ| ≤ z·σ) = confidence_level
///
/// Z-scores for common levels:
///   100% → use a practical maximum (≈ 3.29 for 99.9%)
///   75%  → z = 1.1503 (P(|Z| ≤ 1.1503) = 0.75)
///   50%  → z = 0.6745 (P(|Z| ≤ 0.6745) = 0.50)
///   95%  → z = 1.9600
///   99%  → z = 2.5758
inline double z_for_confidence(double confidence_level) {
    // Use pre-computed values for common levels (exact to 4 decimal places)
    if (std::abs(confidence_level - 1.00) < 0.001) return 3.2905; // 99.9% practical
    if (std::abs(confidence_level - 0.99) < 0.001) return 2.5758;
    if (std::abs(confidence_level - 0.95) < 0.001) return 1.9600;
    if (std::abs(confidence_level - 0.90) < 0.001) return 1.6449;
    if (std::abs(confidence_level - 0.75) < 0.001) return 1.1503;
    if (std::abs(confidence_level - 0.50) < 0.001) return 0.6745;
    if (std::abs(confidence_level - 0.25) < 0.001) return 0.3186;

    // General case: rational approximation of inverse normal CDF
    // Abramowitz & Stegun approximation 26.2.23
    double p = (1.0 + confidence_level) / 2.0; // One-sided probability
    if (p >= 1.0) return 3.2905;
    if (p <= 0.5) return 0.0;

    double t = std::sqrt(-2.0 * std::log(1.0 - p));
    double c0 = 2.515517, c1 = 0.802853, c2 = 0.010328;
    double d1 = 1.432788, d2 = 0.189269, d3 = 0.001308;
    double z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
    return z;
}

// ─────────────────────────────────────────────────────────────────────────────
// Uncertainty estimation
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate the uncertainty (σ_V) of a fair value estimate.
///
/// The uncertainty in the option's fair value comes from two primary sources:
///   1. Implied volatility uncertainty: σ_V_vol = Vega · σ_IV
///   2. Spot price uncertainty: σ_V_spot = Delta · σ_S
///
/// Combined (assuming independence):
///   σ_V = √(Vega² · σ²_IV + Delta² · σ²_S)
///
/// @param vega Option Vega
/// @param delta Option Delta
/// @param iv_uncertainty Uncertainty in implied vol (e.g., 0.02 = 2 vol points)
/// @param spot_uncertainty Uncertainty in spot price (e.g., $0.50)
/// @returns Standard deviation of fair value estimate
inline double estimate_uncertainty(
    double vega, double delta,
    double iv_uncertainty, double spot_uncertainty)
{
    double vol_component = vega * iv_uncertainty;
    double spot_component = delta * spot_uncertainty;
    return std::sqrt(vol_component * vol_component + spot_component * spot_component);
}

/// Estimate uncertainty from vol uncertainty alone (simpler model).
inline double estimate_uncertainty_vol_only(double vega, double iv_uncertainty) {
    return std::abs(vega * iv_uncertainty);
}

// ─────────────────────────────────────────────────────────────────────────────
// Confidence band generation
// ─────────────────────────────────────────────────────────────────────────────

/// Generate confidence bands at the standard 100/75/50% levels.
///
/// @param fair_value Theoretical fair value of the option
/// @param uncertainty Standard deviation of fair value estimate (σ_V)
/// @param min_spread Minimum allowed spread (e.g., $0.01)
/// @returns ConfidenceQuotes with three bands
inline ConfidenceQuotes generate_quotes(
    double fair_value, double uncertainty,
    double min_spread = 0.01)
{
    std::vector<double> levels = {1.00, 0.75, 0.50};
    std::vector<ConfidenceBand> bands;

    for (double level : levels) {
        double z = z_for_confidence(level);
        double half_width = std::max(z * uncertainty, min_spread / 2.0);

        ConfidenceBand band{};
        band.confidence_level = level;
        band.bid = fair_value - half_width;
        band.ask = fair_value + half_width;
        band.spread = 2.0 * half_width;
        band.z_score = z;
        bands.push_back(band);
    }

    return ConfidenceQuotes{fair_value, uncertainty, bands};
}

/// Generate confidence bands at custom confidence levels.
///
/// @param fair_value Theoretical fair value
/// @param uncertainty σ_V
/// @param confidence_levels Vector of confidence levels (e.g., {0.99, 0.95, 0.90, 0.75, 0.50})
/// @param min_spread Minimum allowed spread
/// @returns ConfidenceQuotes with custom bands
inline ConfidenceQuotes generate_quotes_custom(
    double fair_value, double uncertainty,
    const std::vector<double>& confidence_levels,
    double min_spread = 0.01)
{
    std::vector<ConfidenceBand> bands;

    for (double level : confidence_levels) {
        if (level <= 0.0 || level > 1.0) {
            throw std::invalid_argument("ConfidenceQuoter: level must be in (0, 1]");
        }

        double z = z_for_confidence(level);
        double half_width = std::max(z * uncertainty, min_spread / 2.0);

        ConfidenceBand band{};
        band.confidence_level = level;
        band.bid = fair_value - half_width;
        band.ask = fair_value + half_width;
        band.spread = 2.0 * half_width;
        band.z_score = z;
        bands.push_back(band);
    }

    return ConfidenceQuotes{fair_value, uncertainty, bands};
}

// ─────────────────────────────────────────────────────────────────────────────
// Spread quality metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate the quality of a quoted spread relative to the fair value.
///
/// @param bid Quoted bid
/// @param ask Quoted ask
/// @param fair_value Theoretical fair value
/// @returns Spread quality score (0 = terrible, 1 = perfect)
inline double spread_quality(double bid, double ask, double fair_value) {
    if (ask <= bid) return 0.0;

    double spread = ask - bid;
    double mid = 0.5 * (bid + ask);
    double mid_offset = std::abs(mid - fair_value) / spread;

    // Quality = 1.0 when centered on fair value with tight spread
    // Penalize wide spreads and off-center quoting
    double centering_score = std::max(0.0, 1.0 - mid_offset);
    double tightness_score = std::max(0.0, 1.0 - spread / fair_value);

    return centering_score * tightness_score;
}

/// Check if a quoted spread is executable (positive spread, bid < fair < ask).
inline bool is_valid_quote(double bid, double ask, double fair_value) {
    return (ask > bid) && (bid < fair_value) && (ask > fair_value);
}

} // namespace execution
} // namespace davinci
