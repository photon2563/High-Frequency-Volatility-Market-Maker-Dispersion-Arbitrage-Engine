#pragma once
/// @file zscore_signal.hpp
/// @brief Rolling Z-score entry/exit signal for dispersion trading.
///
/// Normalizes dirty implied correlation via a rolling 20-day window.
/// Generates trading signals based on statistical thresholds.

#include <cmath>
#include <vector>
#include <deque>
#include <numeric>
#include <stdexcept>

namespace davinci {
namespace dispersion {

// ─────────────────────────────────────────────────────────────────────────────
// State machine
// ─────────────────────────────────────────────────────────────────────────────

enum class DispersionState {
    FLAT,               // No position
    SHORT_DISPERSION,   // Short index vol, long constituent vol (ρ high → expect mean reversion)
    LONG_DISPERSION     // Long index vol, short constituent vol (ρ low → expect mean reversion)
};

inline const char* to_string(DispersionState s) {
    switch (s) {
        case DispersionState::FLAT:             return "FLAT";
        case DispersionState::SHORT_DISPERSION: return "SHORT_DISPERSION";
        case DispersionState::LONG_DISPERSION:  return "LONG_DISPERSION";
    }
    return "UNKNOWN";
}

// ─────────────────────────────────────────────────────────────────────────────
// Z-score result
// ─────────────────────────────────────────────────────────────────────────────

struct ZScoreResult {
    double zscore;              // Current Z-score
    double mean;                // Rolling mean
    double stddev;              // Rolling standard deviation
    double raw_value;           // Raw dirty correlation
    DispersionState signal;     // Recommended action
    bool   is_entry;            // True if this is a new entry signal
    bool   is_exit;             // True if this is an exit signal
};

// ─────────────────────────────────────────────────────────────────────────────
// Z-score signal parameters
// ─────────────────────────────────────────────────────────────────────────────

struct ZScoreParams {
    int    window = 20;            // Rolling lookback window (trading days)
    double entry_threshold = 0.5;  // Enter when |Z| > this
    double exit_threshold  = 0.05; // Exit when |Z| < this
    double max_zscore      = 3.0;  // Cap Z-score for risk management

    void validate() const {
        if (window < 2)
            throw std::invalid_argument("ZScore: window must be >= 2");
        if (entry_threshold <= exit_threshold)
            throw std::invalid_argument("ZScore: entry_threshold must exceed exit_threshold");
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Rolling Z-score engine
// ─────────────────────────────────────────────────────────────────────────────

class ZScoreEngine {
public:
    explicit ZScoreEngine(const ZScoreParams& params = {})
        : params_(params), state_(DispersionState::FLAT)
    {
        params_.validate();
    }

    /// Update with a new dirty correlation observation.
    ///
    /// @param dirty_correlation Latest dirty implied correlation value
    /// @returns ZScoreResult with signal and diagnostics
    ZScoreResult update(double dirty_correlation) {
        history_.push_back(dirty_correlation);

        // Trim to window
        while (static_cast<int>(history_.size()) > params_.window) {
            history_.pop_front();
        }

        ZScoreResult result{};
        result.raw_value = dirty_correlation;
        result.signal = state_;
        result.is_entry = false;
        result.is_exit = false;

        // Need full window for reliable statistics
        if (static_cast<int>(history_.size()) < params_.window) {
            result.zscore = 0.0;
            result.mean = dirty_correlation;
            result.stddev = 0.0;
            return result;
        }

        // Compute rolling mean
        double sum = std::accumulate(history_.begin(), history_.end(), 0.0);
        double mean = sum / static_cast<double>(history_.size());

        // Compute rolling standard deviation
        double var_sum = 0.0;
        for (double v : history_) {
            double diff = v - mean;
            var_sum += diff * diff;
        }
        double stddev = std::sqrt(var_sum / static_cast<double>(history_.size() - 1));

        result.mean = mean;
        result.stddev = stddev;

        // Compute Z-score
        if (stddev < 1e-15) {
            result.zscore = 0.0;
        } else {
            result.zscore = (dirty_correlation - mean) / stddev;
        }

        // Cap Z-score
        result.zscore = std::clamp(result.zscore, -params_.max_zscore, params_.max_zscore);

        // ── State machine transitions ──
        switch (state_) {
            case DispersionState::FLAT:
                if (result.zscore > params_.entry_threshold) {
                    // High dirty correlation → sell index vol, buy constituent vol
                    state_ = DispersionState::SHORT_DISPERSION;
                    result.is_entry = true;
                } else if (result.zscore < -params_.entry_threshold) {
                    // Low dirty correlation → buy index vol, sell constituent vol
                    state_ = DispersionState::LONG_DISPERSION;
                    result.is_entry = true;
                }
                break;

            case DispersionState::SHORT_DISPERSION:
                if (std::abs(result.zscore) < params_.exit_threshold) {
                    // Mean reversion → unwind
                    state_ = DispersionState::FLAT;
                    result.is_exit = true;
                } else if (result.zscore < -params_.entry_threshold) {
                    // Reverse to long dispersion
                    state_ = DispersionState::LONG_DISPERSION;
                    result.is_exit = true;
                    result.is_entry = true;
                }
                break;

            case DispersionState::LONG_DISPERSION:
                if (std::abs(result.zscore) < params_.exit_threshold) {
                    state_ = DispersionState::FLAT;
                    result.is_exit = true;
                } else if (result.zscore > params_.entry_threshold) {
                    state_ = DispersionState::SHORT_DISPERSION;
                    result.is_exit = true;
                    result.is_entry = true;
                }
                break;
        }

        result.signal = state_;
        return result;
    }

    /// Get current state
    DispersionState state() const { return state_; }

    /// Reset the engine to FLAT state
    void reset() {
        history_.clear();
        state_ = DispersionState::FLAT;
    }

    /// Number of observations in the window
    int window_size() const { return static_cast<int>(history_.size()); }

    /// Is the engine warmed up (full window)?
    bool is_warmed_up() const { return window_size() >= params_.window; }

private:
    ZScoreParams params_;
    std::deque<double> history_;
    DispersionState state_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Batch Z-score computation (for historical analysis)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute Z-scores for an entire time series of dirty correlations.
///
/// @param history Full time series of dirty correlations
/// @param params Z-score parameters
/// @returns Vector of ZScoreResult for each observation
inline std::vector<ZScoreResult> compute_zscore_series(
    const std::vector<double>& history,
    const ZScoreParams& params = {})
{
    ZScoreEngine engine(params);
    std::vector<ZScoreResult> results;
    results.reserve(history.size());

    for (double val : history) {
        results.push_back(engine.update(val));
    }

    return results;
}

} // namespace dispersion
} // namespace davinci
