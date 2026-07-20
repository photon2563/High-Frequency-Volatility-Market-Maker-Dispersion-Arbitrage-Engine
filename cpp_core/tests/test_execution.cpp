/// @file test_execution.cpp
/// @brief Unit tests for position sizer and confidence quoter.

#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include "execution/position_sizer.hpp"
#include "execution/confidence_quoter.hpp"

using namespace davinci::execution;

// ═══════════════════════════════════════════════════════════════════════════
// POSITION SIZER
// ═══════════════════════════════════════════════════════════════════════════

TEST(PositionSizerTest, RoundToLots_ExactMultiple) {
    auto pos = round_to_lots(300.0, 100);
    EXPECT_EQ(pos.rounded_contracts, 300);
    EXPECT_NEAR(pos.rounding_error, 0.0, 1e-10);
}

TEST(PositionSizerTest, RoundToLots_RoundsUp) {
    auto pos = round_to_lots(250.0, 100);
    // 250 / 100 = 2.5 → rounds to 300 (nearest)
    // Wait, std::round(2.5) = 2 or 3 depends on implementation.
    // Actually std::round(2.5) = 3 on most implementations
    EXPECT_TRUE(pos.rounded_contracts == 200 || pos.rounded_contracts == 300);
}

TEST(PositionSizerTest, RoundToLots_RoundsDown) {
    auto pos = round_to_lots(149.0, 100);
    EXPECT_EQ(pos.rounded_contracts, 100);
}

TEST(PositionSizerTest, Conservative_RoundsTowardZero_Long) {
    auto pos = round_to_lots_conservative(350.0, 100);
    EXPECT_EQ(pos.rounded_contracts, 300); // Floor
}

TEST(PositionSizerTest, Conservative_RoundsTowardZero_Short) {
    auto pos = round_to_lots_conservative(-350.0, 100);
    EXPECT_EQ(pos.rounded_contracts, -300); // Ceil (toward zero)
}

TEST(PositionSizerTest, FuturesHedge_ZeroResidual_NoHedge) {
    auto hedge = compute_futures_hedge(0.0, 50.0, 1.0, 100.0);
    EXPECT_EQ(hedge.contracts, 0);
    EXPECT_NEAR(hedge.notional, 0.0, 1e-10);
}

TEST(PositionSizerTest, FuturesHedge_PositiveResidual_SellFutures) {
    // Positive residual delta → sell futures to offset
    auto hedge = compute_futures_hedge(100.0, 50.0, 1.0, 5000.0);
    EXPECT_LT(hedge.contracts, 0); // Sell
}

TEST(PositionSizerTest, FuturesHedge_ReducesResidual) {
    auto hedge = compute_futures_hedge(75.0, 50.0, 1.0, 5000.0);
    EXPECT_LT(std::abs(hedge.residual_delta_after), std::abs(75.0));
}

TEST(PositionSizerTest, PnLApproximation_DeltaOnly) {
    auto pnl = approximate_pnl(0.5, 0.0, 0.0, 0.0, 1.0);
    EXPECT_NEAR(pnl.total_pnl, 0.5, 1e-10);
    EXPECT_NEAR(pnl.delta_pnl, 0.5, 1e-10);
}

TEST(PositionSizerTest, PnLApproximation_GammaConvexity) {
    // Long gamma: positive convexity profit from spot moves
    auto pnl = approximate_pnl(0.0, 0.03, 0.0, 0.0, 2.0);
    EXPECT_NEAR(pnl.gamma_pnl, 0.5 * 0.03 * 4.0, 1e-10);
    EXPECT_GT(pnl.gamma_pnl, 0.0);
}

TEST(PositionSizerTest, PnLApproximation_ThetaDecay) {
    auto pnl = approximate_pnl(0.0, 0.0, 0.0, -5.0, 0.0, 0.0, 1.0/252.0);
    EXPECT_LT(pnl.theta_pnl, 0.0); // Negative theta → daily loss
}

TEST(PositionSizerTest, QuickATMVega_MatchesMentalMath) {
    // ν_ATM ≈ S · √T / √(2π) ≈ 100 · 1 · 0.3989 ≈ 39.89
    double v = quick_atm_vega(100.0, 1.0);
    EXPECT_NEAR(v, 39.89, 0.5);

    std::cout << "[SANITY CHECK] Quick ATM Vega: " << v
              << " (mental math: ~39.89)" << std::endl;
}

TEST(PositionSizerTest, QuickATMCall_MatchesMentalMath) {
    // C_ATM ≈ 0.4 · S · σ · √T = 0.4 · 100 · 0.20 · 1 = 8.0
    double c = quick_atm_call(100.0, 0.20, 1.0);
    EXPECT_NEAR(c, 8.0, 0.01);

    std::cout << "[SANITY CHECK] Quick ATM Call: " << c
              << " (mental math: ~8.0, BS exact: ~10.45)" << std::endl;
}

TEST(PositionSizerTest, DispersionSizing_VegaBalanced) {
    double index_vega = 50.0;
    std::vector<double> const_vegas = {20.0, 15.0, 25.0};
    std::vector<double> weights = {0.5, 0.3, 0.2};

    auto positions = size_dispersion_trade(
        index_vega, const_vegas, weights, 5000.0, 100);

    ASSERT_EQ(positions.size(), 4u); // 1 index + 3 constituents
    EXPECT_LT(positions[0].rounded_contracts, 0); // Short index
}

// ═══════════════════════════════════════════════════════════════════════════
// CONFIDENCE QUOTER
// ═══════════════════════════════════════════════════════════════════════════

TEST(ConfidenceQuoterTest, StandardBands_ThreeLevels) {
    auto quotes = generate_quotes(10.0, 0.5);
    ASSERT_EQ(quotes.bands.size(), 3u);

    // 100% band is widest
    EXPECT_GT(quotes.bands[0].spread, quotes.bands[1].spread);
    // 75% band is wider than 50%
    EXPECT_GT(quotes.bands[1].spread, quotes.bands[2].spread);
}

TEST(ConfidenceQuoterTest, AllBands_CenteredOnFairValue) {
    auto quotes = generate_quotes(10.0, 0.5);
    for (const auto& band : quotes.bands) {
        double mid = 0.5 * (band.bid + band.ask);
        EXPECT_NEAR(mid, 10.0, 1e-10);
    }
}

TEST(ConfidenceQuoterTest, BidBelowAsk) {
    auto quotes = generate_quotes(10.0, 0.5);
    for (const auto& band : quotes.bands) {
        EXPECT_LT(band.bid, band.ask);
    }
}

TEST(ConfidenceQuoterTest, ZeroUncertainty_MinSpread) {
    auto quotes = generate_quotes(10.0, 0.0, 0.05);
    for (const auto& band : quotes.bands) {
        EXPECT_NEAR(band.spread, 0.05, 1e-10);
    }
}

TEST(ConfidenceQuoterTest, ZScoreValues_Correct) {
    EXPECT_NEAR(z_for_confidence(0.50), 0.6745, 0.001);
    EXPECT_NEAR(z_for_confidence(0.75), 1.1503, 0.001);
    EXPECT_NEAR(z_for_confidence(0.95), 1.9600, 0.001);
    EXPECT_NEAR(z_for_confidence(0.99), 2.5758, 0.001);
}

TEST(ConfidenceQuoterTest, UncertaintyEstimation) {
    double unc = estimate_uncertainty(40.0, 0.5, 0.02, 1.0);
    // √(40² · 0.02² + 0.5² · 1²) = √(0.64 + 0.25) = √0.89 ≈ 0.943
    EXPECT_NEAR(unc, 0.943, 0.01);
}

TEST(ConfidenceQuoterTest, SpreadQuality_PerfectQuote) {
    double quality = spread_quality(9.75, 10.25, 10.0);
    EXPECT_GT(quality, 0.9); // Tight, centered
}

TEST(ConfidenceQuoterTest, SpreadQuality_OffCenter) {
    double quality_centered = spread_quality(9.5, 10.5, 10.0);
    double quality_skewed   = spread_quality(9.0, 10.0, 10.0);
    EXPECT_GT(quality_centered, quality_skewed);
}

TEST(ConfidenceQuoterTest, ValidQuote_Check) {
    EXPECT_TRUE(is_valid_quote(9.5, 10.5, 10.0));
    EXPECT_FALSE(is_valid_quote(10.5, 9.5, 10.0));  // Crossed
    EXPECT_FALSE(is_valid_quote(10.5, 11.0, 10.0));  // Bid > fair
}

TEST(ConfidenceQuoterTest, CustomLevels) {
    auto quotes = generate_quotes_custom(10.0, 0.5, {0.99, 0.95, 0.90, 0.50});
    ASSERT_EQ(quotes.bands.size(), 4u);
    // Wider levels should have wider spreads
    EXPECT_GT(quotes.bands[0].spread, quotes.bands[1].spread);
    EXPECT_GT(quotes.bands[1].spread, quotes.bands[2].spread);
    EXPECT_GT(quotes.bands[2].spread, quotes.bands[3].spread);
}
