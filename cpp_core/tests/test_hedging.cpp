/// @file test_hedging.cpp
/// @brief Unit tests for Leland model, Central Risk Book, and Vega Neutralizer.

#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include "hedging/leland.hpp"
#include "hedging/central_risk_book.hpp"
#include "hedging/vega_neutralizer.hpp"

using namespace davinci::hedging;

// ═══════════════════════════════════════════════════════════════════════════
// LELAND MODEL
// ═══════════════════════════════════════════════════════════════════════════

class LelandTest : public ::testing::Test {
protected:
    // Standard params: C=10bps, σ=20%, daily rebalancing
    double C = 0.001, sigma = 0.20, dt = 1.0 / 252.0;
};

TEST_F(LelandTest, LelandNumber_HandCalculation) {
    // Le = √(2/π) · C / (σ · √Δt)
    // Le = 0.7979 · 0.001 / (0.20 · √(1/252))
    // Le = 0.7979 · 0.001 / (0.20 · 0.06299)
    // Le = 0.7979 · 0.001 / 0.012599
    // Le ≈ 0.0633
    double le = leland_number(C, sigma, dt);
    EXPECT_NEAR(le, 0.0633, 0.001);

    // ── Sanity check: mental math ──
    // Quick mental math: C/σ = 0.005, √(2/π) ≈ 0.8, √Δt ≈ 0.063
    // Le ≈ 0.8 · 0.005 / 0.063 ≈ 0.063 ✓
    std::cout << "[SANITY CHECK] Leland number: " << le
              << " (mental math: ~0.063)" << std::endl;
}

TEST_F(LelandTest, EffectiveVol_LongGamma_Increases) {
    double le = leland_number(C, sigma, dt);
    double eff = effective_volatility(sigma, le, 1.0); // Long Gamma
    EXPECT_GT(eff, sigma);
}

TEST_F(LelandTest, EffectiveVol_ShortGamma_Decreases) {
    double le = leland_number(C, sigma, dt);
    double eff = effective_volatility(sigma, le, -1.0); // Short Gamma
    EXPECT_LT(eff, sigma);
}

TEST_F(LelandTest, EffectiveVol_ZeroCost_Unchanged) {
    double le = leland_number(0.0, sigma, dt);
    EXPECT_NEAR(le, 0.0, 1e-15);
    double eff = effective_volatility(sigma, le, 1.0);
    EXPECT_NEAR(eff, sigma, 1e-10);
}

TEST_F(LelandTest, HedgeBand_Width_PositiveForNonZeroGamma) {
    auto band = compute_hedge_band(0.5, 0.03, 100.0, C, sigma, dt);
    EXPECT_GT(band.band_width, 0.0);
    EXPECT_LT(band.band_lower, band.delta_target);
    EXPECT_GT(band.band_upper, band.delta_target);
}

TEST_F(LelandTest, HedgeBand_ZeroGamma_ZeroWidth) {
    auto band = compute_hedge_band(0.5, 0.0, 100.0, C, sigma, dt);
    EXPECT_NEAR(band.band_width, 0.0, 1e-15);
}

TEST_F(LelandTest, HedgeDecision_WithinBand_Hold) {
    auto band = compute_hedge_band(0.5, 0.03, 100.0, C, sigma, dt);
    auto order = evaluate_hedge(0.5, band, 100.0, C);
    EXPECT_EQ(order.decision, HedgeDecision::HOLD);
    EXPECT_NEAR(order.shares_to_trade, 0.0, 1e-15);
}

TEST_F(LelandTest, HedgeDecision_OutsideBand_Rebalance) {
    auto band = compute_hedge_band(0.5, 0.03, 100.0, C, sigma, dt);
    auto order = evaluate_hedge(0.5 + band.band_width, band, 100.0, C);
    EXPECT_EQ(order.decision, HedgeDecision::REBALANCE);
    EXPECT_NE(order.shares_to_trade, 0.0);
}

TEST_F(LelandTest, ProxyHedge_PerfectCorrelation_NoBasicRisk) {
    // ρ = 1 → no basis risk → proxy is always better (if cheaper)
    double benefit = proxy_hedge_benefit(
        1.0, sigma, sigma, 0.03, 100.0,
        0.002, 0.001, dt);
    EXPECT_GT(benefit, 0.0); // Proxy is cheaper and perfectly correlated
}

// ═══════════════════════════════════════════════════════════════════════════
// CENTRAL RISK BOOK
// ═══════════════════════════════════════════════════════════════════════════

class CRBTest : public ::testing::Test {
protected:
    CRBParams params{0.0005, 0.005, 0.20, 0.03, 100.0};
};

TEST_F(CRBTest, InnerZone_Hold) {
    auto order = evaluate_hedge_action(100.0, 100.0, 100.0, params);
    EXPECT_EQ(order.type, OrderType::HOLD);
}

TEST_F(CRBTest, MiddleZone_LimitOrder) {
    double inner = params.inner_threshold();
    double outer = params.outer_threshold();
    double delta_actual = 100.0 + (inner + outer) / 2.0; // In middle zone

    auto order = evaluate_hedge_action(delta_actual, 100.0, 100.0, params);
    EXPECT_EQ(order.type, OrderType::LIMIT_ORDER);
    EXPECT_GT(order.urgency, 0.0);
    EXPECT_LT(order.urgency, 1.0);
}

TEST_F(CRBTest, OuterZone_MarketOrder) {
    double outer = params.outer_threshold();
    double delta_actual = 100.0 + outer * 2.0; // Far beyond outer

    auto order = evaluate_hedge_action(delta_actual, 100.0, 100.0, params);
    EXPECT_EQ(order.type, OrderType::MARKET_ORDER);
    EXPECT_NEAR(order.urgency, 1.0, 1e-10);
}

TEST_F(CRBTest, LimitOrder_BuyDirectionCorrect) {
    double inner = params.inner_threshold();
    double delta_actual = 100.0 - inner * 3; // Below target

    auto order = evaluate_hedge_action(delta_actual, 100.0, 100.0, params);
    if (order.type == OrderType::LIMIT_ORDER || order.type == OrderType::MARKET_ORDER) {
        EXPECT_GT(order.shares, 0.0); // Need to buy
    }
}

TEST_F(CRBTest, Thresholds_OuterExceedsInner) {
    EXPECT_GT(params.outer_threshold(), params.inner_threshold());
}

TEST_F(CRBTest, CRBSavings_Positive) {
    double savings = estimated_crb_savings(0.7, 0.005, 50.0, 0.8);
    EXPECT_GT(savings, 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// VEGA NEUTRALIZER
// ═══════════════════════════════════════════════════════════════════════════

class VegaNeutTest : public ::testing::Test {
protected:
    PortfolioGreeks portfolio{0.5, 0.03, 15.0, -0.05, 0.5, 1.0};
    // vega = 15.0 → need to offset
};

TEST_F(VegaNeutTest, VarswapNeutralization_ResidualNearZero) {
    auto result = neutralize_with_varswap(portfolio, 0.20, 1.0);
    EXPECT_TRUE(result.is_neutralized);
    EXPECT_LT(std::abs(result.portfolio_vega_after), 1e-6);
}

TEST_F(VegaNeutTest, StraddleNeutralization_ResidualNearZero) {
    auto result = neutralize_with_straddle(portfolio, 100.0, 1.0, 0.20);
    EXPECT_TRUE(result.is_neutralized);
    EXPECT_LT(std::abs(result.portfolio_vega_after), 1e-6);
}

TEST_F(VegaNeutTest, VarswapUnits_CorrectSign) {
    // Positive portfolio Vega → need to sell variance
    double units = varswap_units_for_neutralization(15.0, 0.20, 1.0);
    EXPECT_LT(units, 0.0); // Short variance

    // Negative portfolio Vega → need to buy variance
    double units_neg = varswap_units_for_neutralization(-15.0, 0.20, 1.0);
    EXPECT_GT(units_neg, 0.0); // Long variance
}

TEST_F(VegaNeutTest, StraddleVega_MatchesMentalMath) {
    // Mental math: ATM Vega ≈ S · √T / √(2π) ≈ 100 · 1 · 0.3989 ≈ 39.89
    // Straddle Vega = 2 × call Vega ≈ 2 × 39.89 ≈ 79.79
    double straddle_v = atm_straddle_vega(100.0, 1.0);
    EXPECT_NEAR(straddle_v, 79.79, 0.5);

    std::cout << "[SANITY CHECK] ATM straddle Vega: " << straddle_v
              << " (mental math: ~79.79)" << std::endl;
}

TEST_F(VegaNeutTest, GammaScalpingPnL_PositiveWhenRealizedExceedsImplied) {
    double pnl = gamma_scalping_pnl(0.03, 100.0, 0.25, 0.20);
    EXPECT_GT(pnl, 0.0);
}

TEST_F(VegaNeutTest, GammaScalpingPnL_NegativeWhenImpliedExceedsRealized) {
    double pnl = gamma_scalping_pnl(0.03, 100.0, 0.15, 0.20);
    EXPECT_LT(pnl, 0.0);
}

TEST_F(VegaNeutTest, GammaScalpingPnL_ZeroWhenEqual) {
    double pnl = gamma_scalping_pnl(0.03, 100.0, 0.20, 0.20);
    EXPECT_NEAR(pnl, 0.0, 1e-15);
}
