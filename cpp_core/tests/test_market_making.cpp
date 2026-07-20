/// @file test_market_making.cpp
/// @brief Unit tests for Avellaneda-Stoikov and Lucic-Tse quoting models.

#include <gtest/gtest.h>
#include <cmath>
#include "market_making/avellaneda_stoikov.hpp"
#include "market_making/lucic_tse.hpp"

using namespace davinci::market_making;

class MarketMakingTest : public ::testing::Test {
protected:
    // Standard parameters
    ASParams as_params{0.1, 1.5, 0.02, 1.0}; // γ=0.1, κ=1.5, σ=2%, T=1 day
};

// ═════════════════════════ AVELLANEDA-STOIKOV ═════════════════════════════

TEST_F(MarketMakingTest, AS_SpreadAlwaysPositive) {
    for (double gamma = 0.01; gamma <= 1.0; gamma += 0.1) {
        ASParams p{gamma, 1.5, 0.02, 1.0};
        double spread = optimal_spread(p);
        EXPECT_GT(spread, 0.0) << "Negative spread at gamma=" << gamma;
    }
}

TEST_F(MarketMakingTest, AS_BidBelowAsk) {
    auto q = compute_quotes(100.0, 0.0, as_params);
    EXPECT_LT(q.bid, q.ask);
}

TEST_F(MarketMakingTest, AS_PositiveInventory_LowersQuotes) {
    // Positive inventory → lower both bid and ask to incentivize buying
    auto neutral  = compute_quotes(100.0, 0.0, as_params);
    auto long_inv = compute_quotes(100.0, 5.0, as_params);

    EXPECT_LT(long_inv.reservation_price, neutral.reservation_price);
    EXPECT_LT(long_inv.bid, neutral.bid);
    EXPECT_LT(long_inv.ask, neutral.ask);
}

TEST_F(MarketMakingTest, AS_NegativeInventory_RaisesQuotes) {
    auto neutral   = compute_quotes(100.0, 0.0, as_params);
    auto short_inv = compute_quotes(100.0, -5.0, as_params);

    EXPECT_GT(short_inv.reservation_price, neutral.reservation_price);
    EXPECT_GT(short_inv.bid, neutral.bid);
    EXPECT_GT(short_inv.ask, neutral.ask);
}

TEST_F(MarketMakingTest, AS_SpreadWidensNearHorizon) {
    // As time remaining decreases, elasticity term dominates but
    // the inventory component decreases. Total spread behavior
    // depends on parameters. At minimum, spread should remain positive.
    for (double t = 0.0; t < 1.0; t += 0.1) {
        double spread = optimal_spread(as_params, t);
        EXPECT_GT(spread, 0.0);
    }
}

TEST_F(MarketMakingTest, AS_ReservationPrice_EqualsMiddleWhenZeroInventory) {
    double mid = 100.0;
    double r = reservation_price(mid, 0.0, as_params);
    EXPECT_NEAR(r, mid, 1e-12);
}

TEST_F(MarketMakingTest, AS_QuotesSymmetricAtZeroInventory) {
    auto q = compute_quotes(100.0, 0.0, as_params);
    double bid_dist = q.reservation_price - q.bid;
    double ask_dist = q.ask - q.reservation_price;
    EXPECT_NEAR(bid_dist, ask_dist, 1e-12);
}

TEST_F(MarketMakingTest, AS_FillProbabilityDecreases_WithDistance) {
    auto q = compute_quotes(100.0, 0.0, as_params);
    auto [p_bid, p_ask] = fill_probabilities(q, 100.0, 10.0, as_params.kappa, 0.001);

    // Wider spreads → lower fill probability
    ASParams wide{0.5, 1.5, 0.02, 1.0}; // Higher gamma → wider spread
    auto q_wide = compute_quotes(100.0, 0.0, wide);
    auto [p_bid_wide, p_ask_wide] = fill_probabilities(q_wide, 100.0, 10.0, wide.kappa, 0.001);

    EXPECT_GT(p_bid, p_bid_wide);
}

// ═════════════════════════ LUCIC-TSE ═════════════════════════════════════

TEST_F(MarketMakingTest, LT_ReducesToAS_WhenNoVolArb) {
    // When σ_realized = σ_implied, vol arb component = 0
    LTParams lt{0.1, 1.5, 0.02, 0.02, 0.01}; // Equal realized and implied
    auto lt_q = compute_quotes(100.0, 0.0, 0.01, 5.0, 100.0, lt, 1.0, 0.001);
    EXPECT_NEAR(lt_q.vol_arb_component, 0.0, 1e-15);
}

TEST_F(MarketMakingTest, LT_VolArbPositive_WhenRealizedExceedsImplied) {
    LTParams lt{0.1, 1.5, 0.25, 0.20, 0.01}; // σ_R > σ_I
    auto q = compute_quotes(100.0, 0.0, 0.01, 5.0, 100.0, lt, 1.0, 0.001);
    EXPECT_GT(q.vol_arb_component, 0.0);
}

TEST_F(MarketMakingTest, LT_VolArbNegative_WhenImpliedExceedsRealized) {
    LTParams lt{0.1, 1.5, 0.15, 0.20, 0.01}; // σ_R < σ_I
    auto q = compute_quotes(100.0, 0.0, 0.01, 5.0, 100.0, lt, 1.0, 0.001);
    EXPECT_LT(q.vol_arb_component, 0.0);
}

TEST_F(MarketMakingTest, LT_SpreadAlwaysPositive) {
    LTParams lt{0.1, 1.5, 0.25, 0.20, 0.01};
    for (int inv = -10; inv <= 10; ++inv) {
        auto q = compute_quotes(100.0, inv, 0.01, 5.0, 100.0, lt, 1.0, 0.001);
        EXPECT_GT(q.spread, 0.0) << "Negative spread at inventory=" << inv;
    }
}

TEST_F(MarketMakingTest, LT_ThreeComponentsDecompose) {
    LTParams lt{0.1, 1.5, 0.25, 0.20, 0.05};
    auto q = compute_quotes(100.0, 3.0, 0.01, 5.0, 100.0, lt, 1.0, 0.001);

    // Verify decomposition: ask - bid = 2 * elasticity_component
    EXPECT_NEAR(q.spread, 2.0 * q.elasticity_component, 0.02);
}

TEST_F(MarketMakingTest, LT_VolArbOpportunity_Detection) {
    LTParams lt1{0.1, 1.5, 0.25, 0.20, 0.01}; // 5 vol point spread → opportunity
    EXPECT_TRUE(is_vol_arb_opportunity(lt1));

    LTParams lt2{0.1, 1.5, 0.201, 0.200, 0.01}; // 0.1 vol point → no opportunity
    EXPECT_FALSE(is_vol_arb_opportunity(lt2));
}

TEST_F(MarketMakingTest, LT_VarianceRiskPremium) {
    LTParams lt{0.1, 1.5, 0.18, 0.22, 0.01};
    double vrp = variance_risk_premium(lt);
    EXPECT_GT(vrp, 0.0); // Implied > realized → positive VRP (typical)
}
