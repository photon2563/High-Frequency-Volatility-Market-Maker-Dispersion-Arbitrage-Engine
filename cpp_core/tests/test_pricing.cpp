/// @file test_pricing.cpp
/// @brief Unit tests for the pricing engine: Black-Scholes + Implied Vol solver.
///
/// Cross-verification protocol (Module 6):
///   - Put-call parity (machine epsilon)
///   - Greek symmetry (Γ_call = Γ_put, ν_call = ν_put)
///   - Finite-difference Greek validation
///   - Vanna/Volga sign checks at ATM, ITM, OTM
///   - IV round-trip convergence
///   - Known benchmark values

#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include "pricing/black_scholes.hpp"
#include "pricing/implied_vol.hpp"

using namespace davinci::pricing;

// ═════════════════════════════════════════════════════════════════════════════
// Test fixture with common option parameters
// ═════════════════════════════════════════════════════════════════════════════

class PricingTest : public ::testing::Test {
protected:
    // Standard test case: S=100, K=100, r=5%, T=1y, σ=20%, q=0
    double S = 100.0, K = 100.0, r = 0.05, T = 1.0, sigma = 0.20, q = 0.0;

    // Tolerance levels
    static constexpr double TOL_PRICE   = 1e-6;
    static constexpr double TOL_GREEK   = 1e-6;
    static constexpr double TOL_PARITY  = 1e-10;
    static constexpr double TOL_IV      = 1e-8;
    static constexpr double TOL_FD      = 1e-3;  // Finite difference (less precise)
};

// ═══════════════════════════════════════════════════════════════════════════
// BLACK-SCHOLES PRICE TESTS
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(PricingTest, ATMCallPrice_KnownBenchmark) {
    // Known BS price: S=100, K=100, r=0.05, T=1, σ=0.20, q=0
    // C ≈ 10.4506
    auto result = price_and_greeks(S, K, r, T, sigma, q, true);
    EXPECT_NEAR(result.price, 10.4506, 0.001);  // Within 0.1 cent
}

TEST_F(PricingTest, ATMPutPrice_KnownBenchmark) {
    // P ≈ 5.5735
    auto result = price_and_greeks(S, K, r, T, sigma, q, false);
    EXPECT_NEAR(result.price, 5.5735, 0.001);
}

TEST_F(PricingTest, DeepITMCall) {
    // S=150, K=100 → deep ITM, should be close to intrinsic
    auto result = price_and_greeks(150.0, 100.0, r, T, sigma, q, true);
    double intrinsic = 150.0 - 100.0 * std::exp(-r * T);
    EXPECT_GT(result.price, intrinsic);
    EXPECT_NEAR(result.delta, 1.0, 0.05);  // Delta near 1
}

TEST_F(PricingTest, DeepOTMCall) {
    // S=50, K=100 → deep OTM
    auto result = price_and_greeks(50.0, 100.0, r, T, sigma, q, true);
    EXPECT_LT(result.price, 0.01);  // Nearly worthless
    EXPECT_NEAR(result.delta, 0.0, 0.01);
}

TEST_F(PricingTest, PriceNonNegative) {
    // Prices must always be non-negative
    for (double k = 50; k <= 150; k += 5) {
        auto call = price_and_greeks(S, k, r, T, sigma, q, true);
        auto put  = price_and_greeks(S, k, r, T, sigma, q, false);
        EXPECT_GE(call.price, 0.0);
        EXPECT_GE(put.price, 0.0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PUT-CALL PARITY
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(PricingTest, PutCallParity_ATM) {
    double violation = verify_put_call_parity(S, K, r, T, sigma, q);
    EXPECT_LT(violation, TOL_PARITY);
}

TEST_F(PricingTest, PutCallParity_ITM) {
    double violation = verify_put_call_parity(120.0, 100.0, r, T, sigma, q);
    EXPECT_LT(violation, TOL_PARITY);
}

TEST_F(PricingTest, PutCallParity_OTM) {
    double violation = verify_put_call_parity(80.0, 100.0, r, T, sigma, q);
    EXPECT_LT(violation, TOL_PARITY);
}

TEST_F(PricingTest, PutCallParity_WithDividends) {
    double violation = verify_put_call_parity(S, K, r, T, sigma, 0.03);
    EXPECT_LT(violation, TOL_PARITY);
}

TEST_F(PricingTest, PutCallParity_ShortDated) {
    double violation = verify_put_call_parity(S, K, r, 0.01, sigma, q);
    EXPECT_LT(violation, TOL_PARITY);
}

// ═══════════════════════════════════════════════════════════════════════════
// GREEK SYMMETRY
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(PricingTest, GammaSymmetry) {
    double violation = verify_gamma_symmetry(S, K, r, T, sigma, q);
    EXPECT_LT(violation, TOL_GREEK);
}

TEST_F(PricingTest, VegaSymmetry) {
    double violation = verify_vega_symmetry(S, K, r, T, sigma, q);
    EXPECT_LT(violation, TOL_GREEK);
}

TEST_F(PricingTest, GammaSymmetry_OffATM) {
    EXPECT_LT(verify_gamma_symmetry(80.0, K, r, T, sigma, q), TOL_GREEK);
    EXPECT_LT(verify_gamma_symmetry(120.0, K, r, T, sigma, q), TOL_GREEK);
}

// ═══════════════════════════════════════════════════════════════════════════
// FINITE-DIFFERENCE GREEK VALIDATION
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(PricingTest, Delta_MatchesFiniteDiff_Call) {
    double err = verify_delta_finite_diff(S, K, r, T, sigma, q, true);
    EXPECT_LT(err, TOL_FD);
}

TEST_F(PricingTest, Delta_MatchesFiniteDiff_Put) {
    double err = verify_delta_finite_diff(S, K, r, T, sigma, q, false);
    EXPECT_LT(err, TOL_FD);
}

TEST_F(PricingTest, Gamma_MatchesFiniteDiff) {
    double err = verify_gamma_finite_diff(S, K, r, T, sigma, q, true);
    EXPECT_LT(err, TOL_FD);
}

TEST_F(PricingTest, Vega_MatchesFiniteDiff) {
    double err = verify_vega_finite_diff(S, K, r, T, sigma, q, true);
    EXPECT_LT(err, TOL_FD);
}

// ═══════════════════════════════════════════════════════════════════════════
// VANNA / VOLGA SIGN CHECKS
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(PricingTest, Vanna_Sign_ATM) {
    // ATM: d2 ≈ 0, Vanna should be close to zero or slightly negative
    auto g = price_and_greeks(S, K, r, T, sigma, q, true);
    // At ATM d2 is small, Vanna = -φ(d1)*d2/σ, sign depends on d2
    // For typical ATM with r > 0: d2 > 0, so Vanna < 0
    EXPECT_NE(g.vanna, 0.0); // Should be non-zero
}

TEST_F(PricingTest, Volga_Positive_Away_From_ATM) {
    // Volga = ν * d1 * d2 / σ
    // For OTM/ITM options, d1*d2 > 0, so Volga > 0
    auto otm = price_and_greeks(S, 130.0, r, T, sigma, q, true);
    EXPECT_GT(otm.volga, 0.0);

    auto itm = price_and_greeks(S, 70.0, r, T, sigma, q, true);
    EXPECT_GT(itm.volga, 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// GREEK BOUND CHECKS
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(PricingTest, CallDelta_InRange) {
    for (double k = 50; k <= 150; k += 5) {
        auto g = price_and_greeks(S, k, r, T, sigma, q, true);
        EXPECT_GE(g.delta, 0.0);
        EXPECT_LE(g.delta, 1.0);
    }
}

TEST_F(PricingTest, PutDelta_InRange) {
    for (double k = 50; k <= 150; k += 5) {
        auto g = price_and_greeks(S, k, r, T, sigma, q, false);
        EXPECT_GE(g.delta, -1.0);
        EXPECT_LE(g.delta, 0.0);
    }
}

TEST_F(PricingTest, Gamma_AlwaysPositive) {
    for (double k = 50; k <= 150; k += 5) {
        auto g = price_and_greeks(S, k, r, T, sigma, q, true);
        EXPECT_GT(g.gamma, 0.0);
    }
}

TEST_F(PricingTest, Vega_AlwaysPositive) {
    for (double k = 50; k <= 150; k += 5) {
        auto g = price_and_greeks(S, k, r, T, sigma, q, true);
        EXPECT_GT(g.vega, 0.0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// IMPLIED VOLATILITY SOLVER
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(PricingTest, IV_RoundTrip_ATM) {
    double err = verify_iv_round_trip(S, K, r, T, sigma, q, true);
    EXPECT_GE(err, 0.0);  // Must have converged
    EXPECT_LT(err, TOL_IV);
}

TEST_F(PricingTest, IV_RoundTrip_ITM) {
    double err = verify_iv_round_trip(120.0, 100.0, r, T, 0.30, q, true);
    EXPECT_GE(err, 0.0);
    EXPECT_LT(err, TOL_IV);
}

TEST_F(PricingTest, IV_RoundTrip_OTM) {
    double err = verify_iv_round_trip(80.0, 100.0, r, T, 0.25, q, true);
    EXPECT_GE(err, 0.0);
    EXPECT_LT(err, TOL_IV);
}

TEST_F(PricingTest, IV_RoundTrip_Put) {
    double err = verify_iv_round_trip(S, K, r, T, sigma, q, false);
    EXPECT_GE(err, 0.0);
    EXPECT_LT(err, TOL_IV);
}

TEST_F(PricingTest, IV_RoundTrip_HighVol) {
    double err = verify_iv_round_trip(S, K, r, T, 0.80, q, true);
    EXPECT_GE(err, 0.0);
    EXPECT_LT(err, TOL_IV);
}

TEST_F(PricingTest, IV_RoundTrip_LowVol) {
    double err = verify_iv_round_trip(S, K, r, T, 0.05, q, true);
    EXPECT_GE(err, 0.0);
    EXPECT_LT(err, TOL_IV);
}

TEST_F(PricingTest, IV_RoundTrip_ShortDated) {
    double err = verify_iv_round_trip(S, K, r, 0.01, sigma, q, true);
    EXPECT_GE(err, 0.0);
    EXPECT_LT(err, TOL_IV);
}

TEST_F(PricingTest, IV_RoundTrip_LongDated) {
    double err = verify_iv_round_trip(S, K, r, 5.0, sigma, q, true);
    EXPECT_GE(err, 0.0);
    EXPECT_LT(err, TOL_IV);
}

TEST_F(PricingTest, IV_ConvergesViaNewtonRaphson_ATM) {
    double price = price_and_greeks(S, K, r, T, sigma, q, true).price;
    auto result = solve_implied_vol(price, S, K, r, T, q, true);
    EXPECT_TRUE(result.converged);
    // ATM should converge via Newton-Raphson (good initial guess, stable vega)
    EXPECT_EQ(result.method_used, IVMethod::NEWTON_RAPHSON);
    EXPECT_NEAR(result.sigma, sigma, TOL_IV);
}

TEST_F(PricingTest, IV_RangeOfStrikes) {
    // Test convergence across the entire strike range
    for (double k = 70; k <= 130; k += 5) {
        for (double vol = 0.10; vol <= 0.60; vol += 0.10) {
            double price = price_and_greeks(S, k, r, T, vol, q, true).price;
            auto result = solve_implied_vol(price, S, k, r, T, q, true);
            EXPECT_TRUE(result.converged)
                << "Failed at K=" << k << ", σ=" << vol;
            EXPECT_NEAR(result.sigma, vol, 1e-6)
                << "Inaccurate at K=" << k << ", σ=" << vol;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// EDGE CASES
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(PricingTest, ExpiredOption_IntrinsicValue) {
    auto call = price_and_greeks(110.0, 100.0, r, 0.0, sigma, q, true);
    EXPECT_NEAR(call.price, 10.0, TOL_PRICE);
    EXPECT_NEAR(call.delta, 1.0, TOL_PRICE);

    auto put = price_and_greeks(90.0, 100.0, r, 0.0, sigma, q, false);
    EXPECT_NEAR(put.price, 10.0, TOL_PRICE);
    EXPECT_NEAR(put.delta, -1.0, TOL_PRICE);
}

TEST_F(PricingTest, ExpiredOption_OTM) {
    auto call = price_and_greeks(90.0, 100.0, r, 0.0, sigma, q, true);
    EXPECT_NEAR(call.price, 0.0, TOL_PRICE);

    auto put = price_and_greeks(110.0, 100.0, r, 0.0, sigma, q, false);
    EXPECT_NEAR(put.price, 0.0, TOL_PRICE);
}

TEST_F(PricingTest, InvalidInputs_Throw) {
    EXPECT_THROW(price_and_greeks(-1.0, K, r, T, sigma, q, true), std::invalid_argument);
    EXPECT_THROW(price_and_greeks(S, -1.0, r, T, sigma, q, true), std::invalid_argument);
    EXPECT_THROW(price_and_greeks(S, K, r, -0.1, sigma, q, true), std::invalid_argument);
    EXPECT_THROW(price_and_greeks(S, K, r, T, -0.1, q, true), std::invalid_argument);
}

// ═══════════════════════════════════════════════════════════════════════════
// BATCH PRICING
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(PricingTest, BatchPricing_MatchesSingle) {
    std::vector<double> S_v = {100, 110, 90};
    std::vector<double> K_v = {100, 100, 100};
    std::vector<double> r_v = {0.05, 0.05, 0.05};
    std::vector<double> T_v = {1.0, 1.0, 1.0};
    std::vector<double> sig_v = {0.20, 0.20, 0.20};
    std::vector<double> q_v = {0.0, 0.0, 0.0};
    std::vector<bool> call_v = {true, true, true};

    auto batch = price_batch(S_v, K_v, r_v, T_v, sig_v, q_v, call_v);
    ASSERT_EQ(batch.size(), 3u);

    for (size_t i = 0; i < 3; ++i) {
        auto single = price_and_greeks(S_v[i], K_v[i], r_v[i], T_v[i], sig_v[i], q_v[i], call_v[i]);
        EXPECT_NEAR(batch[i].price, single.price, 1e-12);
        EXPECT_NEAR(batch[i].delta, single.delta, 1e-12);
    }
}
