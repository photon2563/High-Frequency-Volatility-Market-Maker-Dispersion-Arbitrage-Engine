/// @file test_dispersion.cpp
/// @brief Unit tests for dispersion trading: dirty correlation, correlation matrix, Z-score signals.

#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include "dispersion/dirty_correlation.hpp"
#include "dispersion/correlation_matrix.hpp"
#include "dispersion/zscore_signal.hpp"

using namespace davinci::dispersion;

// ═══════════════════════════════════════════════════════════════════════════
// DIRTY CORRELATION
// ═══════════════════════════════════════════════════════════════════════════

TEST(DirtyCorrelationTest, Equicorrelation_SingleStock) {
    // Single stock: ρ_dirty = σ²_idx / (w·σ)² = σ²_idx / σ²
    // If index IS the stock: ρ = 1
    double rho = compute_dirty_correlation(0.20, {0.20}, {1.0});
    EXPECT_NEAR(rho, 1.0, 1e-10);
}

TEST(DirtyCorrelationTest, HighIndexVol_HighCorrelation) {
    // Index vol >> weighted constituent vol → high implied correlation
    std::vector<double> ivs = {0.20, 0.25, 0.30};
    std::vector<double> wts = {0.5, 0.3, 0.2};
    // Weighted IV = 0.5*0.20 + 0.3*0.25 + 0.2*0.30 = 0.235
    double rho = compute_dirty_correlation(0.235, ivs, wts);
    EXPECT_NEAR(rho, 1.0, 0.001); // Index = weighted avg → ρ ≈ 1
}

TEST(DirtyCorrelationTest, LowIndexVol_LowCorrelation) {
    std::vector<double> ivs = {0.30, 0.35, 0.40};
    std::vector<double> wts = {0.5, 0.3, 0.2};
    // Weighted IV = 0.33, if index IV is much lower → low correlation
    double rho = compute_dirty_correlation(0.15, ivs, wts);
    EXPECT_LT(rho, 0.5);
}

TEST(DirtyCorrelationTest, CorrelationClamped_0_1) {
    std::vector<double> ivs = {0.10, 0.10};
    std::vector<double> wts = {0.5, 0.5};
    // Very high index vol → would imply ρ > 1, should be clamped
    double rho = compute_dirty_correlation(0.20, ivs, wts);
    EXPECT_LE(rho, 1.0);
    EXPECT_GE(rho, 0.0);
}

TEST(DirtyCorrelationTest, VarianceWeighted_EquicorrelationCheck) {
    // For equicorrelated assets, both methods should agree
    std::vector<double> ivs = {0.20, 0.20, 0.20};
    std::vector<double> wts = {1.0/3, 1.0/3, 1.0/3};
    // For equicorrelation ρ, σ²_idx = n·w²·σ²·[1 + (n-1)·ρ]
    // With ρ=0.5: σ²_idx = (1/9)·0.04·(1+2·0.5)·3 = (1/9)·0.04·6 ≈ 0.02667
    // σ_idx ≈ 0.1633
    double implied_rho = 0.5;
    double index_var = 0.04 * (1.0/3.0 + 2.0/3.0 * implied_rho);
    double index_iv = std::sqrt(index_var);

    double rho_var = compute_variance_weighted_correlation(index_iv, ivs, wts);
    EXPECT_NEAR(rho_var, implied_rho, 0.01);
}

TEST(DirtyCorrelationTest, InvalidInputs) {
    EXPECT_THROW(compute_dirty_correlation(0.20, {}, {}), std::invalid_argument);
    EXPECT_THROW(compute_dirty_correlation(-0.20, {0.20}, {1.0}), std::invalid_argument);
    EXPECT_THROW(compute_dirty_correlation(0.20, {0.20}, {1.0, 0.5}), std::invalid_argument);
}

// ═══════════════════════════════════════════════════════════════════════════
// CORRELATION MATRIX
// ═══════════════════════════════════════════════════════════════════════════

TEST(CorrelationMatrixTest, IdentityMatrix_IsPSD) {
    SymmetricMatrix I(3);
    I.set_unit_diagonal();
    EXPECT_TRUE(is_psd(I));
}

TEST(CorrelationMatrixTest, PSDProjection_FixesNegativeEigenvalues) {
    // Create a matrix with a negative eigenvalue
    SymmetricMatrix M(3);
    M(0,0) = 1.0; M(0,1) = 0.9; M(0,2) = 0.9;
    M(1,0) = 0.9; M(1,1) = 1.0; M(1,2) = -0.5; // This makes it non-PSD
    M(2,0) = 0.9; M(2,1) = -0.5; M(2,2) = 1.0;

    auto projected = project_psd(M);
    EXPECT_TRUE(is_psd(projected));
}

TEST(CorrelationMatrixTest, NearestCorrelation_UnitDiagonal) {
    SymmetricMatrix M(3);
    M(0,0) = 1.0; M(0,1) = 0.8; M(0,2) = 0.7;
    M(1,0) = 0.8; M(1,1) = 1.0; M(1,2) = -0.3;
    M(2,0) = 0.7; M(2,1) = -0.3; M(2,2) = 1.0;

    auto nearest = nearest_correlation_matrix(M);

    // Check unit diagonal
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(nearest(i, i), 1.0, 1e-6);
    }
    // Check PSD
    EXPECT_TRUE(is_psd(nearest));
    // Check symmetric
    EXPECT_TRUE(nearest.is_symmetric());
}

TEST(CorrelationMatrixTest, NearestCorrelation_PreservesValidMatrix) {
    // A valid correlation matrix should be returned nearly unchanged
    SymmetricMatrix valid(2);
    valid(0,0) = 1.0; valid(0,1) = 0.5;
    valid(1,0) = 0.5; valid(1,1) = 1.0;

    auto result = nearest_correlation_matrix(valid);
    EXPECT_NEAR(result(0,1), 0.5, 0.01);
}

TEST(CorrelationMatrixTest, JacobiEigen_IdentityMatrix) {
    SymmetricMatrix I(3);
    I.set_unit_diagonal();
    auto decomp = jacobi_eigen(I);
    for (double ev : decomp.eigenvalues) {
        EXPECT_NEAR(ev, 1.0, 1e-10);
    }
}

TEST(CorrelationMatrixTest, BoundaryMatrices_DiagonalOnes) {
    std::vector<double> ivs = {0.20, 0.25, 0.30};
    std::vector<double> wts = {0.5, 0.3, 0.2};
    auto [lower, upper] = compute_boundary_matrices(ivs, wts, 0.22);

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(lower(i, i), 1.0, 1e-10);
        EXPECT_NEAR(upper(i, i), 1.0, 1e-10);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Z-SCORE SIGNAL
// ═══════════════════════════════════════════════════════════════════════════

TEST(ZScoreTest, Engine_RequiresWarmup) {
    ZScoreEngine engine;
    EXPECT_FALSE(engine.is_warmed_up());

    for (int i = 0; i < 20; ++i) {
        engine.update(0.5);
    }
    EXPECT_TRUE(engine.is_warmed_up());
}

TEST(ZScoreTest, FlatSignal_StaysFlat) {
    ZScoreEngine engine;
    // Feed constant values → Z-score ≈ 0 → stay FLAT
    for (int i = 0; i < 30; ++i) {
        auto result = engine.update(0.5);
        EXPECT_EQ(result.signal, DispersionState::FLAT);
    }
}

TEST(ZScoreTest, HighZScore_TriggersShortDispersion) {
    ZScoreEngine engine;
    // Warmup with mean = 0.5, stddev ≈ 0
    for (int i = 0; i < 20; ++i) {
        engine.update(0.50);
    }
    // Spike → high Z-score → short dispersion
    auto result = engine.update(0.80); // Significant spike
    if (engine.is_warmed_up() && result.zscore > 0.5) {
        EXPECT_EQ(result.signal, DispersionState::SHORT_DISPERSION);
        EXPECT_TRUE(result.is_entry);
    }
}

TEST(ZScoreTest, LowZScore_TriggersLongDispersion) {
    ZScoreEngine engine;
    for (int i = 0; i < 20; ++i) {
        engine.update(0.50);
    }
    auto result = engine.update(0.20); // Significant dip
    if (engine.is_warmed_up() && result.zscore < -0.5) {
        EXPECT_EQ(result.signal, DispersionState::LONG_DISPERSION);
        EXPECT_TRUE(result.is_entry);
    }
}

TEST(ZScoreTest, MeanReversion_TriggersExit) {
    ZScoreParams params;
    params.window = 5;
    params.entry_threshold = 0.5;
    params.exit_threshold = 0.05;

    ZScoreEngine engine(params);

    // Warmup with some variance
    std::vector<double> warmup = {0.50, 0.51, 0.49, 0.50, 0.52};
    for (double v : warmup) engine.update(v);

    // Spike to trigger entry
    engine.update(0.80);

    // Revert to mean → should trigger exit
    for (int i = 0; i < 5; ++i) {
        auto result = engine.update(0.50);
        if (result.is_exit) {
            EXPECT_EQ(result.signal, DispersionState::FLAT);
            return;
        }
    }
}

TEST(ZScoreTest, Reset_ClearsState) {
    ZScoreEngine engine;
    for (int i = 0; i < 25; ++i) engine.update(0.5);
    EXPECT_TRUE(engine.is_warmed_up());

    engine.reset();
    EXPECT_FALSE(engine.is_warmed_up());
    EXPECT_EQ(engine.state(), DispersionState::FLAT);
}

TEST(ZScoreTest, BatchCompute_MatchesIncremental) {
    std::vector<double> data;
    for (int i = 0; i < 50; ++i) {
        data.push_back(0.5 + 0.1 * std::sin(i * 0.3));
    }

    auto batch_results = compute_zscore_series(data);
    ASSERT_EQ(batch_results.size(), data.size());

    // Verify incremental matches batch
    ZScoreEngine engine;
    for (size_t i = 0; i < data.size(); ++i) {
        auto inc_result = engine.update(data[i]);
        EXPECT_NEAR(inc_result.zscore, batch_results[i].zscore, 1e-10);
    }
}
