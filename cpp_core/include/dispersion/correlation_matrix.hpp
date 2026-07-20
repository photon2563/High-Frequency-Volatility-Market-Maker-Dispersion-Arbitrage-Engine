#pragma once
/// @file correlation_matrix.hpp
/// @brief Positive semi-definite (PSD) projection and boundary matrices
///        for implied correlation in dispersion trading.
///
/// Implements Higham's alternating projection algorithm for nearest
/// correlation matrix, plus boundary matrices from IV relationships.

#include <cmath>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <cassert>

namespace davinci {
namespace dispersion {

// ─────────────────────────────────────────────────────────────────────────────
// Simple matrix type (row-major) — avoids Eigen dependency for core logic
// ─────────────────────────────────────────────────────────────────────────────

class SymmetricMatrix {
public:
    explicit SymmetricMatrix(size_t n) : n_(n), data_(n * n, 0.0) {}

    size_t size() const { return n_; }

    double& operator()(size_t i, size_t j) { return data_[i * n_ + j]; }
    double  operator()(size_t i, size_t j) const { return data_[i * n_ + j]; }

    /// Set diagonal to 1.0 (correlation matrix normalization)
    void set_unit_diagonal() {
        for (size_t i = 0; i < n_; ++i) data_[i * n_ + i] = 1.0;
    }

    /// Check if matrix is symmetric within tolerance
    bool is_symmetric(double tol = 1e-10) const {
        for (size_t i = 0; i < n_; ++i)
            for (size_t j = i + 1; j < n_; ++j)
                if (std::abs(data_[i*n_+j] - data_[j*n_+i]) > tol)
                    return false;
        return true;
    }

    /// Force symmetry: A = (A + A^T) / 2
    void symmetrize() {
        for (size_t i = 0; i < n_; ++i)
            for (size_t j = i + 1; j < n_; ++j) {
                double avg = 0.5 * (data_[i*n_+j] + data_[j*n_+i]);
                data_[i*n_+j] = avg;
                data_[j*n_+i] = avg;
            }
    }

    /// Clamp all off-diagonal elements to [-1, 1]
    void clamp_correlations() {
        for (size_t i = 0; i < n_; ++i)
            for (size_t j = 0; j < n_; ++j)
                if (i != j)
                    data_[i*n_+j] = std::clamp(data_[i*n_+j], -1.0, 1.0);
    }

    /// Frobenius norm
    double frobenius_norm() const {
        double sum = 0.0;
        for (double v : data_) sum += v * v;
        return std::sqrt(sum);
    }

    /// Frobenius distance to another matrix
    double frobenius_distance(const SymmetricMatrix& other) const {
        assert(n_ == other.n_);
        double sum = 0.0;
        for (size_t i = 0; i < data_.size(); ++i) {
            double d = data_[i] - other.data_[i];
            sum += d * d;
        }
        return std::sqrt(sum);
    }

    const std::vector<double>& data() const { return data_; }
    std::vector<double>& data() { return data_; }

private:
    size_t n_;
    std::vector<double> data_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Eigendecomposition (Jacobi method for small symmetric matrices)
// ─────────────────────────────────────────────────────────────────────────────

/// Simple Jacobi eigenvalue decomposition for small symmetric matrices.
/// Suitable for correlation matrices (typically n ≤ 50 constituents).
///
/// Returns eigenvalues in ascending order.
struct EigenDecomp {
    std::vector<double> eigenvalues;
    SymmetricMatrix eigenvectors; // Columns are eigenvectors

    EigenDecomp(size_t n) : eigenvalues(n, 0.0), eigenvectors(n) {}
};

/// Jacobi rotation method for symmetric eigenvalue decomposition.
/// O(n³) per sweep, typically 5-10 sweeps for convergence.
inline EigenDecomp jacobi_eigen(const SymmetricMatrix& A, int max_sweeps = 50, double tol = 1e-12) {
    size_t n = A.size();
    EigenDecomp result(n);

    // Work copy
    SymmetricMatrix M(n);
    for (size_t i = 0; i < n * n; ++i) M.data()[i] = A.data()[i];

    // Initialize eigenvectors to identity
    for (size_t i = 0; i < n; ++i) result.eigenvectors(i, i) = 1.0;

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        // Check for convergence: sum of squared off-diag elements
        double off_diag_sum = 0.0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n; ++j)
                off_diag_sum += M(i, j) * M(i, j);

        if (off_diag_sum < tol) break;

        for (size_t p = 0; p < n; ++p) {
            for (size_t q = p + 1; q < n; ++q) {
                if (std::abs(M(p, q)) < tol * 0.01) continue;

                // Compute rotation angle
                double diff = M(q, q) - M(p, p);
                double t; // tan(theta)
                if (std::abs(M(p, q)) < 1e-15 * std::abs(diff)) {
                    t = M(p, q) / diff;
                } else {
                    double phi = diff / (2.0 * M(p, q));
                    t = 1.0 / (std::abs(phi) + std::sqrt(phi * phi + 1.0));
                    if (phi < 0.0) t = -t;
                }

                double c = 1.0 / std::sqrt(t * t + 1.0); // cos(theta)
                double s = t * c;                          // sin(theta)
                double tau = s / (1.0 + c);

                // Update matrix
                double mp_q = M(p, q);
                M(p, q) = 0.0;
                M(p, p) -= t * mp_q;
                M(q, q) += t * mp_q;

                for (size_t r = 0; r < p; ++r) {
                    double g = M(r, p), h = M(r, q);
                    M(r, p) = g - s * (h + g * tau);
                    M(r, q) = h + s * (g - h * tau);
                }
                for (size_t r = p + 1; r < q; ++r) {
                    double g = M(p, r), h = M(r, q);
                    M(p, r) = g - s * (h + g * tau);
                    M(r, q) = h + s * (g - h * tau);
                }
                for (size_t r = q + 1; r < n; ++r) {
                    double g = M(p, r), h = M(q, r);
                    M(p, r) = g - s * (h + g * tau);
                    M(q, r) = h + s * (g - h * tau);
                }

                // Update eigenvectors
                for (size_t r = 0; r < n; ++r) {
                    double g = result.eigenvectors(r, p);
                    double h = result.eigenvectors(r, q);
                    result.eigenvectors(r, p) = g - s * (h + g * tau);
                    result.eigenvectors(r, q) = h + s * (g - h * tau);
                }
            }
        }
    }

    // Extract eigenvalues from diagonal
    for (size_t i = 0; i < n; ++i)
        result.eigenvalues[i] = M(i, i);

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// PSD projection (Higham's algorithm)
// ─────────────────────────────────────────────────────────────────────────────

/// Project a symmetric matrix onto the cone of positive semi-definite matrices.
///
/// Clamps negative eigenvalues to ε > 0 and reconstructs.
/// Σ_hat = Q · max(Λ, ε·I) · Q^T
///
/// @param M Input symmetric matrix
/// @param min_eigenvalue Floor for eigenvalues (default 1e-8)
/// @returns PSD-projected matrix
inline SymmetricMatrix project_psd(
    const SymmetricMatrix& M, double min_eigenvalue = 1e-8)
{
    size_t n = M.size();
    auto decomp = jacobi_eigen(M);

    // Clamp negative eigenvalues
    for (size_t i = 0; i < n; ++i) {
        decomp.eigenvalues[i] = std::max(decomp.eigenvalues[i], min_eigenvalue);
    }

    // Reconstruct: Σ = Q · Λ · Q^T
    SymmetricMatrix result(n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i; j < n; ++j) {
            double val = 0.0;
            for (size_t k = 0; k < n; ++k) {
                val += decomp.eigenvectors(i, k) * decomp.eigenvalues[k]
                     * decomp.eigenvectors(j, k);
            }
            result(i, j) = val;
            result(j, i) = val;
        }
    }

    return result;
}

/// Higham's nearest correlation matrix algorithm.
///
/// Alternating projection between:
///   S_u: set of symmetric PSD matrices
///   S_c: set of matrices with unit diagonal
///
/// Iterates until convergence or max_iter reached.
///
/// @param input Input approximate correlation matrix
/// @param max_iter Maximum iterations
/// @param tol Convergence tolerance
/// @returns Nearest valid correlation matrix (PSD with unit diagonal)
inline SymmetricMatrix nearest_correlation_matrix(
    const SymmetricMatrix& input,
    int max_iter = 100, double tol = 1e-8)
{
    size_t n = input.size();
    SymmetricMatrix Y = input;     // Working copy
    SymmetricMatrix dS(n);         // Dykstra correction

    for (int iter = 0; iter < max_iter; ++iter) {
        SymmetricMatrix Y_prev = Y;

        // Project onto PSD cone (with Dykstra correction)
        SymmetricMatrix R(n);
        for (size_t i = 0; i < n * n; ++i)
            R.data()[i] = Y.data()[i] - dS.data()[i];

        SymmetricMatrix X = project_psd(R);

        // Update Dykstra correction
        for (size_t i = 0; i < n * n; ++i)
            dS.data()[i] = X.data()[i] - R.data()[i];

        // Project onto unit diagonal set
        Y = X;
        Y.set_unit_diagonal();

        // Clamp off-diagonal to [-1, 1]
        Y.clamp_correlations();

        // Check convergence
        if (Y.frobenius_distance(Y_prev) < tol) break;
    }

    return Y;
}

// ─────────────────────────────────────────────────────────────────────────────
// Boundary matrices from implied volatilities
// ─────────────────────────────────────────────────────────────────────────────

/// Compute correlation boundary matrices from constituent implied volatilities.
///
/// Upper bound: ρ_ij^max = min[(σ²_i + σ²_j - (σ_i - σ_j)²) / (2σ_iσ_j), 1]
///            = min[2σ_iσ_j / (2σ_iσ_j), 1] = 1 (trivially)
///
/// More useful practical bounds come from the index variance constraint:
///   σ²_index = Σ w_i w_j ρ_ij σ_i σ_j
///
/// @param constituent_ivs Implied vols
/// @param weights Market-cap weights
/// @param index_iv Index implied vol
/// @returns Pair of {lower_bound, upper_bound} correlation matrices
inline std::pair<SymmetricMatrix, SymmetricMatrix> compute_boundary_matrices(
    const std::vector<double>& constituent_ivs,
    const std::vector<double>& weights,
    double index_iv)
{
    size_t n = constituent_ivs.size();
    SymmetricMatrix lower(n), upper(n);

    for (size_t i = 0; i < n; ++i) {
        lower(i, i) = 1.0;
        upper(i, i) = 1.0;

        for (size_t j = i + 1; j < n; ++j) {
            double si = constituent_ivs[i];
            double sj = constituent_ivs[j];

            // Theoretical bounds from triangle inequality
            // |σ_i - σ_j| ≤ σ_portfolio ≤ σ_i + σ_j
            // This constrains ρ_ij

            // Lower bound: can be negative (especially for uncorrelated assets)
            double rho_min = -1.0;

            // Upper bound: always ≤ 1
            double rho_max = 1.0;

            // Tighter bounds from index constraint:
            // Given the index vol and weights, each pairwise correlation
            // is bounded by the implied equicorrelation ± spread
            if (si > 0 && sj > 0) {
                // From Cauchy-Schwarz: ρ_ij ≥ -1
                // From PSD requirement on the full matrix
                double ratio = index_iv * index_iv;
                double own_var = 0.0;
                for (size_t k = 0; k < n; ++k)
                    own_var += weights[k] * weights[k] * constituent_ivs[k] * constituent_ivs[k];

                double cross_sum = 0.0;
                for (size_t k = 0; k < n; ++k)
                    for (size_t l = k + 1; l < n; ++l)
                        if (k != i || l != j)
                            cross_sum += 2.0 * weights[k] * weights[l]
                                       * constituent_ivs[k] * constituent_ivs[l];

                double wiwj_sisj = 2.0 * weights[i] * weights[j] * si * sj;
                if (std::abs(wiwj_sisj) > 1e-15) {
                    // ρ_ij = (σ²_idx - own_var - cross_sum * ρ_avg) / (w_i w_j σ_i σ_j)
                    // Bound by assuming other correlations at their extremes
                    double remaining = ratio - own_var - cross_sum; // if all others ρ=0
                    rho_min = std::max(-1.0, (remaining - cross_sum) / wiwj_sisj);
                    rho_max = std::min(1.0, (remaining + cross_sum) / wiwj_sisj);
                }
            }

            rho_min = std::max(rho_min, -1.0);
            rho_max = std::min(rho_max, 1.0);

            lower(i, j) = rho_min;
            lower(j, i) = rho_min;
            upper(i, j) = rho_max;
            upper(j, i) = rho_max;
        }
    }

    return {lower, upper};
}

/// Check if a matrix is positive semi-definite (all eigenvalues ≥ 0).
inline bool is_psd(const SymmetricMatrix& M, double tol = -1e-10) {
    auto decomp = jacobi_eigen(M);
    for (double ev : decomp.eigenvalues) {
        if (ev < tol) return false;
    }
    return true;
}

} // namespace dispersion
} // namespace davinci
