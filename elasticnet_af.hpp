/// \file
/// elasticnet_af library
///
/// Copyright Filip Matzner 2024.
///
/// Use, modification and distribution is subject to the
/// MIT License. See the accompanying file LICENSE.
///
/// Project home: https://github.com/FloopCZ/elasticnet_af

#pragma once

#include <algorithm>
#include <arrayfire.h>
#include <cassert>
#include <fmt/format.h>
#include <numeric>
#include <random>
#include <stdexcept>

namespace elasticnet_af {

/// Generate a geometric sequence of numbers (based on NumPy geomspace).
///
/// Expects two matrices of dimensions (1, n) with desired start and end values.
/// Returns a matrix of dimensions (num, n) with the corresponding geometric sequence in each
/// column.
inline af::array geomspace(const af::array& start, const af::array& end, long num)
{
    assert(start.dims(0) == 1 && end.dims(0) == 1 && start.elements() == end.elements());
    long n_seqs = start.elements();
    af::array flip_mask = af::flat(end < start);
    af::array log_start = af::log10(af::min(start, end));
    af::array log_end = af::log10(af::max(start, end));
    af::array xs = af::tile(af::seq(num), 1, n_seqs).as(start.type());
    xs = af::pow(10., log_start + xs * (log_end - log_start) / (num - 1));
    xs(af::span, flip_mask) = af::flip(xs(af::span, flip_mask), 0);
    return xs;
}

/// Return 1 for positive values, 0 for zero and -1 for negative values.
inline af::array signum(const af::array& arr)
{
    return -af::sign(arr) + af::sign(-arr);
}

/// ElasticNet optimizer.
///
/// Combination of ridge and lasso regression with the L1 and L2 regularization.
///
/// The ElasticNet model solves the optimization problem:
///     min_{B} 1/2 * ||Y - X @ B||^2 + lambda * (alpha * ||B||_1 + (1 - alpha) * ||B||_2^2)
/// where X is the input matrix with dimensions (n_samples, n_features), Y is the output matrix
/// with dimensions (n_samples, n_targets), B is the coefficients matrix with dimensions
/// (n_features, n_targets), lambda is the regularization strength, and alpha is the mixing
/// parameter between L1 and L2 regularization.
///
/// See the object constructor for more details on the parameters.
class ElasticNet {
public:
    struct options {
        double lambda = 0.01;
        double alpha = 0.5;
        double tol = 1e-8;
        long path_len = 100;
        long max_grad_steps = 1000;
        bool standardize_var = true;
        bool warm_start = false;
        bool random_index = true;
    };

protected:
    options opts_;

    af::array mean_;
    af::array std_;
    af::array nonzero_std_;
    af::array B_star_;
    af::array intercept_;

    /// Store the mean and std of the input matrix.
    void store_standardization_stats(const af::array& X)
    {
        mean_ = af::mean(X, 0);
        std_ = af::stdev(X, AF_VARIANCE_POPULATION, 0);
        nonzero_std_ = af::where(af::flat(std_));
    }

    /// Standardize the input matrix to have zero mean and optionally unit variance.
    ///
    /// Uses the standardization coefficients stored in the object.
    /// Constant columns are left unchanged.
    af::array standardize(af::array X) const
    {
        if (X.dims(1) != mean_.dims(1))
            throw std::invalid_argument(fmt::format(
              "The input matrix has a different number of columns ({}) than the stored mean ({}).",
              X.dims(1), mean_.dims(1)));
        X(af::span, nonzero_std_) -= mean_(0, nonzero_std_);
        if (opts_.standardize_var) X(af::span, nonzero_std_) /= std_(0, nonzero_std_);
        return X;
    }

public:
    /// Create an ElasticNet object with the given parameters.
    ///
    /// \param lambda The strength of the regularization.
    /// \param alpha The mixing parameter between L1 and L2 regularization. The
    ///              larger the alpha, the more L1 regularization is used.
    /// \param tol The tolerance in the coefficients update for the convergence of the algorithm.
    /// \param path_len The number of lambda values to use in the pathwise descent.
    /// \param max_grad_steps The maximum number of gradient steps to take.
    /// \param standardize_var Standardize the input matrix to unit variance.
    /// \param warm_start Warm start the algorithm from ridge regression solution. In this case,
    ///                   the path_len parameter must be set to 1, because the first path step
    ///                   lambda is set such that all the coefficients are zero.
    /// \param random_index Randomly shuffle indices before every coordinate descent loop.
    /// \throws std::invalid_argument If the parameters are invalid.
    ElasticNet(options opts) : opts_(std::move(opts))
    {
        if (opts_.lambda <= 0) throw std::invalid_argument("The lambda must be positive.");
        if (opts_.alpha < 0 || opts_.alpha > 1)
            throw std::invalid_argument("The alpha must be between 0 and 1.");
        if (opts_.tol <= 0) throw std::invalid_argument("The tolerance must be positive.");
        if (opts_.path_len < 0)
            throw std::invalid_argument("The path length must be non-negative.");
        if (opts_.warm_start && opts_.path_len != 1)
            throw std::invalid_argument("The path length must be set to 1 when using warm start.");
    }

    /// Fit the ElasticNet model to the given input-output data.
    ///
    /// \param X The input matrix with dimensions (n_samples, n_features).
    /// \param Y The output matrix with dimensions (n_samples, n_targets).
    /// \throws std::invalid_argument If the input matrices have incompatible dimensions.
    /// \return True if the algorithm converged, false otherwise.
    bool fit(af::array X, af::array Y)
    {
        const long n_samples = X.dims(0);
        const long n_predictors = X.dims(1);
        const long n_targets = Y.dims(1);
        const af::dtype type = X.type();

        // Check the parameters.
        if (Y.dims(0) != n_samples)
            throw std::invalid_argument(fmt::format(
              "The input matrix X has {} rows, but the output matrix Y has {} rows.", n_samples,
              Y.dims(0)));
        if (Y.type() != type) throw std::invalid_argument("X and Y must have the same data type.");

        // Standardize the predictors.
        store_standardization_stats(X);
        if (nonzero_std_.elements() == 0)
            throw std::invalid_argument("All columns of the input matrix are constant.");
        X = standardize(std::move(X));
        X = X(af::span, nonzero_std_);  // Remove constant columns (std == 0).
        const long n_nonconst_predictors = X.dims(1);

        // Subtract the intercept from the targets.
        intercept_ = af::mean(Y, 0);
        Y -= intercept_;

        // Initial guess are zero coefficients.
        B_star_ = af::constant(0, n_nonconst_predictors, n_targets, type);

        // Warm start from the ridge regression solution if requested.
        if (opts_.warm_start) {
            af::array reg = std::sqrt(opts_.lambda * (1. - opts_.alpha))
              * af::identity(X.dims(1), X.dims(1), X.type());
            af::array X_reg = af::join(0, X, std::move(reg));
            af::array Y_reg = af::join(0, Y, af::constant(0, {X.dims(1), Y.dims(1)}, Y.type()));
            B_star_ = af::solve(X_reg, Y_reg);
        }

        // Generate "the path" of lambda values for each target.
        const af::array lambda_path = [&]() {
            if (opts_.path_len == 0) return af::array{};
            if (opts_.path_len == 1 || opts_.alpha < opts_.tol)
                return af::constant(opts_.lambda, opts_.path_len, n_targets, type);
            const af::array lambda_max =
              af::max(af::abs(af::matmulTN(X, Y)), 0) / n_samples / opts_.alpha;
            return geomspace(
              lambda_max, af::constant(opts_.lambda, 1, n_targets, type), opts_.path_len);
        }();

        // Precompute covariance matrices.
        const af::array X_X_covs = af::matmulTN(X, X);
        const af::array Y_X_covs = af::matmulTN(Y, X);

        // Coordinate array and random generator for selecting random indices.
        std::vector<long> idxs(n_nonconst_predictors);
        std::iota(idxs.begin(), idxs.end(), 0L);
        std::seed_seq seed_seq{n_predictors, n_targets, n_samples, opts_.path_len};
        std::minstd_rand idx_prng{seed_seq};

        // Run the coordinate graient descent.
        bool converged = true;
        for (long path_step = 0; path_step < opts_.path_len; ++path_step) {
            af::array lambda = lambda_path(path_step, af::span);
            long grad_step = 0;
            for (; grad_step < opts_.max_grad_steps; ++grad_step) {
                af::array B = B_star_;
                if (opts_.random_index) std::shuffle(idxs.begin(), idxs.end(), idx_prng);

                // Use the covariance update rule with the precomputed Gram matrices.
                for (long j : idxs) {
                    af::array cov_update = Y_X_covs(af::span, j).T()
                      - af::matmulTN(X_X_covs(af::span, j), B_star_)
                      + X_X_covs(j, j) * B_star_(j, af::span);
                    af::array soft_update =
                      signum(cov_update) * af::max(af::abs(cov_update) - lambda * opts_.alpha, 0.);
                    B_star_(j, af::span) =
                      soft_update / (X_X_covs(j, j) + lambda * (1. - opts_.alpha));
                }

                // Terminating condition.
                af::array B_star_max = af::max(af::abs(B_star_), 0);
                af::array delta_ratio = af::max(af::abs(B_star_ - B), 0) / B_star_max;
                if (af::count<long>(B_star_max) == 0 || af::allTrue<bool>(delta_ratio < opts_.tol))
                    break;
            }
            if (grad_step == opts_.max_grad_steps) {
                converged = false;
                break;
            }
        }

        // Adapt the coefficients and intercept to non-standardized predictors.
        if (opts_.standardize_var) B_star_ /= af::tile(std_(0, nonzero_std_).T(), 1, n_targets);
        intercept_ -= af::matmul(mean_(0, nonzero_std_), B_star_);
        // Extend the coefficients to the full predictor matrix including the constant columns.
        af::array B_star_full = af::constant(0, n_predictors, n_targets, type);
        B_star_full(nonzero_std_, af::span) = B_star_;
        B_star_ = B_star_full;
        return converged;
    }

    /// Predict the output values for the given input matrix.
    ///
    /// \param X The input matrix with dimensions (n_samples, n_features).
    /// \return The predicted output matrix with dimensions (n_samples, n_targets).
    /// \throws std::logic_error If the model has not been fitted yet.
    /// \throws std::invalid_argument If the input matrix has incompatible dimensions.
    af::array predict(const af::array& X) const
    {
        if (B_star_.isempty()) throw std::logic_error("The model has not been fitted yet.");
        if (X.dims(1) != mean_.dims(1))
            throw std::invalid_argument(fmt::format(
              "The input matrix has a different number of columns ({}) than the fitted matrix "
              "({}).",
              X.dims(1), mean_.dims(1)));
        return intercept_ + af::matmul(X, B_star_);
    }

    /// Compute the ElasticNet cost.
    ///
    /// \param X The input matrix with dimensions (n_samples, n_features).
    /// \param Y The output matrix with dimensions (n_samples, n_targets).
    /// \return The cost for each target dimensions (1, n_targets).
    /// \throws std::logic_error Forwarded from \ref predict.
    /// \throws std::invalid_argument If the output matrix has incompatible dimensions or forwarded
    ///         from \ref predict..
    af::array cost(const af::array& X, const af::array& Y) const
    {
        if (Y.dims(1) != intercept_.dims(1))
            throw std::invalid_argument(fmt::format(
              "The output matrix has a different number of columns ({}) than the fitted matrix "
              "({}).",
              Y.dims(1), intercept_.dims(1)));
        af::array sse = af::sum(af::pow(Y - predict(X), 2.), 0) / 2.;
        af::array l2 = opts_.lambda * (1. - opts_.alpha) * af::sum(B_star_ * B_star_, 0) / 2.;
        af::array l1 = opts_.lambda * opts_.alpha * af::sum(af::abs(B_star_), 0);
        return sse + l2 + l1;
    }

    /// Return the coefficients of the model.
    ///
    /// \param intercept Whether to include the intercept as the zeroth coefficient.
    /// \return The coefficients matrix with dimensions (n_features, n_targets).
    /// \throws std::logic_error If the model has not been fitted yet.
    af::array coefficients(bool intercept = false) const
    {
        if (B_star_.isempty()) throw std::logic_error("The model has not been fitted yet.");
        if (intercept) return af::join(0, intercept_, B_star_);
        return B_star_;
    }

    /// Return the intercept of the model.
    ///
    /// \return The intercept vector with dimensions (1, n_targets).
    /// \throws std::logic_error If the model has not been fitted yet.
    af::array intercept() const
    {
        if (B_star_.isempty()) throw std::logic_error("The model has not been fitted yet.");
        return intercept_;
    }
};

};  // namespace elasticnet_af