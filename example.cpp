#include "elasticnet_af.hpp"

#include <iostream>

using elasticnet_af::ElasticNet;

int main()
{
    af::info();

    ElasticNet enet{{
      .lambda = 100,
      .alpha = 0.01,
    }};

    // Generate data.
    long n_features = 100;
    long n_samples = 1000;
    long n_targets = 10;
    af::dtype type = af::dtype::f64;
    af::array X = af::randu(n_samples, n_features, type) * af::randn(1, n_features, type) * 1e-2;
    af::array B = af::randu(n_features, n_targets, type);
    af::array Y = af::matmul(X, B) + af::randn(n_samples, n_targets) * 0.1;

    // Fit.
    if (!enet.fit(X, Y)) std::cout << "The ElasticNet optimization did not converge.\n";

    // Predict.
    af::array Y_hat = enet.predict(X);
    std::cout << "MSE: " << af::mean<double>(af::pow(Y - Y_hat, 2)) << "\n";
    std::cout << "cost: " << af::mean<double>(enet.cost(X, Y)) << "\n";

    return 0;
}