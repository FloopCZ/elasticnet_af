# ElasticNet AF
### A simple and efficient implementation of multi-target ElasticNet with GPU support.

Searching for an efficient linear regression optimizer with both L1 (lasso) and L2 (ridge) regularization and GPU support?
Search no more.

## Features

ElasticNet optimizer using coordinate gradient descent with covariance update rule.
Supports:
- warm-starting from ridge regression solution
- random index selection
- nonregularized intercept
- lambda regularization paths
- input standardization to unit variance
- precomputing Gram matrices
- observation weights
- GPU (Cuda, OpenCL) and CPU through [Arrayfire](https://arrayfire.org/)

## Example
```C++
    // Generate data.
    af::array X = af::randu(100, 10);
    af::array Y = af::randu(100, 3);

    // Configure
    ElasticNet enet{{.lambda = 100, .alpha = 0.01}};

    // Fit.
    if (!enet.fit(X, Y)) std::cout << "The ElasticNet optimization did not converge.\n";

    // Predict.
    af::array Y_hat = enet.predict(X);
    std::cout << "MSE: " << af::mean<double>(af::pow(Y - Y_hat, 2)) << "\n";
    std::cout << "cost: " << af::mean<double>(enet.cost(X, Y)) << "\n";
```

For a full example, see [example.cpp](example.cpp).

For more details and supported parameters, see the [ElasticNet](elasticnet_af.hpp) class.

## Build

Use as a single-header library with dependencies to [fmt](https://fmt.dev/) and [Arrayfire](https://arrayfire.org/).
Currently requires ArrayFire >= 3.9.0, if you need support for older versions, let me know.

To build the example, run the following commands:
```bash
cmake -B build
cmake --build build
./build/example
```

## References

Based on the following paper:

```
@article{JSSv033i01,
 title={Regularization Paths for Generalized Linear Models via Coordinate Descent},
 volume={33},
 url={https://www.jstatsoft.org/index.php/jss/article/view/v033i01},
 doi={10.18637/jss.v033.i01},
 journal={Journal of Statistical Software},
 author={Friedman, Jerome H. and Hastie, Trevor and Tibshirani, Rob},
 year={2010},
 pages={1–22}
}
```
