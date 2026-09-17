#pragma once

#include <cstddef>

extern "C" double
nn_dot_product_f64(const double* left, const double* right, std::size_t length);

extern "C" double nn_relu_f64(double value);
extern "C" double nn_relu_derivative_from_output_f64(double output);
extern "C" double nn_exp_approx_f64(double value);

extern "C" double nn_sigmoid_f64(double value);
extern "C" double nn_sigmoid_derivative_from_output_f64(double output);
