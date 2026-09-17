#pragma once

#include <cstddef>

extern "C" double nn_dot_product_f64(
    const double* left,
    const double* right,
    std::size_t length
);
