#pragma once
#include <cmath>
#include <functional>

using ActivationFn = std::function<double(double)>;

inline double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }

inline double tanhActivation(double z) { return std::tanh(z); }

inline double relu(double z) { return std::max(0.0, z); }
