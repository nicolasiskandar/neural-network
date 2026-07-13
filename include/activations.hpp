#pragma once
#include <cmath>
#include <functional>

using ActivationFn = std::function<double(double)>;

struct Activation {
    ActivationFn forward;
    ActivationFn derivativeFromOutput;
};

inline double sigmoidFn(double z) { return 1.0 / (1.0 + std::exp(-z)); }
inline double sigmoidDerivFromOutput(double y) { return y * (1.0 - y); }

inline double tanhFn(double z) { return std::tanh(z); }
inline double tanhDerivFromOutput(double y) { return 1.0 - y * y; }

inline double reluFn(double z) { return std::max(0.0, z); }
inline double reluDerivFromOutput(double y) { return y > 0.0 ? 1.0 : 0.0; }

inline const Activation Sigmoid{sigmoidFn, sigmoidDerivFromOutput};
inline const Activation Tanh{tanhFn, tanhDerivFromOutput};
inline const Activation ReLU{reluFn, reluDerivFromOutput};
