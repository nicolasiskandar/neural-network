#pragma once

#include <cstddef>

// Internal numerical-kernel ABI. All routines use the System V AMD64 C ABI;
// callers retain ownership of every input and output buffer.
extern "C" double
nn_dot_product_f64(const double* left, const double* right, std::size_t length);

extern "C" double nn_relu_f64(double value);
extern "C" double nn_relu_derivative_from_output_f64(double output);
extern "C" double nn_exp_approx_f64(double value);
extern "C" double nn_sigmoid_f64(double value);
extern "C" double nn_sigmoid_derivative_from_output_f64(double output);
extern "C" double nn_tanh_f64(double value);
extern "C" double nn_tanh_derivative_from_output_f64(double output);

enum NNActivationKind {
    NN_ACTIVATION_SIGMOID = 0,
    NN_ACTIVATION_TANH = 1,
    NN_ACTIVATION_RELU = 2,
};

extern "C" void nn_fast_layer_forward_f64(
    const double* input,
    const double* weights,
    const double* biases,
    double* output,
    std::size_t numInputs,
    std::size_t numOutputs,
    int activationKind
);

extern "C" void nn_fast_layer_backward_f64(
    const double* input,
    const double* weights,
    const double* output,
    const double* dLoss_dOutput,
    double* gradWeights,
    double* gradBiases,
    double* dLoss_dInput,
    std::size_t numInputs,
    std::size_t numOutputs,
    int activationKind
);

extern "C" void nn_apply_gradients_f64(
    double* values,
    const double* gradients,
    std::size_t count,
    double learningRate
);

extern "C" double nn_neuron_forward_f64(
    const double* input,
    const double* weights,
    std::size_t count,
    double bias,
    int activationKind
);
