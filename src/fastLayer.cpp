#include "fastLayer.hpp"

#include "nn_asm.hpp"

namespace {

int assemblyActivationKind(const PlainActivation& activation) {
    if (activation.forward == sigmoidFn) return NN_ACTIVATION_SIGMOID;
    if (activation.forward == tanhFn) return NN_ACTIVATION_TANH;
    if (activation.forward == reluFn) return NN_ACTIVATION_RELU;
    return -1;
}

double applyForwardActivation(const PlainActivation& activation, double value) {
    if (activation.forward == reluFn) return nn_relu_f64(value);
    if (activation.forward == sigmoidFn) return nn_sigmoid_f64(value);
    if (activation.forward == tanhFn) return nn_tanh_f64(value);
    return activation.forward(value);
}

double
applyDerivativeActivation(const PlainActivation& activation, double output) {
    if (activation.derivativeFromOutput == reluDerivFromOutput)
        return nn_relu_derivative_from_output_f64(output);
    if (activation.derivativeFromOutput == sigmoidDerivFromOutput)
        return nn_sigmoid_derivative_from_output_f64(output);
    if (activation.derivativeFromOutput == tanhDerivFromOutput)
        return nn_tanh_derivative_from_output_f64(output);
    return activation.derivativeFromOutput(output);
}

}  // namespace

std::vector<double> FastLayer::forward(const std::vector<double>& input) {
    lastInput_ = input;
    lastOutput_.resize(numOutputs_);
    int activationKind = assemblyActivationKind(activation_);
    if (activationKind >= 0) {
        nn_fast_layer_forward_f64(
            input.data(), weights_.data(), biases_.data(), lastOutput_.data(),
            numInputs_, numOutputs_, activationKind
        );
        return lastOutput_;
    }

    for (std::size_t o = 0; o < numOutputs_; ++o) {
        const double* row = &weights_[o * numInputs_];
        double z =
            biases_[o] + nn_dot_product_f64(row, input.data(), numInputs_);
        lastOutput_[o] = applyForwardActivation(activation_, z);
    }
    return lastOutput_;
}

std::vector<double> FastLayer::backward(
    const std::vector<double>& dLoss_dOutput
) {
    gradWeights_.assign(numOutputs_ * numInputs_, 0.0);
    gradBiases_.assign(numOutputs_, 0.0);
    std::vector<double> dLoss_dInput(numInputs_, 0.0);

    for (std::size_t o = 0; o < numOutputs_; ++o) {
        double delta = dLoss_dOutput[o] *
                       applyDerivativeActivation(activation_, lastOutput_[o]);
        gradBiases_[o] = delta;
        double* gradRow = &gradWeights_[o * numInputs_];
        const double* weightRow = &weights_[o * numInputs_];
        for (std::size_t i = 0; i < numInputs_; ++i) {
            gradRow[i] = delta * lastInput_[i];
            dLoss_dInput[i] += delta * weightRow[i];
        }
    }
    return dLoss_dInput;
}

void FastLayer::applyGradients(double learningRate) {
    for (std::size_t k = 0; k < weights_.size(); ++k)
        weights_[k] -= learningRate * gradWeights_[k];
    for (std::size_t o = 0; o < numOutputs_; ++o)
        biases_[o] -= learningRate * gradBiases_[o];
}
