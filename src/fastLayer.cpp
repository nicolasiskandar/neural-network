#include "fastLayer.hpp"

std::vector<double> FastLayer::forward(const std::vector<double>& input) {
    lastInput_ = input;
    lastOutput_.resize(numOutputs_);
    for (std::size_t o = 0; o < numOutputs_; ++o) {
        double z = biases_[o];
        const double* row = &weights_[o * numInputs_];
        for (std::size_t i = 0; i < numInputs_; ++i) z += row[i] * input[i];
        lastOutput_[o] = activation_.forward(z);
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
        double delta =
            dLoss_dOutput[o] * activation_.derivativeFromOutput(lastOutput_[o]);
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