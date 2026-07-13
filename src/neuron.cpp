#include "neuron.hpp"

#include <stdexcept>

double Neuron::forward(const std::vector<double>& input) {
    if (input.size() != weights_.size())
        throw std::invalid_argument(
            "Neuron::forward: input size does not match weight count"
        );

    lastInput_ = input;

    double z = bias_;
    for (std::size_t i = 0; i < weights_.size(); ++i)
        z += weights_[i] * input[i];

    lastOutput_ = activation_.forward(z);

    return lastOutput_;
}

std::vector<double> Neuron::backward(double dLoss_dOutput) {
    double dOutput_dZ = activation_.derivativeFromOutput(lastOutput_);
    double delta = dLoss_dOutput * dOutput_dZ;

    gradWeights_.assign(weights_.size(), 0.0);
    for (std::size_t i = 0; i < weights_.size(); ++i)
        gradWeights_[i] = delta * lastInput_[i];

    gradBias_ = delta;

    std::vector<double> dLoss_dInput(weights_.size());
    for (std::size_t i = 0; i < weights_.size(); ++i)
        dLoss_dInput[i] = delta * weights_[i];

    return dLoss_dInput;
}

void Neuron::applyGradients(double learningRate) {
    for (std::size_t i = 0; i < weights_.size(); ++i)
        weights_[i] -= learningRate * gradWeights_[i];
    bias_ -= learningRate * gradBias_;
}