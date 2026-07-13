#include "layer.hpp"

std::vector<double> Layer::forward(const std::vector<double>& input) {
    inputSize_ = input.size();
    std::vector<double> output;
    output.reserve(neurons_.size());
    for (Neuron& neuron : neurons_) output.push_back(neuron.forward(input));
    return output;
}

std::vector<double> Layer::backward(const std::vector<double>& dLoss_dOutput) {
    std::vector<double> dLoss_dInput(inputSize_, 0.0);
    for (std::size_t i = 0; i < neurons_.size(); ++i) {
        std::vector<double> dIn = neurons_[i].backward(dLoss_dOutput[i]);
        for (std::size_t j = 0; j < dIn.size(); ++j) dLoss_dInput[j] += dIn[j];
    }
    return dLoss_dInput;
}

void Layer::applyGradients(double learningRate) {
    for (Neuron& neuron : neurons_) neuron.applyGradients(learningRate);
}