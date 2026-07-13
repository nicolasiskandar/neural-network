#pragma once
#include <vector>

#include "activations.hpp"

class Neuron {
   public:
    Neuron(std::vector<double> weights, double bias, ActivationFn activation)
        : weights_(std::move(weights)),
          bias_(bias),
          activation_(std::move(activation)) {}

    double forward(const std::vector<double>& input) const;

    const std::vector<double>& weights() const { return weights_; }
    double bias() const { return bias_; }

   private:
    std::vector<double> weights_;
    double bias_;
    ActivationFn activation_;
};
