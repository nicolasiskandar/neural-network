#include "neuron.hpp"

#include <stdexcept>

double Neuron::forward(const std::vector<double>& input) const {
    if (input.size() != weights_.size()) {
        throw std::invalid_argument(
            "Neuron::forward: input size does not match weight count"
        );
    }

    double z = bias_;
    for (std::size_t i = 0; i < weights_.size(); ++i)
        z += weights_[i] * input[i];

    return activation_(z);
}
