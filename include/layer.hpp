#pragma once
#include <vector>

#include "neuron.hpp"

class Layer {
   public:
    explicit Layer(std::vector<Neuron> neurons)
        : neurons_(std::move(neurons)) {}

    std::vector<double> forward(const std::vector<double>& input) const {
        std::vector<double> output;
        output.reserve(neurons_.size());
        for (const Neuron& neuron : neurons_)
            output.push_back(neuron.forward(input));
        return output;
    }

    std::size_t size() const { return neurons_.size(); }

   private:
    std::vector<Neuron> neurons_;
};
