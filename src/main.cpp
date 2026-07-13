#include <iostream>

#include "layer.hpp"
#include "neuron.hpp"

int main() {
    Layer layer({
        Neuron({0.2, 0.4, -0.5}, 1.0, relu),
        Neuron({-0.3, 0.1, 0.2}, 0.0, sigmoid),
        Neuron({1.0, -1.0, 0.0}, 0.5, tanhActivation)
    });

    std::vector<double> input = {1.0, 2.0, 3.0};
    std::vector<double> output = layer.forward(input);

    std::cout << "Layer output: ";
    for (double v : output) std::cout << v << " ";
    std::cout << "\n";

    return 0;
}