#include <iomanip>
#include <iostream>
#include <random>

#include "layer.hpp"
#include "losses.hpp"
#include "neuron.hpp"

int main() {
    std::mt19937 rng(2);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    Layer hidden({
        Neuron({dist(rng), dist(rng)}, dist(rng), Tanh),
        Neuron({dist(rng), dist(rng)}, dist(rng), Tanh)
    });

    Layer output({Neuron({dist(rng), dist(rng)}, dist(rng), Sigmoid)});

    std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> targets = {{0}, {1}, {1}, {0}};

    const double learningRate = 0.8;
    const int epochs = 8000;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalLoss = 0.0;
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            std::vector<double> hiddenOut = hidden.forward(inputs[i]);
            std::vector<double> networkOut = output.forward(hiddenOut);

            LossResult result = meanSquaredError(networkOut, targets[i]);
            totalLoss += result.loss;

            std::vector<double> dLoss_dHidden =
                output.backward(result.dLoss_dOutput);
            hidden.backward(dLoss_dHidden);

            output.applyGradients(learningRate);
            hidden.applyGradients(learningRate);
        }

        if (epoch % 2000 == 0 || epoch == epochs - 1) 
            std::cout << "epoch " << epoch
                      << "  avg loss = " << totalLoss / inputs.size() << "\n";
    }

    std::cout << "\nFinal predictions:\n" << std::fixed << std::setprecision(4);
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        std::vector<double> h = hidden.forward(inputs[i]);
        std::vector<double> o = output.forward(h);
        std::cout << "  XOR(" << inputs[i][0] << ", " << inputs[i][1] << ") -> "
                  << o[0] << "  (target " << targets[i][0] << ")\n";
    }

    return 0;
}