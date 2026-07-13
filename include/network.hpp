#pragma once
#include <functional>
#include <vector>

#include "layer.hpp"
#include "losses.hpp"

using LossFn = std::function<
    LossResult(const std::vector<double>&, const std::vector<double>&)>;

class NeuralNetwork {
   public:
    explicit NeuralNetwork(std::vector<Layer> layers)
        : layers_(std::move(layers)) {}

    std::vector<double> predict(const std::vector<double>& input) {
        std::vector<double> current = input;
        for (Layer& layer : layers_) current = layer.forward(current);
        return current;
    }

    double trainStep(
        const std::vector<double>& input,
        const std::vector<double>& target,
        double learningRate,
        const LossFn& lossFn
    ) {
        std::vector<double> output = predict(input);
        LossResult result = lossFn(output, target);

        std::vector<double> grad = result.dLoss_dOutput;
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
            grad = it->backward(grad);

        for (Layer& layer : layers_) layer.applyGradients(learningRate);

        return result.loss;
    }

   private:
    std::vector<Layer> layers_;
};