# Neural Network

A neural network from scratch with no external dependencies.

## Building

Requires `g++` (or any C++17-capable compiler).

```sh
make          # build everything (tests, experiments, bench)
make test     # run the test suite
make run      # XOR experiment
make circle   # circle classification experiment
make bench    # FastLayer vs Layer benchmark
make clean    # remove binaries
```

## Project Structure

```
include/           Public headers
  neuron.hpp       Neuron (weights, bias, activation)
  layer.hpp        Layer (vector of neurons)
  fastLayer.hpp    FastLayer (flat-buffer layer, function-pointer activations)
  network.hpp      NeuralNetwork (chains layers, forward/backward/train)
  activations.hpp  Sigmoid, Tanh, ReLU (std::function-based)
  losses.hpp       Mean Squared Error, Binary Cross Entropy
  serialize.hpp    Save/load network weights to text files
src/               Implementations (neuron, layer, fastLayer, serialize)
tests/             Test suite with gradient checks
experiments/       XOR, circle classification, benchmark
```

## Usage

```cpp
#include "network.hpp"
#include "losses.hpp"

NeuralNetwork net({
    Layer({
        Neuron({0.5, -0.3}, 0.1, Tanh),
        Neuron({0.8, 0.2}, -0.5, Tanh)
    }),
    Layer({Neuron({0.4, -0.6}, 0.0, Sigmoid)})
});

// Train
net.trainStep({0.0, 1.0}, {1.0}, 0.8, meanSquaredError);

// Predict
std::vector<double> out = net.predict({0.0, 1.0});

// Save / load
saveNetwork(net, "model.txt");
NeuralNetwork loaded = loadNetwork("model.txt");
```

## Activation Functions

| Name    | Forward       | Derivative (from output) |
|---------|---------------|--------------------------|
| Sigmoid | 1 / (1 + e^-z) | y * (1 - y)            |
| Tanh    | tanh(z)       | 1 - y^2                  |
| ReLU    | max(0, z)     | y > 0 ? 1 : 0           |

## Loss Functions

- **Mean Squared Error** (`meanSquaredError`): general purpose regression
- **Binary Cross Entropy** (`binaryCrossEntropy`): binary classification

## Serialization

Networks can be saved and loaded from plain text files:

```cpp
saveNetwork(net, "model.txt");
NeuralNetwork loaded = loadNetwork("model.txt");
```

The format stores layer count, neuron count per layer, and for each neuron the activation name, bias, and all weights.
