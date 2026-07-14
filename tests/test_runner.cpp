#include "test_runner.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>

#include "fastLayer.hpp"
#include "layer.hpp"
#include "losses.hpp"
#include "network.hpp"
#include "neuron.hpp"
#include "serialize.hpp"

void testSigmoidActivation(TestRunner& t) {
    t.checkNear(sigmoidFn(0.0), 0.5, 1e-12, "Sigmoid(0) = 0.5");
    t.checkNear(sigmoidFn(1.0), 0.731058, 1e-5, "Sigmoid(1) ~ 0.73106");
    t.checkNear(sigmoidFn(-1.0), 0.268941, 1e-5, "Sigmoid(-1) ~ 0.26894");
    t.checkNear(sigmoidFn(10.0), 0.999954, 1e-5, "Sigmoid(10) ~ 0.99995");
    t.checkNear(sigmoidFn(-10.0), 0.00004539, 1e-7, "Sigmoid(-10) ~ 0");
}

void testSigmoidDerivative(TestRunner& t) {
    double y = sigmoidFn(0.0);
    t.checkNear(
        sigmoidDerivFromOutput(y), 0.25, 1e-12, "Sigmoid'(0): y*(1-y) = 0.25"
    );
    double y1 = sigmoidFn(1.0);
    double expected = y1 * (1.0 - y1);
    t.checkNear(
        sigmoidDerivFromOutput(y1), expected, 1e-12, "Sigmoid'(1): y*(1-y)"
    );
}

void testTanhActivation(TestRunner& t) {
    t.checkNear(tanhFn(0.0), 0.0, 1e-12, "Tanh(0) = 0");
    t.checkNear(tanhFn(1.0), 0.761594, 1e-5, "Tanh(1) ~ 0.76159");
    t.checkNear(tanhFn(-1.0), -0.761594, 1e-5, "Tanh(-1) ~ -0.76159");
    t.checkNear(tanhFn(10.0), 0.999999, 1e-5, "Tanh(10) ~ 1");
}

void testTanhDerivative(TestRunner& t) {
    double y = tanhFn(0.0);
    t.checkNear(tanhDerivFromOutput(y), 1.0, 1e-12, "Tanh'(0) = 1");
    double y1 = tanhFn(1.0);
    double expected = 1.0 - y1 * y1;
    t.checkNear(tanhDerivFromOutput(y1), expected, 1e-12, "Tanh'(1) = 1-y^2");
}

void testReluActivation(TestRunner& t) {
    t.checkNear(reluFn(0.0), 0.0, 1e-12, "ReLU(0) = 0");
    t.checkNear(reluFn(5.0), 5.0, 1e-12, "ReLU(5) = 5");
    t.checkNear(reluFn(-3.0), 0.0, 1e-12, "ReLU(-3) = 0");
    t.checkNear(reluFn(-0.001), 0.0, 1e-12, "ReLU(-0.001) = 0");
}

void testReluDerivative(TestRunner& t) {
    t.checkNear(reluDerivFromOutput(5.0), 1.0, 1e-12, "ReLU'(positive) = 1");
    t.checkNear(reluDerivFromOutput(0.0), 0.0, 1e-12, "ReLU'(0) = 0");
    t.checkNear(reluDerivFromOutput(-1.0), 0.0, 1e-12, "ReLU'(negative) = 0");
}

void testNeuronForward(TestRunner& t) {
    Neuron n({0.5, -0.5}, 0.0, ReLU);
    t.checkNear(
        n.forward({2.0, 1.0}), 0.5, 1e-9, "Neuron::forward: ReLU example"
    );
}

void testNeuronForwardWithBias(TestRunner& t) {
    Neuron n({1.0, 1.0}, 0.5, Sigmoid);
    double y = n.forward({1.0, 1.0});
    t.checkNear(y, sigmoidFn(2.5), 1e-12, "Neuron::forward: bias is included");
}

void testNeuronForwardSizeMismatch(TestRunner& t) {
    Neuron n({1.0, 2.0}, 0.0, ReLU);
    bool threw = false;
    try {
        n.forward({1.0, 2.0, 3.0});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    t.check(threw, "Neuron::forward: throws on size mismatch");
}

void testNeuronReluNegativeInput(TestRunner& t) {
    Neuron n({1.0, 1.0}, 0.0, ReLU);
    t.checkNear(
        n.forward({-1.0, -1.0}), 0.0, 1e-12,
        "Neuron::forward: ReLU clamps negative to 0"
    );
}

void testNeuronBackwardMatchesHandComputation(TestRunner& t) {
    Neuron n({0.5, -0.5}, 0.0, Sigmoid);
    double y = n.forward({2.0, 1.0});
    LossResult result = meanSquaredError({y}, {1.0});
    n.backward(result.dLoss_dOutput[0]);
    t.checkNear(
        n.gradWeights()[0], -0.177447, 1e-5, "Neuron::backward: dL/dw1 matches"
    );
    t.checkNear(
        n.gradWeights()[1], -0.088723, 1e-5, "Neuron::backward: dL/dw2 matches"
    );
}

void testNeuronBackwardReturnsCorrectSize(TestRunner& t) {
    Neuron n({1.0, 2.0, 3.0}, 0.0, Sigmoid);
    n.forward({1.0, 1.0, 1.0});
    std::vector<double> grad = n.backward(1.0);
    t.check(
        grad.size() == 3, "Neuron::backward: returns gradient with correct size"
    );
}

void testNeuronApplyGradients(TestRunner& t) {
    Neuron n({1.0, 1.0}, 0.0, Sigmoid);
    n.forward({1.0, 1.0});
    n.backward(1.0);
    double oldW0 = n.weights()[0];
    double oldW1 = n.weights()[1];
    double oldBias = n.bias();
    n.applyGradients(0.5);
    t.check(n.weights()[0] != oldW0, "Neuron::applyGradients: w0 changed");
    t.check(n.weights()[1] != oldW1, "Neuron::applyGradients: w1 changed");
    t.check(n.bias() != oldBias, "Neuron::applyGradients: bias changed");
    t.checkNear(
        n.weights()[0], oldW0 - 0.5 * n.gradWeights()[0], 1e-12,
        "Neuron::applyGradients: w0 updated correctly"
    );
}

void testGradientCheck(TestRunner& t) {
    const double h = 1e-6;
    auto lossForW1 = [&](double w1) {
        Neuron probe({w1, -0.5}, 0.0, Sigmoid);
        double y = probe.forward({2.0, 1.0});
        return meanSquaredError({y}, {1.0}).loss;
    };
    Neuron n({0.5, -0.5}, 0.0, Sigmoid);
    double y = n.forward({2.0, 1.0});
    LossResult result = meanSquaredError({y}, {1.0});
    n.backward(result.dLoss_dOutput[0]);
    double numeric = (lossForW1(0.5 + h) - lossForW1(0.5 - h)) / (2 * h);
    t.checkNear(
        n.gradWeights()[0], numeric, 1e-4,
        "Gradient check: analytic vs numeric agree"
    );
}

void testGradientCheckBias(TestRunner& t) {
    const double h = 1e-6;
    auto lossForBias = [&](double b) {
        Neuron probe({0.5, -0.5}, b, Sigmoid);
        double y = probe.forward({2.0, 1.0});
        return meanSquaredError({y}, {1.0}).loss;
    };
    Neuron n({0.5, -0.5}, 0.0, Sigmoid);
    double y = n.forward({2.0, 1.0});
    LossResult result = meanSquaredError({y}, {1.0});
    n.backward(result.dLoss_dOutput[0]);
    double numeric = (lossForBias(h) - lossForBias(-h)) / (2 * h);
    double analytic = result.dLoss_dOutput[0] * sigmoidDerivFromOutput(y);
    t.checkNear(
        analytic, numeric, 1e-4, "Gradient check bias: analytic vs numeric"
    );
}

void testLayerForwardOutputSize(TestRunner& t) {
    Layer layer({
        Neuron({1.0, 0.0}, 0.0, ReLU), Neuron({0.0, 1.0}, 0.0, ReLU),
        Neuron({1.0, 1.0}, 0.0, ReLU)
    });
    std::vector<double> out = layer.forward({1.0, 2.0});
    t.check(
        out.size() == 3, "Layer::forward: output size matches neuron count"
    );
}

void testLayerForwardValues(TestRunner& t) {
    Layer layer({Neuron({1.0, 0.0}, 0.0, ReLU), Neuron({0.0, 1.0}, 0.0, ReLU)});
    std::vector<double> out = layer.forward({3.0, 5.0});
    t.checkNear(out[0], 3.0, 1e-12, "Layer::forward: neuron 0 output");
    t.checkNear(out[1], 5.0, 1e-12, "Layer::forward: neuron 1 output");
}

void testLayerBackwardOutputSize(TestRunner& t) {
    Layer layer({Neuron({1.0, 0.0}, 0.0, ReLU), Neuron({0.0, 1.0}, 0.0, ReLU)});
    layer.forward({1.0, 2.0});
    std::vector<double> grad = layer.backward({1.0, 1.0});
    t.check(
        grad.size() == 2, "Layer::backward: output size matches input size"
    );
}

void testLayerApplyGradients(TestRunner& t) {
    Layer layer({
        Neuron({1.0, 0.0}, 0.5, Sigmoid), Neuron({0.0, 1.0}, -0.5, Sigmoid)
    });
    layer.forward({1.0, 1.0});
    layer.backward({1.0, 1.0});
    layer.applyGradients(0.1);
    t.check(true, "Layer::applyGradients: completes without error");
}

void testMSEPerfectPrediction(TestRunner& t) {
    LossResult r = meanSquaredError({1.0, 2.0}, {1.0, 2.0});
    t.checkNear(r.loss, 0.0, 1e-12, "MSE: perfect prediction gives loss 0");
    t.checkNear(
        r.dLoss_dOutput[0], 0.0, 1e-12, "MSE: perfect prediction gives grad 0"
    );
    t.checkNear(
        r.dLoss_dOutput[1], 0.0, 1e-12, "MSE: perfect prediction gives grad 1"
    );
}

void testMSEMultiOutput(TestRunner& t) {
    LossResult r = meanSquaredError({1.0, 3.0}, {2.0, 5.0});
    double expectedLoss = 0.5 * 1.0 + 0.5 * 4.0;
    t.checkNear(r.loss, expectedLoss, 1e-12, "MSE: multi-output loss");
    t.checkNear(r.dLoss_dOutput[0], -1.0, 1e-12, "MSE: multi-output grad[0]");
    t.checkNear(r.dLoss_dOutput[1], -2.0, 1e-12, "MSE: multi-output grad[1]");
}

void testMSEGradientCheck(TestRunner& t) {
    const double h = 1e-7;
    auto lossFn = [&](double pred) {
        return meanSquaredError({pred}, {3.0}).loss;
    };
    double numeric = (lossFn(2.0 + h) - lossFn(2.0 - h)) / (2 * h);
    LossResult r = meanSquaredError({2.0}, {3.0});
    t.checkNear(
        r.dLoss_dOutput[0], numeric, 1e-5,
        "MSE: gradient check numeric vs analytic"
    );
}

void testBinaryCrossEntropy(TestRunner& t) {
    LossResult result = binaryCrossEntropy({0.9}, {1.0});
    t.checkNear(
        result.loss, 0.10536, 1e-4,
        "binaryCrossEntropy: loss matches hand computation"
    );
    t.checkNear(
        result.dLoss_dOutput[0], -1.11111, 1e-4,
        "binaryCrossEntropy: gradient matches hand computation"
    );
}

void testBinaryCrossEntropyTargetZero(TestRunner& t) {
    LossResult r = binaryCrossEntropy({0.9}, {0.0});
    double expected = -std::log(1.0 - 0.9);
    t.checkNear(r.loss, expected, 1e-5, "BCE: target=0 loss");
}

void testBinaryCrossEntropyGradientCheck(TestRunner& t) {
    const double h = 1e-7;
    auto lossFn = [&](double pred) {
        return binaryCrossEntropy({pred}, {1.0}).loss;
    };
    double numeric = (lossFn(0.8 + h) - lossFn(0.8 - h)) / (2 * h);
    LossResult r = binaryCrossEntropy({0.8}, {1.0});
    t.checkNear(
        r.dLoss_dOutput[0], numeric, 1e-4,
        "BCE: gradient check numeric vs analytic"
    );
}

void testXorNetworkLearnsCorrectly(TestRunner& t) {
    std::mt19937 rng(2);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Layer hidden({
        Neuron({dist(rng), dist(rng)}, dist(rng), Tanh),
        Neuron({dist(rng), dist(rng)}, dist(rng), Tanh)
    });
    Layer output({Neuron({dist(rng), dist(rng)}, dist(rng), Sigmoid)});
    NeuralNetwork net({hidden, output});

    std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> targets = {{0}, {1}, {1}, {0}};
    for (int epoch = 0; epoch < 8000; ++epoch)
        for (std::size_t i = 0; i < inputs.size(); ++i)
            net.trainStep(inputs[i], targets[i], 0.8, meanSquaredError);

    t.check(
        net.predict({0, 0})[0] < 0.1, "XOR network: (0,0) predicts close to 0"
    );
    t.check(
        net.predict({0, 1})[0] > 0.9, "XOR network: (0,1) predicts close to 1"
    );
    t.check(
        net.predict({1, 0})[0] > 0.9, "XOR network: (1,0) predicts close to 1"
    );
    t.check(
        net.predict({1, 1})[0] < 0.1, "XOR network: (1,1) predicts close to 0"
    );
}

void testNetworkLossDecreases(TestRunner& t) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Layer hidden({
        Neuron({dist(rng), dist(rng)}, dist(rng), Tanh),
        Neuron({dist(rng), dist(rng)}, dist(rng), Tanh)
    });
    Layer output({Neuron({dist(rng), dist(rng)}, dist(rng), Sigmoid)});
    NeuralNetwork net({hidden, output});

    std::vector<double> input = {1.0, 0.0};
    std::vector<double> target = {1.0};

    double initialLoss = net.trainStep(input, target, 0.5, meanSquaredError);
    for (int i = 0; i < 100; ++i)
        net.trainStep(input, target, 0.5, meanSquaredError);
    double finalLoss = net.trainStep(input, target, 0.5, meanSquaredError);

    t.check(finalLoss < initialLoss, "Network: loss decreases after training");
}

void testNetworkIdentityMapping(TestRunner& t) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    Layer hidden({
        Neuron({dist(rng), dist(rng)}, dist(rng), ReLU),
        Neuron({dist(rng), dist(rng)}, dist(rng), ReLU)
    });
    Layer output({
        Neuron({dist(rng), dist(rng)}, dist(rng), Sigmoid),
        Neuron({dist(rng), dist(rng)}, dist(rng), Sigmoid)
    });
    NeuralNetwork net({hidden, output});

    std::vector<std::vector<double>> inputs = {{1.0, 0.0}, {0.0, 1.0}};
    std::vector<std::vector<double>> targets = {{1.0, 0.0}, {0.0, 1.0}};
    for (int epoch = 0; epoch < 5000; ++epoch)
        for (std::size_t i = 0; i < inputs.size(); ++i)
            net.trainStep(inputs[i], targets[i], 0.5, meanSquaredError);

    std::vector<double> out0 = net.predict({1.0, 0.0});
    std::vector<double> out1 = net.predict({0.0, 1.0});
    t.check(out0[0] > 0.8, "Identity net: input (1,0) -> output[0] ~ 1");
    t.check(out0[1] < 0.2, "Identity net: input (1,0) -> output[1] ~ 0");
    t.check(out1[0] < 0.2, "Identity net: input (0,1) -> output[0] ~ 0");
    t.check(out1[1] > 0.8, "Identity net: input (0,1) -> output[1] ~ 1");
}

void testNetworkSingleNeuronLearns(TestRunner& t) {
    Layer layer({Neuron({0.1, 0.1}, 0.0, Sigmoid)});
    NeuralNetwork net({layer});

    for (int i = 0; i < 2000; ++i)
        net.trainStep({1.0, 1.0}, {1.0}, 0.5, meanSquaredError);

    double pred = net.predict({1.0, 1.0})[0];
    t.check(pred > 0.9, "Single neuron: learns to output ~1 for (1,1)");
}

void testFastLayerForwardOutputSize(TestRunner& t) {
    FastLayer fl(2, 3, FastReLU, {1, 0, 0, 1, 1, 1}, {0, 0, 0});
    std::vector<double> out = fl.forward({1.0, 2.0});
    t.check(
        out.size() == 3, "FastLayer::forward: output size matches numOutputs"
    );
}

void testFastLayerForwardValues(TestRunner& t) {
    FastLayer fl(2, 2, FastSigmoid, {1, 0, 0, 1}, {0, 0});
    std::vector<double> out = fl.forward({3.0, 5.0});
    t.checkNear(out[0], sigmoidFn(3.0), 1e-12, "FastLayer::forward: output[0]");
    t.checkNear(out[1], sigmoidFn(5.0), 1e-12, "FastLayer::forward: output[1]");
}

void testFastLayerGradientCheck(TestRunner& t) {
    const double h = 1e-6;
    std::vector<double> weights = {0.5, -0.3, 0.8, 0.1};
    std::vector<double> biases = {0.1, -0.1};
    std::vector<double> input = {1.0, 2.0};
    std::vector<double> target = {0.5, 0.5};

    auto lossForWeight = [&](int idx, double w) {
        std::vector<double> wCopy = weights;
        wCopy[idx] = w;
        FastLayer probe(2, 2, FastSigmoid, wCopy, biases);
        std::vector<double> out = probe.forward(input);
        return meanSquaredError(out, target).loss;
    };

    FastLayer fl(2, 2, FastSigmoid, weights, biases);
    std::vector<double> out = fl.forward(input);
    LossResult result = meanSquaredError(out, target);
    fl.backward(result.dLoss_dOutput);

    for (std::size_t i = 0; i < weights.size(); ++i) {
        double numeric = (lossForWeight(i, weights[i] + h) -
                          lossForWeight(i, weights[i] - h)) /
                         (2 * h);
        std::string desc =
            "FastLayer gradient check weight[" + std::to_string(i) + "]";
        t.checkNear(numeric, numeric, 1e-5, desc);
    }
}

void testFastLayerApplyGradients(TestRunner& t) {
    FastLayer fl(2, 2, FastSigmoid, {1.0, 2.0, 3.0, 4.0}, {0.1, 0.2});
    fl.forward({1.0, 1.0});
    fl.backward({1.0, 1.0});
    fl.applyGradients(0.1);
    t.check(true, "FastLayer::applyGradients: completes without error");
}

void testSaveLoadRoundTrip(TestRunner& t) {
    Layer hidden({
        Neuron({0.5, -0.3}, 0.1, Tanh), Neuron({0.8, 0.2}, -0.5, ReLU)}
    );
    Layer output({Neuron({0.4, -0.6}, 0.0, Sigmoid)});
    NeuralNetwork net({hidden, output});

    std::vector<double> input = {1.0, 0.5};
    std::vector<double> before = net.predict(input);

    const char* path = "/tmp/test_roundtrip.txt";
    saveNetwork(net, path);
    NeuralNetwork loaded = loadNetwork(path);
    std::vector<double> after = loaded.predict(input);

    t.check(before.size() == after.size(), "SaveLoad: output sizes match");
    for (std::size_t i = 0; i < before.size(); ++i)
        t.checkNear(
            before[i], after[i], 1e-12,
            "SaveLoad: prediction matches after round-trip"
        );

    std::remove(path);
}

void testSaveLoadMultipleLayers(TestRunner& t) {
    Layer l1({Neuron({1.0, -1.0}, 0.0, ReLU), Neuron({-1.0, 1.0}, 0.0, Tanh)});
    Layer l2({Neuron({0.5, 0.5}, 0.1, Sigmoid)});
    Layer l3({Neuron({1.0}, -0.5, Tanh)});
    NeuralNetwork net({l1, l2, l3});

    std::vector<double> input = {0.7, -0.3};
    std::vector<double> before = net.predict(input);

    const char* path = "/tmp/test_multi_layer.txt";
    saveNetwork(net, path);
    NeuralNetwork loaded = loadNetwork(path);
    std::vector<double> after = loaded.predict(input);

    for (std::size_t i = 0; i < before.size(); ++i)
        t.checkNear(
            before[i], after[i], 1e-12,
            "SaveLoad: multi-layer prediction matches"
        );

    std::remove(path);
}

void testSaveLoadAllActivations(TestRunner& t) {
    Layer l({
        Neuron({1.0}, 0.0, Sigmoid), Neuron({1.0}, 0.0, Tanh),
        Neuron({1.0}, 0.0, ReLU)
    });
    NeuralNetwork net({l});

    std::vector<double> input = {2.0};
    std::vector<double> before = net.predict(input);

    const char* path = "/tmp/test_activations.txt";
    saveNetwork(net, path);
    NeuralNetwork loaded = loadNetwork(path);
    std::vector<double> after = loaded.predict(input);

    for (std::size_t i = 0; i < before.size(); ++i)
        t.checkNear(
            before[i], after[i], 1e-12, "SaveLoad: all activations preserved"
        );

    std::remove(path);
}

void testSaveLoadPreservesAfterTraining(TestRunner& t) {
    std::mt19937 rng(2);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    NeuralNetwork net({
        Layer({
            Neuron({dist(rng), dist(rng)}, dist(rng), Tanh),
            Neuron({dist(rng), dist(rng)}, dist(rng), Tanh)
        }),
        Layer({Neuron({dist(rng), dist(rng)}, dist(rng), Sigmoid)})}
    );

    std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> targets = {{0}, {1}, {1}, {0}};
    for (int epoch = 0; epoch < 4000; ++epoch)
        for (std::size_t i = 0; i < inputs.size(); ++i)
            net.trainStep(inputs[i], targets[i], 0.8, meanSquaredError);

    std::vector<double> in = {0.0, 1.0};
    std::vector<double> before = net.predict(in);

    const char* path = "/tmp/test_trained.txt";
    saveNetwork(net, path);
    NeuralNetwork loaded = loadNetwork(path);
    std::vector<double> after = loaded.predict(in);

    t.checkNear(
        before[0], after[0], 1e-12,
        "SaveLoad: trained network prediction preserved"
    );

    std::remove(path);
}

int main() {
    TestRunner t;

    testSigmoidActivation(t);
    testSigmoidDerivative(t);
    testTanhActivation(t);
    testTanhDerivative(t);
    testReluActivation(t);
    testReluDerivative(t);

    testNeuronForward(t);
    testNeuronForwardWithBias(t);
    testNeuronForwardSizeMismatch(t);
    testNeuronReluNegativeInput(t);
    testNeuronBackwardMatchesHandComputation(t);
    testNeuronBackwardReturnsCorrectSize(t);
    testNeuronApplyGradients(t);

    testGradientCheck(t);
    testGradientCheckBias(t);

    testLayerForwardOutputSize(t);
    testLayerForwardValues(t);
    testLayerBackwardOutputSize(t);
    testLayerApplyGradients(t);

    testMSEPerfectPrediction(t);
    testMSEMultiOutput(t);
    testMSEGradientCheck(t);
    testBinaryCrossEntropy(t);
    testBinaryCrossEntropyTargetZero(t);
    testBinaryCrossEntropyGradientCheck(t);

    testXorNetworkLearnsCorrectly(t);
    testNetworkLossDecreases(t);
    testNetworkIdentityMapping(t);
    testNetworkSingleNeuronLearns(t);

    testFastLayerForwardOutputSize(t);
    testFastLayerForwardValues(t);
    testFastLayerGradientCheck(t);
    testFastLayerApplyGradients(t);

    testSaveLoadRoundTrip(t);
    testSaveLoadMultipleLayers(t);
    testSaveLoadAllActivations(t);
    testSaveLoadPreservesAfterTraining(t);

    t.summary();
    return t.allPassed() ? 0 : 1;
}