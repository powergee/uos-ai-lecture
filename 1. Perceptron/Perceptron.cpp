#include "Perceptron.h"
#include <cstdlib>
#include <iostream>

double Perceptron::generateRandomly(double min, double max) {
    double range = max-min;
    return min + ((double)rand() / RAND_MAX * range);
}

Perceptron::Perceptron(int inputDim) {
    if (inputDim <= 0) {
        std::cerr << "Perceptron: inputDim must be greater than 0.";
        exit(1);
    }
    bias = generateRandomly(-1, 1);
    weights.resize(inputDim);
    for (double& w : weights) {
        w = generateRandomly(-1, 1);
    }
}

int Perceptron::getInputSize() {
    return weights.size();
}

void Perceptron::updateBias(double bias) {
    this->bias = bias;
}

void Perceptron::updateWeights(std::vector<double>& newWeights) {
    if (newWeights.size() <= 0) {
        std::cerr << "updateWeights: The size of newWeights must be greater than 0.";
        exit(1);
    }
    weights = newWeights;
}

bool Perceptron::getOutput(std::vector<bool>& inputs) {
    if (weights.size() != inputs.size()) {
        std::cerr << "getOutput: The size of weight vector and input vector are different.";
        exit(1);
    }
    double sum = bias;
    for (int i = 0; i < inputs.size(); ++i) {
        sum += weights[i] * (inputs[i] ? 1 : 0);
    }
    return (sum > 0);
}