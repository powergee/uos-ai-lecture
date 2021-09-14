#include "Perceptron.h"
#include <cstdlib>
#include <iostream>

// Generate random real number between min and max. (inclusive)
double Perceptron::generateRandomly(double min, double max) {
    double range = max-min;
    return min + ((double)rand() / RAND_MAX * range);
}

// Create an AND perceptron which get n-dimensional inputs
// and calculate a boolean output.
Perceptron::Perceptron(int inputDim) {
    // The number of dimensions must greater than 0.
    if (inputDim <= 0) {
        std::cerr << "Perceptron: inputDim must be greater than 0.";
        exit(1);
    }

    // bias and weights are picked randomly between -1 and 1.
    bias = generateRandomly(-1, 1);
    weights.resize(inputDim);
    for (double& w : weights) {
        w = generateRandomly(-1, 1);
    }
}

// Get the number of dimensions of this perceptron.
int Perceptron::getInputSize() {
    return weights.size();
}

// Update bias(w0) value.
void Perceptron::updateBias(double bias) {
    this->bias = bias;
}

// Update all values of weights(w1~wn).
void Perceptron::updateWeights(std::vector<double>& newWeights) {
    // The number of dimensions must greater than 0.
    if (newWeights.size() <= 0) {
        std::cerr << "updateWeights: The size of newWeights must be greater than 0.";
        exit(1);
    }
    weights = newWeights;
}

// Calculate output of AND gate.
// It returns true when w0 + (w1*x1 + w2*x2 + ... + wn*xn) > 0 holds.
// Otherwise, it returns false.
bool Perceptron::getOutput(std::vector<bool>& inputs) {
    // In order to calculate the sum of products, 
    // the size of weight vector and input vector must be equal.
    if (weights.size() != inputs.size()) {
        std::cerr << "getOutput: The size of weight vector and input vector are different.";
        exit(1);
    }

    // sum = w0 + (w1*x1 + w2*x2 + ... + wn*xn)
    double sum = bias;
    for (int i = 0; i < inputs.size(); ++i) {
        sum += weights[i] * (inputs[i] ? 1 : 0);
    }
    // It returns true when w0 + (w1*x1 + w2*x2 + ... + wn*xn) > 0 holds.
    return (sum > 0);
}