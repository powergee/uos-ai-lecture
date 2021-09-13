#include "Perceptron.h"
#include <cstdlib>

double Weight::generateRandomly(double min, double max) {
    double range = max-min;
    return min + ((double)rand() / RAND_MAX * range);
}

Weight::Weight(double bias, double w1, double w2) {
    this->bias = bias;
    this->w1 = w1;
    this->w2 = w2;
}

Weight::Weight() {
    bias = generateRandomly(-1, 1);
    w1 = generateRandomly(-1, 1);
    w2 = generateRandomly(-1, 1);
}

Perceptron::Perceptron() : weights() {}

void Perceptron::updateWeights(Weight newWeights) {
    weights = newWeights;
}

bool Perceptron::getOutput(int x1, int x2) {
    double sum = weights.bias;
    sum += x1 * weights.w1;
    sum += x2 * weights.w2;
    return (sum > 0);
}