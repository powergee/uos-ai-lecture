#include "Perceptron.h"
#include <iostream>

Perceptron::Perceptron(int inputDim) {
    modules.push_back(new ProductLayer(inputDim, 1));
    modules.push_back(new StepFunction(1));
    this->inputDim = inputDim;
}

Perceptron::~Perceptron() {
    for (Module* m : modules) {
        delete m;
    }
}

int Perceptron::getInputSize() {
    return inputDim;
}

int Perceptron::getOutputSize() {
    return 1;
}

std::vector<double> Perceptron::getOutput(std::vector<double>& inputs) {
    if (inputDim != inputs.size()) {
        std::cerr << "getOutput: The size of input vector is invalid.";
    }

    std::vector<double> currVector = inputs;
    for (Module* m : modules) {
        currVector = m->getOutput(currVector);
    }
    return currVector;
}

std::vector<double> Perceptron::doBackpropagation(std::vector<double>& upperGradients, double learningRate) {
    if (upperGradients.size() != 1) {
        std::cerr << "doBackpropagation: The size of upper gradient vector must be 1.";
        exit(1);
    }
    
    std::vector<double> gradients = upperGradients;
    for (int i = int(modules.size())-1; i >= 0; --i) {
        gradients = modules[i]->doBackpropagation(gradients, learningRate);
    }
    return gradients;
}