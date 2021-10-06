#include "Modules.h"
#include <iostream>

// Generate random real number between min and max. (inclusive)
double ProductLayer::generateRandomly(double min, double max) {
    double range = max-min;
    return min + ((double)rand() / RAND_MAX * range);
}

// Create a layer that accept an {inputDim}-dimensional input vector and
// procduce an {outputDim}-dimensional output vector.
ProductLayer::ProductLayer(int inputDim, int outputDim) {
    // Exception handling.
    if (inputDim <= 0 || outputDim <= 0) {
        std::cerr << "ProductLayer: inputDim and outputDim must be greater than 0.";
        exit(1);
    }

    // Initialize bias vector with {outputDim} elements.
    bias.resize(outputDim);
    for (int i = 0; i < outputDim; ++i) {
        bias[i] = generateRandomly(-1, 1);
    }

    // Initialize weight matrix with {inputDim}*{outputDim} size.
    weights.resize(inputDim, std::vector<double>(outputDim));
    for (int i = 0; i < inputDim; ++i) {
        for (int j = 0; j < outputDim; ++j) {
            weights[i][j] = generateRandomly(-1, 1);
        }
    }

    this->inputDim = inputDim;
    this->outputDim = outputDim;
}

// Get the number of dimensions of an input vector.
int ProductLayer::getInputSize() {
    return inputDim;
}

// Get the number of dimensions of an ouput vector.
int ProductLayer::getOutputSize() {
    return outputDim;
}

// Calculate ouput vector of this layer with a given input vector.
// It is equivalent calculation to matrix multiplication.
std::vector<double> ProductLayer::getOutput(std::vector<double>& inputs) {
    // Exception handling.
    if (inputs.size() != inputDim) {
        std::cerr << "getOutput: The size of weight vector and input vector are different.";
        exit(1);
    }

    // Calculate an output vector by multiplication.
    std::vector<double> output(outputDim, 0);
    for (int j = 0; j < outputDim; ++j) {
        // output[j] = bias[j] + sum { weights[i][j]*inputs[i] }
        output[j] = bias[j];
        for (int i = 0; i < inputDim; ++i) {
            output[j] += weights[i][j] * inputs[i];
        }
    }

    // Make a copy of input vector to use at later backpropagation.
    prevInput = inputs;
    return output;
}

// Accumulate current gradient with the gradient vector from the upper node,
// and update weight matrix with given learning rate.
// The returning object is a gradient vector of this module.
std::vector<double> ProductLayer::doBackpropagation(std::vector<double>& upperGradients, double learningRate) {
    // Exception handling.
    if (upperGradients.size() != outputDim) {
        std::cerr << "doBackpropagation: The size of output dimension and upper gradients vector are different.";
        exit(1);
    }

    // Calculate current gradient vector.
    std::vector<double> currGradients(inputDim, 0);
    for (int i = 0; i < inputDim; ++i) {
        for (int j = 0; j < outputDim; ++j) {
            currGradients[i] += prevInput[i] * upperGradients[j];

            // Update weights and bias.
            weights[i][j] -= (prevInput[i] * upperGradients[j]) * learningRate;
            bias[i] -= upperGradients[j] * learningRate;
        }
    }

    return currGradients;
}

// Create a step function layer that accept an {dimSize}-dimensional input vector.
StepFunction::StepFunction(int dimSize) {
    this->dimSize = dimSize;
}

// Get the number of dimensions of an input vector.
int StepFunction::getInputSize() {
    return dimSize;
}

// Get the number of dimensions of an ouput vector.
int StepFunction::getOutputSize() {
    return dimSize;
}

// For every elements of input vector, apply step function.
std::vector<double> StepFunction::getOutput(std::vector<double>& inputs) {
    // Exception handling.
    if (inputs.size() != dimSize) {
        std::cerr << "getOutput: The size of input vector is invalid.";
        exit(1);
    }

    // Apply step function to every input elements.
    std::vector<double> outputs(dimSize);
    for (int i = 0; i < dimSize; ++i) {
        outputs[i] = inputs[i] > 0 ? 1 : 0;
    }

    return outputs;
}

// Accumulate current gradient with the gradient vector from the upper node.
std::vector<double> StepFunction::doBackpropagation(std::vector<double>& upperGradients, double learningRate) {
    // Exception handling.
    if (upperGradients.size() != dimSize) {
        std::cerr << "doBackpropagation: The size of upper gradients vector is invalid.";
        exit(1);
    }

    // Mathematically, it is difficult to use step function for activation function
    // because differential of a function is 0 on every points except 0.

    // However, when training a single layer perceptron for logic gate,
    // it is good to think f'(x) = 1. (from Video: 4-3 Perceptron training algorithm, 11:49)
    return upperGradients;
}

double MeanSquareError::getTotalError(std::vector<double>& labels, std::vector<double>& outputs) {
    // Exception handling.
    if (labels.size() != outputs.size()) {
        std::cerr << "getErrorValue: The size of label vector and output vector are different.";
        exit(1);
    }

    // Accumulate the square of error and divide by its size.
    double total = 0;
    for (int i = 0; i < labels.size(); ++i) {
        double diff = labels[i] - outputs[i];
        total += diff*diff;
    }

    return total / labels.size();
}

std::vector<double> MeanSquareError::getGradient(std::vector<double>& labels, std::vector<double>& outputs) {
    // Exception handling.
    if (labels.size() != outputs.size()) {
        std::cerr << "getGradient: The size of label vector and output vector are different.";
        exit(1);
    }
    
    // Set elements by partial derivative function of MSE
    std::vector<double> gradients(labels.size());
    for (int i = 0; i < labels.size(); ++i) {
        gradients[i] = -(labels[i] - outputs[i]) * 2.0 / labels.size();
    }

    return gradients;
}