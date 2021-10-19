#include "Modules.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <cmath>

namespace fs = std::filesystem;

// Generate random real number between min and max. (inclusive)
double FullyConnected::generateRandomly(double min, double max) {
    double range = max-min;
    return min + ((double)rand() / RAND_MAX * range);
}

// Create a layer that accept an {inputDim}-dimensional input vector and
// procduce an {outputDim}-dimensional output vector.
FullyConnected::FullyConnected(int inputDim, int outputDim) {
    // Exception handling.
    if (inputDim <= 0 || outputDim <= 0) {
        std::cerr << "FullyConnected: inputDim and outputDim must be greater than 0.";
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
int FullyConnected::getInputSize() {
    return inputDim;
}

// Get the number of dimensions of an ouput vector.
int FullyConnected::getOutputSize() {
    return outputDim;
}

// Calculate ouput vector of this layer with a given input vector.
// It is equivalent calculation to matrix multiplication.
std::vector<double> FullyConnected::getOutput(std::vector<double>& inputs) {
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
std::vector<double> FullyConnected::doBackpropagation(std::vector<double>& upperGradients, double learningRate) {
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
            // Update weights.
            weights[i][j] -= (prevInput[i] * upperGradients[j]) * learningRate;
        }
    }
    for (int j = 0; j < outputDim; ++j) {
        // Update bias.
        bias[j] -= upperGradients[j] * learningRate;
    }

    return currGradients;
}

bool FullyConnected::saveWeights(std::string pathStr) {
    fs::path path = pathStr;
    fs::create_directories(path.parent_path());

    std::ofstream wOut(path.string() + ".csv", std::ios::out);
    for (auto& row : weights) {
        for (int i = 0; i < row.size(); ++i) {
            wOut << row[i];
            if (i != int(row.size())-1) {
                wOut << ",";
            }
        }
        wOut << "\n";
    }
    wOut.close();

    std::ofstream bOut(path.string() + "-bias.csv", std::ios::out);
    for (auto& b : bias) {
        bOut << b << "\n";
    }
    bOut.close();
    return true;
}

double Sigmoid::calcSigmoid(double x) {
    double e = exp(x);
    return e / (e+1);
}

// Create a step function layer that accept an {dimSize}-dimensional input vector.
Sigmoid::Sigmoid(int dimSize) {
    this->dimSize = dimSize;
}

// Get the number of dimensions of an input vector.
int Sigmoid::getInputSize() {
    return dimSize;
}

// Get the number of dimensions of an ouput vector.
int Sigmoid::getOutputSize() {
    return dimSize;
}

// For every elements of input vector, apply sigmoid function.
std::vector<double> Sigmoid::getOutput(std::vector<double>& inputs) {
    // Exception handling.
    if (inputs.size() != dimSize) {
        std::cerr << "getOutput: The size of input vector is invalid.";
        exit(1);
    }

    // Apply sigmoid function to every input elements.
    std::vector<double> outputs(dimSize);
    for (int i = 0; i < dimSize; ++i) {
        outputs[i] = calcSigmoid(inputs[i]);
    }

    // Make a copy of input vector to use at later backpropagation.
    prevInput = inputs;
    return outputs;
}

// Accumulate current gradient with the gradient vector from the upper node.
std::vector<double> Sigmoid::doBackpropagation(std::vector<double>& upperGradients, double learningRate) {
    // Exception handling.
    if (upperGradients.size() != dimSize) {
        std::cerr << "doBackpropagation: The size of upper gradients vector is invalid.";
        exit(1);
    }

    // Calculate current gradient vector.
    std::vector<double> currGradients(dimSize, 0);
    for (int i = 0; i < dimSize; ++i) {
        double sig = calcSigmoid(prevInput[i]);
        double grad = sig * (1-sig);
        currGradients[i] = grad * upperGradients[i];
    }

    return currGradients;
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