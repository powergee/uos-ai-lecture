#include "Modules.h"
#include <iostream>

double ProductLayer::generateRandomly(double min, double max) {
    double range = max-min;
    return min + ((double)rand() / RAND_MAX * range);
}

ProductLayer::ProductLayer(int inputDim, int outputDim) {
    if (inputDim <= 0 || outputDim <= 0) {
        std::cerr << "ProductLayer: inputDim and outputDim must be greater than 0.";
        exit(1);
    }

    bias.resize(outputDim);
    for (int i = 0; i < outputDim; ++i) {
        bias[i] = generateRandomly(-1, 1);
    }

    weights.resize(inputDim, std::vector<double>(outputDim));
    for (int i = 0; i < inputDim; ++i) {
        for (int j = 0; j < outputDim; ++j) {
            weights[i][j] = generateRandomly(-1, 1);
        }
    }

    this->inputDim = inputDim;
    this->outputDim = outputDim;
}

int ProductLayer::getInputSize() {
    return inputDim;
}

int ProductLayer::getOutputSize() {
    return outputDim;
}

std::vector<double> ProductLayer::getOutput(std::vector<double>& inputs) {
    if (inputs.size() != inputDim) {
        std::cerr << "getOutput: The size of weight vector and input vector are different.";
        exit(1);
    }

    std::vector<double> output(outputDim, 0);
    for (int j = 0; j < outputDim; ++j) {
        output[j] = bias[j];
        for (int i = 0; i < inputDim; ++i) {
            output[j] += weights[i][j] * inputs[i];
        }
    }

    prevInput = inputs;
    return output;
}

std::vector<double> ProductLayer::doBackpropagation(std::vector<double>& upperGradients, double learningRate) {
    if (upperGradients.size() != outputDim) {
        std::cerr << "doBackpropagation: The size of output dimension and upper gradients vector are different.";
        exit(1);
    }

    std::vector<double> currGradients(inputDim, 0);
    for (int i = 0; i < inputDim; ++i) {
        for (int j = 0; j < outputDim; ++j) {
            currGradients[i] += prevInput[i] * upperGradients[j];
            weights[i][j] -= prevInput[i] * upperGradients[j];
            bias[i] -= upperGradients[j];
        }
    }

    return currGradients;
}

StepFunction::StepFunction(int dimSize) {
    this->dimSize = dimSize;
}

int StepFunction::getInputSize() {
    return dimSize;
}

int StepFunction::getOutputSize() {
    return dimSize;
}

std::vector<double> StepFunction::getOutput(std::vector<double>& inputs) {
    if (inputs.size() != dimSize) {
        std::cerr << "getOutput: The size of input vector is invalid.";
        exit(1);
    }

    std::vector<double> outputs(dimSize);
    for (int i = 0; i < dimSize; ++i) {
        outputs[i] = inputs[i] >= 0 ? 1 : 0;
    }

    return outputs;
}

std::vector<double> StepFunction::doBackpropagation(std::vector<double>& upperGradients, double learningRate) {
    if (upperGradients.size() != dimSize) {
        std::cerr << "doBackpropagation: The size of upper gradients vector is invalid.";
        exit(1);
    }

    return upperGradients;
}

double MeanSquareError::getTotalError(std::vector<double>& labels, std::vector<double>& outputs) {
    if (labels.size() != outputs.size()) {
        std::cerr << "getErrorValue: The size of label vector and output vector are different.";
        exit(1);
    }

    double total = 0;
    for (int i = 0; i < labels.size(); ++i) {
        double diff = labels[i] - outputs[i];
        total += diff*diff;
    }

    return total / labels.size();
}

std::vector<double> MeanSquareError::getGradient(std::vector<double>& labels, std::vector<double>& outputs) {
    if (labels.size() != outputs.size()) {
        std::cerr << "getGradient: The size of label vector and output vector are different.";
        exit(1);
    }
    
    std::vector<double> gradients(labels.size());
    for (int i = 0; i < labels.size(); ++i) {
        gradients[i] = -(labels[i] - outputs[i]);
    }

    return gradients;
}