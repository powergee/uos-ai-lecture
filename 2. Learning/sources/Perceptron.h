#ifndef PERCEPTRON_H
#define PERCEPTRON_H
#include "Modules.h"

class Perceptron : public Module {
private:
    std::vector<Module*> modules;
    int inputDim;

public:
    Perceptron(int inputDim);
    ~Perceptron();
    int getInputSize();
    int getOutputSize();
    std::vector<double> getOutput(std::vector<double>& inputs);
    std::vector<double> doBackpropagation(std::vector<double>& upperGradients, double learningRate);
};

#endif