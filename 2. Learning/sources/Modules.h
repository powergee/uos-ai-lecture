#ifndef MODULES_H
#define MODULES_H
#include <vector>

class Module {
public:
    virtual ~Module() { };
    virtual int getInputSize() = 0;
    virtual int getOutputSize() = 0;
    virtual std::vector<double> getOutput(std::vector<double>& inputs) = 0;
    virtual std::vector<double> doBackpropagation(std::vector<double>& upperGradients, double learningRate) = 0;
};

class ProductLayer : public Module {
private:
    std::vector<double> bias;
    std::vector<std::vector<double>> weights;
    int inputDim, outputDim;
    std::vector<double> prevInput;

    double generateRandomly(double min, double max);

public:
    ProductLayer(int inputDim, int outputDim);
    int getInputSize();
    int getOutputSize();
    std::vector<double> getOutput(std::vector<double>& inputs);
    std::vector<double> doBackpropagation(std::vector<double>& upperGradients, double learningRate);
};

class StepFunction : public Module {
private:
    int dimSize;

public:
    StepFunction(int dimSize);
    int getInputSize();
    int getOutputSize();
    std::vector<double> getOutput(std::vector<double>& inputs);
    std::vector<double> doBackpropagation(std::vector<double>& upperGradients, double learningRate);
};

class Error {
public:
    virtual ~Error() { };
    virtual double getTotalError(std::vector<double>& labels, std::vector<double>& outputs) = 0;
    virtual std::vector<double> getGradient(std::vector<double>& labels, std::vector<double>& outputs) = 0;
};

class MeanSquareError : public Error {
public:
    double getTotalError(std::vector<double>& labels, std::vector<double>& outputs);
    std::vector<double> getGradient(std::vector<double>& labels, std::vector<double>& outputs);
};

#endif