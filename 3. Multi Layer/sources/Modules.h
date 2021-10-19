#ifndef MODULES_H
#define MODULES_H
#include <vector>
#include <string>

/*
    Abstract class to represent every module that support forward-pass and backward-pass.
*/
class Module {
public:
    virtual ~Module() { };
    
    // Get the number of dimensions of an input vector.
    virtual int getInputSize() = 0;

    // Get the number of dimensions of an ouput vector.
    virtual int getOutputSize() = 0;

    // Calculate ouput vector of this module with a given input vector.
    virtual std::vector<double> getOutput(std::vector<double>& inputs) = 0;

    // Accumulate current gradient with the gradient vector from the upper node.
    // If this module has trainable parameters, they are updated in this procedure.
    // The returning object is a gradient vector of this module.
    virtual std::vector<double> doBackpropagation(std::vector<double>& upperGradients, double learningRate) = 0;

    // Save learnable weights to given path.
    // If this module doesn't have any learnable weights, it returns false.
    virtual bool saveWeights(std::string path) { return false; }
};


/*
    A layer that do matrix multiplication to produce an output vector from an input vector.
    It support multiple input dimensions and multiple output dimensions.

    (But in this assignment, a logic gate produce only one output.)
*/
class FullyConnected : public Module {
private:
    // Bias(w0) value of this layer.
    std::vector<double> bias;

    // Weight matrix that produce an output vector from an input vector.
    // weights[i][j] means a weight value that connect i th input to j th output.
    std::vector<std::vector<double>> weights;
    int inputDim, outputDim;

    // A copy of the input vector in previos output calculation.
    std::vector<double> prevInput;

    // Generate random real number between min and max. (inclusive)
    double generateRandomly(double min, double max);

public:
    // Create a layer that accept an {inputDim}-dimensional input vector and
    // procduce an {outputDim}-dimensional output vector.
    FullyConnected(int inputDim, int outputDim);

    // Get the number of dimensions of an input vector.
    int getInputSize();
    // Get the number of dimensions of an ouput vector.
    int getOutputSize();

    // Calculate ouput vector of this layer with a given input vector.
    // It is equivalent calculation to matrix multiplication.
    std::vector<double> getOutput(std::vector<double>& inputs);

    // Accumulate current gradient with the gradient vector from the upper node,
    // and update weight matrix with given learning rate.
    // The returning object is a gradient vector of this module.
    std::vector<double> doBackpropagation(std::vector<double>& upperGradients, double learningRate);

    // Save learnable weights to given path.
    bool saveWeights(std::string path);
};


/*
    An activation function that calculate sigmoid values.
*/
class Sigmoid : public Module {
private:
    int dimSize;
    // A copy of the input vector in previos output calculation.
    std::vector<double> prevInput;

    double calcSigmoid(double x);

public:
    // Create a step function layer that accept an {dimSize}-dimensional input vector.
    Sigmoid(int dimSize);

    // Get the number of dimensions of an input vector.
    int getInputSize();
    // Get the number of dimensions of an ouput vector.
    int getOutputSize();

    // For every elements of input vector, apply sigmoid function.
    std::vector<double> getOutput(std::vector<double>& inputs);

    // Accumulate current gradient with the gradient vector from the upper node.
    std::vector<double> doBackpropagation(std::vector<double>& upperGradients, double learningRate);
};


/*
    Abstract class to represent every error functions.
*/
class Error {
public:
    virtual ~Error() { };

    // Calculate an error value with actual answers(labels) and ouputs of a trained network.
    virtual double getTotalError(std::vector<double>& labels, std::vector<double>& outputs) = 0;

    // Calculate a gradient vector.
    // It is used for backpropagation of a training network.
    virtual std::vector<double> getGradient(std::vector<double>& labels, std::vector<double>& outputs) = 0;
};


class MeanSquareError : public Error {
public:
    // Calculate an error value by Mean Square Error(MSE).
    double getTotalError(std::vector<double>& labels, std::vector<double>& outputs);

    // Calculate a gradient vector, which is a differentiation of MSE function.
    std::vector<double> getGradient(std::vector<double>& labels, std::vector<double>& outputs);
};

#endif