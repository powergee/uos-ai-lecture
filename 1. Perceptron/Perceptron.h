#ifndef PERCEPTRON_H
#define PERCEPTRON_H
#include <vector>

/*
    Perceptron class to simulate AND Gate.
*/
class Perceptron {
private:
    // Bias(w0) value of perceptron.
    double bias;

    // Weights(w1~wn) values of perceptron.
    // weights[i] means w{i+1}. (0 <= i < input_dimensions)
    std::vector<double> weights;

    // Generate random real number between min and max. (inclusive)
    double generateRandomly(double min, double max);
    
public:
    // Create an AND perceptron which get n-dimensional inputs
    // and calculate a boolean output.
    Perceptron(int inputDim);

    // Get the number of dimensions of this perceptron.
    int getInputSize();

    // Update bias(w0) value.
    void updateBias(double bias);

    // Update all values of weights(w1~wn).
    void updateWeights(std::vector<double>& newWeights);

    // Calculate output of AND gate.
    // It returns true when w0 + (w1*x1 + w2*x2 + ... + wn*xn) > 0 holds.
    // Otherwise, it returns false.
    bool getOutput(std::vector<bool>& inputs);
};

#endif