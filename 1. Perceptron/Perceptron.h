#ifndef PERCEPTRON_H
#define PERCEPTRON_H
#include <vector>

/*
    Perceptron class to simulate AND Gate.
*/
class Perceptron {
private:
    double bias;
    std::vector<double> weights;
    double generateRandomly(double min, double max);
    
public:
    Perceptron(int inputDim);
    int getInputSize();
    void updateBias(double bias);
    void updateWeights(std::vector<double>& newWeights);
    bool getOutput(std::vector<bool>& inputs);
};

#endif