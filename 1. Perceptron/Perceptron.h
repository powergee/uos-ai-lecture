#ifndef PER_H
#define PER_H

/*
    Represent weights of a perceptron.
*/
class Weight {
private:
    double generateRandomly(double min, double max);

public:
    double bias, w1, w2;
    Weight(double bias, double w1, double w2);
    Weight();
};

/*
    Perceptron class to simulate AND Gate.
*/
class Perceptron {
private:
    Weight weights;
    
public:
    Perceptron();
    void updateWeights(Weight newWeights);
    bool getOutput(int x1, int x2);
};

#endif