#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include "Perceptron.h"
#define LEARNING_RATE 0.2

namespace fs = std::filesystem;

using DataPair = std::pair<std::vector<double>, std::vector<double>>;
using Dataset = std::vector<DataPair>;

// Train specific logic gate with given dataset.
// After training, save results in CSV files (./results/*.csv)
void trainLogicGate(std::string name, Dataset data);

// Save error per epoch and final truth table CSV files. 
void saveCSV(Perceptron& gate, std::string name, std::vector<double>& errorHistory);

int main() {
    srand(time(NULL));

    // Dataset to train AND gate.
    const Dataset andData = {
        { { 0, 0 }, { 0 } },
        { { 0, 1 }, { 0 } },
        { { 1, 0 }, { 0 } },
        { { 1, 1 }, { 1 } }
    };
    // Dataset to train OR gate.
    const Dataset orData = {
        { { 0, 0 }, { 0 } },
        { { 0, 1 }, { 1 } },
        { { 1, 0 }, { 1 } },
        { { 1, 1 }, { 1 } }
    };
    // Dataset to train XOR gate.
    const Dataset xorData = {
        { { 0, 0 }, { 0 } },
        { { 0, 1 }, { 1 } },
        { { 1, 0 }, { 1 } },
        { { 1, 1 }, { 0 } }
    };

    trainLogicGate("AND", andData);
    trainLogicGate("OR", orData);
    trainLogicGate("XOR", xorData);
    
    return 0;
}

void trainLogicGate(std::string name, Dataset data) {
    std::cout << "Training " << name << " Gate...\n";

    // Use a perceptron with 2 inputs.
    Perceptron gate(2);
    // Use MSE for backpropagation.
    MeanSquareError mse;
    // Current count of iterations. (0 <= iterCount <= 11)
    int iterCount = 0;
    // Total errors at each epoch.
    std::vector<double> errorHistory;

    // Loop until all testcases are correct or the goal is not satisfied after 12 iterations.
    do {
        double errorSum = 0;
        // Do training for every records of given dataset.
        for (DataPair pair : data) {
            auto input = pair.first;
            auto label = pair.second;

            // Get output and calculate MSE.
            auto output = gate.getOutput(input);
            double error = mse.getTotalError(label, output);
            errorSum += error;

            // Calculate initial gradients by MSE and do backpropagation.
            auto initGradients = mse.getGradient(label, output);
            gate.doBackpropagation(initGradients, LEARNING_RATE);
        }
        std::cout << "    - Epoch #" << iterCount+1 << " MSE: " << errorSum << "\n";
        errorHistory.push_back(errorSum);
    } while (++iterCount < 12 && errorHistory.back() > 0);

    if (errorHistory.back() == 0) {
        std::cout << "    Successfully trained " << name << " Gate!\n";
    } else {
        std::cout << "    Failed to train " << name << " Gate...\n";
    }

    // Save results.
    saveCSV(gate, name, errorHistory);
    std::cout << "    * Saved result files(./results/*.csv).\n" << std::endl;
}

void saveCSV(Perceptron& gate, std::string name, std::vector<double>& errorHistory) {
    // If results directory doesn't exist, create one.
    fs::directory_entry resultDir("./results");
    if (!resultDir.exists()) {
        fs::create_directory(resultDir);
    }

    // 1. Create {name}-error.csv and write all error histories.
    fs::path errorPath("./results/" + name + "-error.csv");
    std::ofstream errorStream(errorPath);
    for (int i = 0; i < errorHistory.size(); ++i) {
        errorStream << i+1 << "," << errorHistory[i] << "\n";
    }
    errorStream.close();

    // 2. Create {name}-table.csv and write final truth table.
    fs::path tablePath("./results/" + name + "-table.csv");
    std::ofstream tableStream(tablePath);
    for (int i = 0; i < 4; ++i) {
        double x1 = (i & 0b01) ? 1 : 0;
        double x2 = (i & 0b10) ? 1 : 0;
        std::vector<double> input = { x1, x2 };
        double output = gate.getOutput(input)[0];
        tableStream << x1 << "," << x2 << "," << output << "\n";
    }
    tableStream.close();
}