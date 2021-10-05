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

void trainLogicGate(std::string name, Dataset data);
void saveCSV(Perceptron& gate, std::string name, std::vector<double>& errorHistory);

int main() {
    srand(time(NULL));

    const Dataset andData = {
        { { 0, 0 }, { 0 } },
        { { 0, 1 }, { 0 } },
        { { 1, 0 }, { 0 } },
        { { 1, 1 }, { 1 } }
    };
    const Dataset orData = {
        { { 0, 0 }, { 0 } },
        { { 0, 1 }, { 1 } },
        { { 1, 0 }, { 1 } },
        { { 1, 1 }, { 1 } }
    };
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
    std::cout << "Trainning " << name << " Gate...\n";

    Perceptron gate(2);
    MeanSquareError mse;
    int iterCount = 0;
    double errorSum;
    std::vector<double> errorHistory;

    do {
        errorSum = 0;
        for (DataPair pair : data) {
            auto input = pair.first;
            auto label = pair.second;

            auto output = gate.getOutput(input);
            double error = mse.getTotalError(label, output);
            errorSum += error;

            auto initGradients = mse.getGradient(label, output);
            gate.doBackpropagation(initGradients, LEARNING_RATE);
        }
        std::cout << "    - Epoch #" << iterCount+1 << " MSE: " << errorSum << "\n";
        errorHistory.push_back(errorSum);
    } while (++iterCount < 12 && errorSum > 0);

    if (errorSum == 0) {
        std::cout << "    Successfully trained " << name << " Gate!\n";
    } else {
        std::cout << "    Failed to train " << name << " Gate...\n";
    }

    saveCSV(gate, name, errorHistory);
    std::cout << "    * Saved result files(./results/*.csv).\n" << std::endl;
}

void saveCSV(Perceptron& gate, std::string name, std::vector<double>& errorHistory) {
    fs::directory_entry resultDir("./results");
    if (!resultDir.exists()) {
        fs::create_directory(resultDir);
    }

    fs::path errorPath("./results/" + name + "-error.csv");
    std::ofstream errorStream(errorPath);
    for (int i = 0; i < errorHistory.size(); ++i) {
        errorStream << i+1 << "," << errorHistory[i] << "\n";
    }
    errorStream.close();

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