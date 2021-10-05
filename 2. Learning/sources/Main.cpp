#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>
#include "Perceptron.h"
#define LEARNING_RATE 0.2

using DataPair = std::pair<std::vector<double>, std::vector<double>>;
using Dataset = std::vector<DataPair>;

void trainLogicGate(std::string name, Dataset data);
void printTable(Perceptron& gate, std::string title, std::string indent);
std::string formatCenter(int width, std::string str);

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
    } while (++iterCount < 20 && errorSum > 0);

    if (errorSum == 0) {
        std::cout << "    Successfully trained " << name << " Gate!\n";
    } else {
        std::cout << "    Failed to train " << name << " Gate...\n";
    }

    std::cout << "\n";
    printTable(gate, "< Final " + name + " Gate >", "    ");
    std::cout << "\n" << std::endl;
}

void printTable(Perceptron& gate, std::string title, std::string indent) {
    const int colWidth = 8;
    const int rowWidth = colWidth*3 + 2;

    // Print a title.
    std::cout << indent << formatCenter(rowWidth, title) << "\n";

    // Print header columns
    std::cout << indent << formatCenter(colWidth, "x1") << "|";
    std::cout << formatCenter(colWidth, "x2") << "|";
    std::cout << formatCenter(colWidth, "Output") << "\n";

    // Print a horizontal divider to seperate header and body of a table.
    std::cout << indent << std::string(rowWidth, '-') << "\n";

    // Nested for loops to print all contents of a table.
    for (int i = 0; i < 4; ++i) {
        double x1 = (i & 0b01) ? 1 : 0;
        double x2 = (i & 0b10) ? 1 : 0;
        std::vector<double> input = { x1, x2 };
        double out = gate.getOutput(input)[0];

        std::cout << indent << formatCenter(colWidth, std::to_string(x1 > 0 ? 1 : 0)) << "|";
        std::cout << formatCenter(colWidth, std::to_string(x2 > 0 ? 1 : 0)) << "|";
        std::cout << formatCenter(colWidth, std::to_string(out > 0 ? 1 : 0)) << "\n";
    }
}

std::string formatCenter(int width, std::string str) {
    int leftPadding = (width - (int)str.size()) / 2;
    int rightPadding = width - leftPadding - str.size();

    // Add paddings to align str.
    return std::string(leftPadding, ' ') + str + std::string(rightPadding, ' ');
}