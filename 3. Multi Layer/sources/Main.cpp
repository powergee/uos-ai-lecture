#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include "MLP.h"
#define TRUE 1
#define FALSE 0
#define LEARNING_RATE 0.05
#define ITER_MAX 1000000

using DataPair = std::pair<std::vector<double>, std::vector<double>>;
using Dataset = std::vector<DataPair>;

// Train specific logic gate with given dataset.
void trainLogicGate(std::string name, Dataset data, std::vector<int> nodesPerLayer);

int main() {
    srand(time(NULL));

    // Dataset to train AND gate.
    const Dataset andData = {
        { { FALSE, FALSE }, { FALSE } },
        { { FALSE, TRUE  }, { FALSE } },
        { { TRUE , FALSE }, { FALSE } },
        { { TRUE , TRUE  }, { TRUE  } }
    };
    // Dataset to train OR gate.
    const Dataset orData = {
        { { FALSE, FALSE }, { FALSE } },
        { { FALSE, TRUE  }, { TRUE  } },
        { { TRUE , FALSE }, { TRUE  } },
        { { TRUE , TRUE  }, { TRUE  } }
    };
    // Dataset to train XOR gate.
    const Dataset xorData = {
        { { FALSE, FALSE }, { FALSE } },
        { { FALSE, TRUE  }, { TRUE  } },
        { { TRUE , FALSE }, { TRUE  } },
        { { TRUE , TRUE  }, { FALSE } }
    };

    const Dataset donutData = {
        { { 0.0, 0.0 }, { 0 } },
        { { 0.0, 1.0 }, { 0 } },
        { { 1.0, 0.0 }, { 0 } },
        { { 1.0, 1.0 }, { 0 } },
        { { 0.5, 1.0 }, { 0 } },
        { { 1.0, 0.5 }, { 0 } },
        { { 0.0, 0.5 }, { 0 } },
        { { 0.5, 0.0 }, { 0 } },
        { { 0.5, 0.5 }, { 1 } }
    };

    trainLogicGate("AND", andData, { 2, 2, 1 });
    trainLogicGate("OR", orData, { 2, 2, 1 });
    trainLogicGate("XOR", xorData, { 2, 2, 1 });
    trainLogicGate("Donut", donutData, { 2, 3, 1 });
    
    return 0;
}

void trainLogicGate(std::string name, Dataset data, std::vector<int> nodesPerLayer) {
    std::cout << "Training " << name << " Gate...\n";

    // Use a MLP with given count of nodes(including input layer).
    MLP gate(int(nodesPerLayer.size()), &nodesPerLayer[0]);

    // Use MSE for backpropagation.
    MeanSquareError mse;
    // Current count of iterations. (0 <= iterCount <= ITER_MAX-1)
    int iterCount = 0;
    int correctCount = 0;
    // Total errors at each epoch.
    std::vector<double> errorHistory;

    // Loop until all testcases are correct or the goal is not satisfied after ITER_MAX iterations.
    do {
        double errorSum = 0;
        std::vector<bool> refined;
        correctCount = 0;

        // Do training for every records of given dataset.
        for (DataPair pair : data) {
            auto input = pair.first;
            auto label = pair.second;

            // Get output and calculate MSE.
            auto output = gate.getOutput(input);
            refined.push_back(output[0] > 0.5);
            correctCount += (refined.back() == (pair.second[0] == TRUE) ? 1 : 0);

            double error = mse.getTotalError(label, output);
            errorSum += error;

            // Calculate initial gradients by MSE and do backpropagation.
            auto initGradients = mse.getGradient(label, output);
            gate.doBackpropagation(initGradients, LEARNING_RATE);
        }
        errorHistory.push_back(errorSum / data.size());
    } while (++iterCount < ITER_MAX && correctCount < data.size());

    if (correctCount == data.size()) {
        std::cout << "    Successfully trained " << name << " Gate! (after " << errorHistory.size() << " iterations)\n";
    } else {
        std::cout << "    Failed to train " << name << " Gate... (after " << errorHistory.size() << " iterations)\n";
    }

    std::string dirPath = "./results/" + name;
    gate.saveWeights(dirPath);
    std::cout << "    Saved weight matrices. (" << dirPath << ")\n";
}