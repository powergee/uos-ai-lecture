#include <iostream>
#include <string>
#include "Perceptron.h"

std::string formatCenter(int width, std::string str) {
    int leftPadding = (width - (int)str.size()) / 2;
    int rightPadding = width - leftPadding - str.size();

    return std::string(leftPadding, ' ') + str + std::string(rightPadding, ' ');
}

std::string getOrdinal(int number) {
    std::string baseStr = std::to_string(number);
    switch (baseStr.back())
    {
    case '1':
        return baseStr + "st";
    case '2':
        return baseStr + "nd";
    case '3':
        return baseStr + "rd";
    default:
        return baseStr + "th";
    }
}

void printTable(std::vector<std::vector<bool>> table, int tryCount) {
    int colWidth = 8;
    int colCount = table[0].size();
    int rowWidth = colWidth*colCount + colCount-1;
    int rowCount = table.size();

    std::cout << formatCenter(rowWidth, "< " + getOrdinal(tryCount) + " Truth Table >") << std::endl;
    for (int i = 1; i <= colCount-1; ++i) {
        std::cout << formatCenter(colWidth, "x" + std::to_string(i)) << "|";
    }
    std::cout << formatCenter(colWidth, "Output") << std::endl;
    std::cout << std::string(rowWidth, '-') << std::endl;

    for (int i = 0; i < rowCount; ++i) {
        for (int j = 0; j < colCount; ++j) {
            std::cout << formatCenter(colWidth, std::to_string(table[i][j]));
            std::cout << (j == colCount-1 ? "\n" : "|");
        }
    }
}

bool checkOutputs(Perceptron& andGate, int& tryCount) {
    int inputDim = andGate.getInputSize();
    int rowCount = (1 << inputDim);
    std::vector<std::vector<bool>> table(rowCount);
    int correctCount = 0;

    for (int r = 0; r < rowCount; ++r) {
        bool answer = true;
        for (int xIdx = 0; xIdx < inputDim; ++xIdx) {
            bool xVal = (r & (1<<xIdx)) > 0;
            table[r].push_back(xVal);
            answer = answer && xVal;
        }

        bool output = andGate.getOutput(table[r]);
        table[r].push_back(output);
        correctCount += (answer == output ? 1 : 0);
    }

    printTable(table, ++tryCount);
    std::cout << "Count of Correct Outputs = " << correctCount << std::endl;
    return (correctCount == rowCount);
}

int main() {
    srand(time(nullptr));

    int inputDim;
    std::cout << "Enter input dimensions of AND Gate: ";
    std::cin >> inputDim;

    int tryCount = 0;
    Perceptron andGate(inputDim);
    std::cout << "Generated perceptron with random bias and weights!" << std::endl << std::endl;

    while (!checkOutputs(andGate, tryCount)) {
        std::cout << "\nSome outputs are wrong." << std::endl;
        std::cout << "You MUST update bias and weights." << std::endl << std::endl;

        double bias;
        std::vector<double> newWeights(inputDim);
        std::cout << "Update bias(w0): ";
        std::cin >> bias;

        for (int i = 0; i < inputDim; ++i) {
            std::cout << "Update w" << i+1 << ": ";
            std::cin >> newWeights[i];
        }

        andGate.updateBias(bias);
        andGate.updateWeights(newWeights);
        std::cout << std::endl;
    }
    std::cout << std::endl << "Finally all outputs are CORRECT!" << std::endl;
    
    return 0;
}