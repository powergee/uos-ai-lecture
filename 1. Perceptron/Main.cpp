#include <iostream>
#include <string>
#include "Perceptron.h"

const int ROW_COUNT = 4;
const int COL_COUNT = 3;
using TruthTable = int[ROW_COUNT][COL_COUNT];

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

void printTable(TruthTable table, int tryCount) {
    const int COL1_WIDTH = 8;
    const int COL2_WIDTH = 8;
    const int COL3_WIDTH = 10;
    const int ROW_WIDTH = (COL1_WIDTH + COL2_WIDTH + COL3_WIDTH + COL_COUNT-1);

    std::cout << formatCenter(ROW_WIDTH, "< " + getOrdinal(tryCount) + " Truth Table >") << "\n";
    std::cout << formatCenter(COL1_WIDTH, "x1") << "|";
    std::cout << formatCenter(COL2_WIDTH, "x2") << "|";
    std::cout << formatCenter(COL3_WIDTH, "Output") << std::endl;
    std::cout << std::string(ROW_WIDTH, '-') << std::endl;

    for (int i = 0; i < ROW_COUNT; ++i) {
        std::cout << formatCenter(COL1_WIDTH, std::to_string(table[i][0])) << "|";
        std::cout << formatCenter(COL2_WIDTH, std::to_string(table[i][1])) << "|";
        std::cout << formatCenter(COL3_WIDTH, std::to_string(table[i][2])) << std::endl;
    }
}

bool checkOutputs(Perceptron& andGate, int& tryCount) {
    TruthTable table;
    int answerCount = 0;

    for (int x1 = 0; x1 <= 1; ++x1) {
        for (int x2 = 0; x2 <= 1; ++x2) {
            bool result = andGate.getOutput(x1, x2);
            answerCount += (result == (x1 && x2) ? 1 : 0);

            int rowIdx = 2*x1+x2;
            table[rowIdx][0] = x1;
            table[rowIdx][1] = x2;
            table[rowIdx][2] = result;
        }
    }

    printTable(table, ++tryCount);
    std::cout << "Count of Correct Outputs = " << answerCount << std::endl;
    return (answerCount == ROW_COUNT);
}

int main() {
    srand(time(nullptr));

    int tryCount = 0;
    Perceptron andGate;
    std::cout << "Generated perceptron with random weights!" << std::endl << std::endl;

    while (!checkOutputs(andGate, tryCount)) {
        std::cout << "\nSome outputs are wrong." << std::endl;
        std::cout << "You MUST update weights." << std::endl << std::endl;

        double bias, w1, w2;
        std::cout << "Update w0(bias): ";
        std::cin >> bias;
        std::cout << "Update w1: ";
        std::cin >> w1;
        std::cout << "Update w2: ";
        std::cin >> w2;

        andGate.updateWeights(Weight(bias, w1, w2));
        std::cout << std::endl;
    }
    std::cout << std::endl << "Finally all outputs are CORRECT!" << std::endl;
    
    return 0;
}