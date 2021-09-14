#include <iostream>
#include <string>
#include "Perceptron.h"

using TruthTable = std::vector<std::vector<bool>>;

// Check all outputs of AND perceptron and increase attemptsCount by 1.
// It returns true if and only if andGate give all valid outputs.
bool checkOutputs(Perceptron& andGate, int& attemptsCount);

// Print truth table on terminal and print attempsCount.
void printTable(TruthTable& table, int attemptsCount);

// Convert a number to ordinal number. (ex. 1 -> "1st", 2 -> "2nd")
std::string getOrdinal(int number);

// Add blanks to left and right of str to align str to center.
std::string formatCenter(int width, std::string str);

int main() {
    srand(time(nullptr));

    // Create 1-layer perceptron with n-dimentional inputs.
    // Bias and weights are initialized with random values.
    int inputDim;
    std::cout << "Enter input dimensions of AND Gate: ";
    std::cin >> inputDim;

    int attemptsCount = 0;
    Perceptron andGate(inputDim);
    std::cout << "Generated perceptron with random bias and weights!" << std::endl << std::endl;

    // Loop forever until AND perceptron give all valid outputs.
    while (!checkOutputs(andGate, attemptsCount)) {
        std::cout << "\nSome outputs are wrong." << std::endl;
        std::cout << "You MUST update bias and weights." << std::endl << std::endl;

        double bias;
        std::vector<double> newWeights(inputDim);

        // Get a new bias value from the user.
        std::cout << "Update bias(w0): ";
        std::cin >> bias;

        // Get new weight values from the user.
        for (int i = 0; i < inputDim; ++i) {
            std::cout << "Update w" << i+1 << ": ";
            std::cin >> newWeights[i];
        }

        // Update bias and weights to new values.
        andGate.updateBias(bias);
        andGate.updateWeights(newWeights);

        // Flush standard output stream and print a new line.
        std::cout << std::endl;
    }
    // If AND perceptron gives all valid outputs, loop is breaked.

    std::cout << std::endl << "Finally all outputs are CORRECT!" << std::endl;
    
    return 0;
}

bool checkOutputs(Perceptron& andGate, int& attemptsCount) {
    const int inputDim = andGate.getInputSize();
    // If the number of input dimensions are n,
    // the number of all possible input combinations are 2^n.
    const int rowCount = (1 << inputDim);

    // Truth table of a perceptron.
    // It is a matrix to store inputs and outputs.
    TruthTable table(rowCount);
    int correctCount = 0;

    for (int r = 0; r < rowCount; ++r) {
        // Real answer of AND gate.
        // answer = x1 and x2 and ... and xn
        bool answer = true;

        // For loop to iterate x1~xn.
        for (int xIdx = 0; xIdx < inputDim; ++xIdx) {
            bool xVal = (r & (1<<xIdx)) > 0;
            table[r].push_back(xVal);
            answer = answer && xVal;
        }

        // Get an output of a perceptron.
        bool output = andGate.getOutput(table[r]);
        table[r].push_back(output);

        // If the output of a perceptron is equal to real answer of AND gate,
        // increase correctCount by 1.
        correctCount += (answer == output ? 1 : 0);
    }

    // Print truth table on terminal.
    printTable(table, ++attemptsCount);
    std::cout << "Count of Correct Outputs = " << correctCount << std::endl;

    return (correctCount == rowCount);
}

void printTable(TruthTable& table, int attemptsCount) {
    const int colWidth = 8;
    const int colCount = table[0].size();
    const int rowWidth = colWidth*colCount + colCount-1;
    const int rowCount = table.size();

    // Print a title.
    std::cout << formatCenter(rowWidth, "< " + getOrdinal(attemptsCount) + " Truth Table >") << std::endl;
    
    // Print header columns (x1, x2, ..., xn, Output)
    for (int i = 1; i <= colCount-1; ++i) {
        std::cout << formatCenter(colWidth, "x" + std::to_string(i)) << "|";
    }
    std::cout << formatCenter(colWidth, "Output") << std::endl;

    // Print a horizontal divider to seperate header and body of a table.
    std::cout << std::string(rowWidth, '-') << std::endl;

    // Nested for loops to print all contents of a table.
    for (int i = 0; i < rowCount; ++i) {
        for (int j = 0; j < colCount; ++j) {
            std::cout << formatCenter(colWidth, std::to_string(table[i][j]));
            std::cout << (j == colCount-1 ? "\n" : "|");
        }
    }
}

std::string formatCenter(int width, std::string str) {
    int leftPadding = (width - (int)str.size()) / 2;
    int rightPadding = width - leftPadding - str.size();

    // Add paddings to align str.
    return std::string(leftPadding, ' ') + str + std::string(rightPadding, ' ');
}

std::string getOrdinal(int number) {
    std::string baseStr = std::to_string(number);
    switch (baseStr.back())
    {
    case '1': return baseStr + "st";
    case '2': return baseStr + "nd";
    case '3': return baseStr + "rd";
    default:  return baseStr + "th";
    }
}