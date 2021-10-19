'''
    Python script to draw graphs of previous results.

    dependencies: matplotlib, numpy
'''
import numpy as np
import matplotlib.pyplot as plt


def extract_error_csv(path):
    x = []; y = []
    with open(path) as file:
        lines = file.readlines()
        tokens = [line.split(',') for line in lines]
        x = list(map(lambda pair: float(pair[0]), tokens))
        y = list(map(lambda pair: float(pair[1]), tokens))
    return x, y


def plot_gate(cell, name, color):
    error_x, error_y = extract_error_csv(f"./results/{name}/error.csv")
    cell.plot(error_x, error_y, color=color, label=f"MSE of {name} Gate")
    cell.legend(loc="upper right")
    cell.grid(linestyle="dotted", linewidth=1)


if __name__ == "__main__":
    fig, ax = plt.subplots(2, 2)
    plot_gate(ax[0, 0], "AND", "red")
    plot_gate(ax[0, 1], "OR", "blue")
    plot_gate(ax[1, 0], "XOR", "green")
    plot_gate(ax[1, 1], "Donut", "purple")

    plt.show()