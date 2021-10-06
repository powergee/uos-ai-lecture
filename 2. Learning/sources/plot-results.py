'''
    Python script to draw graphs of previous results.

    dependencies: matplotlib, numpy
'''
from matplotlib import lines
import numpy as np
import matplotlib.pyplot as plt


def extract_error_csv(path):
    x = []; y = []
    with open(path) as file:
        lines = file.readlines()
        tokens = [line.split(',') for line in lines]
        x = list(map(lambda pair: int(pair[0]), tokens))
        y = list(map(lambda pair: int(pair[1]), tokens))
    return x, y


def extract_table_csv(path, logic):
    table = []; correct = []
    with open(path) as file:
        lines = file.readlines()
        for line in lines:
            row = list(map(int, line.split(',')))
            table.append(row)
            correct.append(logic(row[0], row[1]) == row[2])
    return table, correct


def plot_gate(ax, index, name, color):
    logical_ops = {
        "AND": (lambda x1, x2: x1 & x2),
        "OR": (lambda x1, x2: x1 | x2),
        "XOR": (lambda x1, x2: x1 ^ x2)
    }

    error_x, error_y = extract_error_csv(f"./results/{name}-error.csv")
    table, correct = extract_table_csv(f"./results/{name}-table.csv", logical_ops[name])
    col_label = ("X1", "X2", name)

    ax[index, 0].plot(error_x, error_y, color=color, label=f"{name} Gate")
    ax[index, 0].legend(loc="lower right")
    ax[index, 0].grid(linestyle="dotted", linewidth=1)

    colors = []
    for corr in correct:
        curr_color = "palegreen" if corr else "lightcoral"
        colors.append([curr_color, curr_color, curr_color])

    ax[index, 1].axis("tight")
    ax[index, 1].axis("off")
    ax[index, 1].table(cellText=table, cellLoc='center', cellColours=colors, colLabels=col_label, loc="center")
    ax[index, 1].set_title(f"Final Truth Table of {name}")


if __name__ == "__main__":
    fig, ax = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(12, 7))
    plot_gate(ax, 0, "AND", "red")
    plot_gate(ax, 1, "OR", "blue")
    plot_gate(ax, 2, "XOR", "green")
    ax[0, 0].set_title("Mean Square Error of Gates")

    plt.setp(ax,
        xticks=np.arange(1, 13, step=1),
        yticks=np.arange(0, 5, step=1),
        xlabel="Epoch",
        ylabel="MSE"
    )
    plt.show()