'''
    Python script to draw graphs of previous results.

    dependencies: matplotlib, numpy
'''
import numpy as np
import matplotlib.pyplot as plt


def extract_weight_csv(path):
    with open(path) as file:
        lines = file.readlines()
        tokens = [list(map(float, line.split(','))) for line in lines]
        return tokens


def draw_line(cell, m, y0, color):
    pivots = np.linspace(-0.5, 1.5, 100)
    xs = []; ys = []
    for x in pivots:
        y = m*x + y0
        if -0.5 <= x <= 1.5 and -0.5 <= y <= 1.5:
            xs.append(x)
            ys.append(y)
    cell.plot(xs, ys, color=color, linestyle="dashed")


def plot_gate(cell, name, color, dataset):
    w = extract_weight_csv(f"./results/{name}/layer-1.csv")
    b = extract_weight_csv(f"./results/{name}/layer-1-bias.csv")
    for i in range(len(w[0])):
        m = -w[0][i]/w[1][i]
        y0 = -b[i][0]/w[1][i]
        print(f"{name} -> y = {m} x + ({y0})")
        draw_line(cell, m, y0, color)

    pos_x = []; pos_y = []
    neg_x = []; neg_y = []
    for pair in dataset:
        if pair[1] == 1:
            pos_x.append(pair[0][0])
            pos_y.append(pair[0][1])
        else:
            neg_x.append(pair[0][0])
            neg_y.append(pair[0][1])
    cell.scatter(pos_x, pos_y, s=80, marker="o")
    cell.scatter(neg_x, neg_y, s=80, marker="X")
    cell.grid(linestyle="dotted", linewidth=1)


if __name__ == "__main__":
    and_data = [
        [[0, 0], 0],
        [[0, 1], 0],
        [[1, 0], 0],
        [[1, 1], 1],
    ]
    or_data = [
        [[0, 0], 0],
        [[0, 1], 1],
        [[1, 0], 1],
        [[1, 1], 1],
    ]
    xor_data = [
        [[0, 0], 0],
        [[0, 1], 1],
        [[1, 0], 1],
        [[1, 1], 0],
    ]
    donut_data = [
        [ [ 0.0, 0.0 ], 0 ],
        [ [ 0.0, 1.0 ], 0 ],
        [ [ 1.0, 0.0 ], 0 ],
        [ [ 1.0, 1.0 ], 0 ],
        [ [ 0.5, 1.0 ], 0 ],
        [ [ 1.0, 0.5 ], 0 ],
        [ [ 0.0, 0.5 ], 0 ],
        [ [ 0.5, 0.0 ], 0 ],
        [ [ 0.5, 0.5 ], 1 ]
    ]

    fig, ax = plt.subplots(2, 2)
    plot_gate(ax[0, 0], "AND", "red", and_data)
    plot_gate(ax[0, 1], "OR", "blue", or_data)
    plot_gate(ax[1, 0], "XOR", "green", xor_data)
    plot_gate(ax[1, 1], "Donut", "purple", donut_data)

    plt.setp(ax,
        xticks=np.linspace(-0.5, 1.5, 6),
        yticks=np.linspace(-0.5, 1.5, 6)
    )

    plt.show()