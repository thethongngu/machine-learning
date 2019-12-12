import numpy as np
import matplotlib.pyplot as plt

beta = 5


def kernel(X, Y, variance, a, l):
    return variance * (1 + (np.subtract(X, Y) ** 2) / (2 * a * l)) ** -a


# Data (34 x 2)
f = open("input.data", "r")
raw_data = f.read()
raw_data = [e.split(' ') for e in raw_data.split('\n')]
data = []
for e in raw_data[:-1]:
    data.append([float(e[0]), float(e[1])])
n = len(data)

# C (34 x 34)
C = np.zeros((34, 34), dtype=float)
for i in range(34):
    for j in range(34):
        C[i][j] = kernel(data[i], data[j]) + beta ** (-1)

fig, axs = plt.subplots(1, 1)

graph01 = axs[0]
graph01.set_title('Gaussian Process')

x = [data[i][0] for i in range(n)]
y = [data[i][1] for i in range(n)]
graph01.scatter(x, y, color='red')

