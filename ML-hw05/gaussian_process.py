import numpy as np

beta = 5


def kernel(X, Y):
    pass


# Data (34 x 2)
f = open("input.data", "r")
raw_data = f.read()
raw_data = [e.split(' ') for e in raw_data.split('\n')]
data = []
for e in raw_data[:-1]:
    data.append([float(e[0]), float(e[1])])

# C (34 x 34)
C = np.zeros((34, 34), dtype=float)
for i in range(34):
    for j in range(34):
        C[i][j] = kernel(data[i], data[j]) + beta ** (-1)

# C = np.array()
