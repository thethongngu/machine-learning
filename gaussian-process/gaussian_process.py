import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

beta = 5
alpha = 1.0
length = 1.0
variance = 1.0


def kernel(x, y, variance, alpha, length):
    return variance * ((1 + cdist(x, y, 'sqeuclidean') / (2 * alpha * length ** 2)) ** -alpha)


# Read data
f = open("input.data", "r")
raw_data = f.read()
raw_data = [e.split(' ') for e in raw_data.split('\n')]
train_x = np.zeros((34, 1))
train_y = np.zeros((34, 1))
for i in range(34):
    train_x[i] = float(raw_data[i][0])
    train_y[i] = float(raw_data[i][1])

# GP
mu = np.zeros((34, 2))
C = kernel(train_x, train_x, variance, alpha, length) + (1.0 / beta)  # (34, 34)

# GP predict
test_x = np.linspace(-60, 60, 200).reshape(-1, 1)  # (200, 1)

k_x_xstar = kernel(train_x, test_x, variance, alpha, length)  # (34, 200)
k_star = kernel(test_x, test_x, variance, alpha, length) + (1.0 / beta)  # (200, 200)

C_inv = np.linalg.inv(C)
mu_y = np.matmul(np.matmul(k_x_xstar.T, C_inv), train_y)  # (200, 34) (34, 34) (34, 1) = (200, 1)
variance_y = np.subtract(k_star, np.matmul(np.matmul(k_x_xstar.T, C_inv), k_x_xstar))  # (200. 200)
upper_y = mu_y + 1.96 * (np.diag(np.sqrt(variance_y)).reshape(-1, 1))
lower_y = mu_y - 1.96 * (np.diag(np.sqrt(variance_y)).reshape(-1, 1))

# Draw graph
fig, axs = plt.subplots(1, 1)
axs.set_title('Gaussian Process')
axs.set_xlim(-60.0, 60.0)
axs.plot(test_x.ravel(), mu_y.ravel(), color='blue')
axs.fill_between(test_x.ravel(), upper_y.ravel(), lower_y.ravel(), facecolor='pink')
axs.plot(test_x.ravel(), upper_y.ravel(), color='red')
axs.plot(test_x.ravel(), lower_y.ravel(), color='red')
axs.scatter(train_x, train_y, color='black')

fig.show()


def loss_func(params):
    cov = kernel(train_x, train_x, params[0], params[1], params[2]) + (1.0 / beta)  # (34, 34)
    res = 0.5 * np.log(np.linalg.det(cov)) + \
          0.5 * np.matmul(np.matmul(train_y.T, np.linalg.inv(cov)), train_y) + \
          0.5 * 34 * np.log(2 * np.pi)
    return res[0][0]


# Optimize parameters
result = minimize(fun=loss_func, x0=np.array([variance, alpha, length]))
variance = result.x[0]
alpha = result.x[1]
length = result.x[2]

# GP
C = kernel(train_x, train_x, variance, alpha, length) + (1.0 / beta)  # (34, 34)

# GP predict
C_inv = np.linalg.inv(C)

k_x_xstar = kernel(train_x, test_x, variance, alpha, length)  # (34, 200)
k_star = kernel(test_x, test_x, variance, alpha, length) + (1.0 / beta)  # (200, 200)

mu_y = np.matmul(np.matmul(k_x_xstar.T, C_inv), train_y)  # (200, 34) (34, 34) (34, 1) = (200, 1)
variance_y = np.subtract(k_star, np.matmul(np.matmul(k_x_xstar.T, C_inv), k_x_xstar))  # (200. 200)
upper_y = mu_y + 1.96 * (np.diag(np.sqrt(variance_y)).reshape(-1, 1))
lower_y = mu_y - 1.96 * (np.diag(np.sqrt(variance_y)).reshape(-1, 1))

# Draw graph
fig, axs = plt.subplots(1, 1)
axs.set_title('Gaussian Process')
axs.set_xlim(-60.0, 60.0)
axs.plot(test_x.ravel(), mu_y.ravel(), color='blue')
axs.fill_between(test_x.ravel(), upper_y.ravel(), lower_y.ravel(), facecolor='pink')
axs.plot(test_x.ravel(), upper_y.ravel(), color='red')
axs.plot(test_x.ravel(), lower_y.ravel(), color='red')
axs.scatter(train_x, train_y, color='black')

fig.show()
