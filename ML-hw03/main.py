import numpy as np


class UnivariateGaussianDataGenerator:

    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance
        self.standard_deviation = np.sqrt(variance)

    """ Z = (X - mu) * standard_deviation"""

    def sample(self):
        uniform_points = np.random.uniform(0, 1, 12)
        standard_normal_points = np.sum(uniform_points) - 6
        return standard_normal_points * self.standard_deviation + self.mean


class PolynomialBasicLinearDataGenerator:

    def __init__(self, n, w, a):
        self.n = n
        self.w = w
        self.a = a
        self.error = UnivariateGaussianDataGenerator(0, a)

    def sample(self):
        x = np.random.uniform(-1, 1, 1)
        y = 0.0
        basic = 1.0
        for i in range(self.n):
            y += self.w[i] * basic
            basic *= x
        return y + self.error.sample()


if __name__ == '__main__':
    print("Machine Learning - HW03")
