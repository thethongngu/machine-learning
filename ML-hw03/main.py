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


class SequentialEstimation:

    def __init__(self, dist):
        self.dist = dist

    def estimate(self):
        print("Data point source function: N(%s, %s)" % (self.dist.mean, self.dist.variance))

        mean, variance = 0.0, 0.0

        for n in range(1, 10000):
            x = self.dist.sample()

            new_mean = float(x + mean * (n - 1)) / n
            new_variance = variance + np.square(mean) - np.square(new_mean) + \
                           ((np.square(x) - variance - np.square(mean)) / n)
            mean = new_mean
            variance = new_variance

            print("Add data point: %s" % x)
            print("Mean = %s   Variance = %s" % (mean, variance))


if __name__ == '__main__':
    print("Machine Learning - HW03")

    print("Input (m, s): ", end="")
    m, s = [float(i) for i in input().split()]
    normal_distribution = UnivariateGaussianDataGenerator(m, s)
    answer02 = SequentialEstimation(normal_distribution)
    answer02.estimate()
