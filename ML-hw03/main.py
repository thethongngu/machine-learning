import matplotlib.pyplot as plt
import numpy as np

from matrix import Matrix


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
        x = np.random.uniform(-1, 1, 1)[0]
        basic = 1.0
        x_bold = Matrix(self.n, 1)

        for i in range(self.n):
            x_bold.set_element(i, 0, basic)
            basic *= x

        y = w.tranpose().mul_matrix(x_bold).e[0][0]
        return x_bold, y + self.error.sample()


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


class BayesianLinearRegression:

    def __init__(self, b, n, a, w):
        self.generator = PolynomialBasicLinearDataGenerator(n, w, a)
        self.n = n
        self.a = a
        self.w = w
        self.b = b

    def estimate(self):
        mean = Matrix(self.n, 1)
        variance = Matrix.get_identity_matrix(n).mul_scalar(1 / self.b)
        x_data = []
        y_data = []

        for i in range(100):
            x, y = self.generator.sample()
            x_data.append(x.e[1][0])
            y_data.append(y)

            old_variance_inv = variance.inverse()
            variance = old_variance_inv.add_matrix(x.mul_matrix(x.tranpose()).mul_scalar(self.a)).inverse()
            mean = variance.mul_matrix(old_variance_inv.mul_matrix(mean).add_matrix(x.mul_scalar(y * self.a)))

            predictive_mean = mean.tranpose().mul_matrix(x).e[0][0]
            predictive_variance = x.tranpose().mul_matrix(variance.mul_matrix(x)).e[0][0] + float(1.0 / self.a)

            print("Add data point (%s, %s):" % (x.e[1][0], y), end="\n\n")
            print("Posterior mean:")
            print(mean)
            print("Posterior variance: ")
            print(variance)
            print("Predictive distribution ~ N(%s, %s)" % (predictive_mean, predictive_variance))
            print("-------------------------------------------------------------------")

        fig, axs = plt.subplots(2, 2)

        graph01 = axs[0][0]
        graph01.set_title('Ground truth')
        graph01.set_xlim(-2.0, 2.0)
        graph01.set_ylim(-20, 20)
        ground_f = np.poly1d([row[0] for row in np.flip(self.w.e)])
        x = np.linspace(-2.0, 2.0, 30)
        y = ground_f(x)
        graph01.plot(x, y, color='black')

        upper_var_f = np.poly1d([row[0] for row in np.flip(self.w.e)])
        y_upper_var = upper_var_f(x) + self.a
        graph01.plot(x, y_upper_var, color='red')

        lower_var_f = np.poly1d([row[0] for row in np.flip(self.w.e)])
        y_lower_var = lower_var_f(x) - self.a
        graph01.plot(x, y_lower_var, color='red')

        graph02 = axs[0][1]
        graph02.set_title('Predict result')
        graph02.set_xlim(-2.0, 2.0)
        graph02.set_ylim(-20, 20)
        graph02.scatter(x_data, y_data, color="blue")
        predict_f = np.poly1d([row[0] for row in np.flip(self.mean.e)])

        # axs[1, 0].plot(x, -y, 'tab:green')
        axs[1, 0].set_title('After 10 incomes')
        # axs[1, 1].plot(x, -y, 'tab:red')
        axs[1, 1].set_title('After 50 incomes')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()

        fig.show()


if __name__ == '__main__':
    print("Machine Learning - HW03")
    print("阮世聰 - 0860832")
    print()

    # ------------- Answer 02 --------------------- #
    #     print("Input (m, s): ", end=" ")
    #     m, s = [float(i) for i in input().split()]
    #
    #     normal_distribution = UnivariateGaussianDataGenerator(m, s)
    #     answer02 = SequentialEstimation(normal_distribution)
    #     answer02.estimate()

    # ------------- Answer 03 --------------------- #
    #     print("Input n:", end=" ")
    #     n = int(input())
    #     print("Input a:", end=" ")
    #     a = int(input())
    #     print("Input w:", end=" ")
    #     elements = [[float(i)] for i in input().split()]]
    #     w = Matrix(n, 1).set_elements(elements)
    #     print("Input b:", end=" ")
    #     b = int(input())

    n = 4
    a = 1.0
    w = Matrix(n, 1)
    w.set_elements([[1], [2], [3], [4]])
    b = 1.0

    answer03 = BayesianLinearRegression(b, n, a, w)
    answer03.estimate()
