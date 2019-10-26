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

        for sample_size in range(1, 10000):
            x = self.dist.sample()

            new_mean = float(x + mean * (sample_size - 1)) / sample_size
            new_variance = variance + np.square(mean) - np.square(new_mean) + \
                           ((np.square(x) - variance - np.square(mean)) / sample_size)
            mean = new_mean
            variance = new_variance

            print("Add data point: %s" % x)
            print("Mean = %s   Variance = %s" % (mean, variance))


# noinspection DuplicatedCode
class BayesianLinearRegression:

    def __init__(self, b, n, a, w):
        self.generator = PolynomialBasicLinearDataGenerator(n, w, a)
        self.n = n
        self.a = a
        self.w = w
        self.b = b

    def estimate(self):
        mean = Matrix(self.n, 1)
        variance = Matrix.get_identity_matrix(n).mul_scalar(1.0 / self.b)
        x_data = []
        y_data = []
        mean_10 = mean
        mean_50 = mean
        variance_10 = variance
        variance_50 = variance

        for i in range(150):
            x, y = self.generator.sample()
            x_data.append(x.e[1][0])
            y_data.append(y)

            old_variance_inv = variance.inverse()
            variance = old_variance_inv.add_matrix(x.mul_matrix(x.tranpose()).mul_scalar(1.0 / self.a)).inverse()
            mean = variance.mul_matrix(old_variance_inv.mul_matrix(mean).add_matrix(x.mul_scalar(y * (1.0 / self.a))))

            predictive_mean = mean.tranpose().mul_matrix(x).e[0][0]
            predictive_variance = x.tranpose().mul_matrix(variance.mul_matrix(x)).e[0][0] + self.a

            if i == 10:
                mean_10 = mean
                variance_10 = variance
            if i == 50:
                mean_50 = mean
                variance_50 = variance

            print("Add data point (%s, %s):" % (x.e[1][0], y), end="\n\n")
            print("Posterior mean:")
            print(mean)
            print("Posterior variance: ")
            print(variance)
            print("Predictive distribution ~ N(%s, %s)" % (predictive_mean, predictive_variance))
            print("-------------------------------------------------------------------")

        fig, axs = plt.subplots(2, 2)

        # ------------- graph 01 ---------------------
        graph01 = axs[0][0]
        graph01.set_title('Ground truth')
        graph01.set_xlim(-2.0, 2.0)
        graph01.set_ylim(-14, 22)
        ground_f = np.poly1d([row[0] for row in np.flip(self.w.e)])
        x = np.linspace(-2.0, 2.0, 30)
        y = ground_f(x)
        graph01.plot(x, y, color='black')

        y_upper_var = ground_f(x) + self.a
        graph01.plot(x, y_upper_var, color='red')

        y_lower_var = ground_f(x) - self.a
        graph01.plot(x, y_lower_var, color='red')

        # -------------------- graph 02 --------------------------
        graph02 = axs[0][1]
        graph02.set_title('Predict result')
        graph02.set_xlim(-2.0, 2.0)
        graph02.set_ylim(-14, 22)
        graph02.scatter(x_data, y_data, color="blue")
        x = np.linspace(-2.0, 2.0, 30)
        predict_f = np.poly1d([row[0] for row in np.flip(mean.e)])
        predict_y = predict_f(x)
        graph02.plot(x, predict_y, color="black")

        upper_variance = []
        lower_variance = []
        for i in range(len(x)):
            x_bold = Matrix(self.n, 1)

            basic = 1.0
            for j in range(self.n):
                x_bold.set_element(j, 0, basic)
                basic *= x[i]

            upper_variance.append(
                predict_y[i] + (x_bold.tranpose().mul_matrix(variance.mul_matrix(x_bold)).e[0][0] + self.a)
            )
            lower_variance.append(
                predict_y[i] - (x_bold.tranpose().mul_matrix(variance.mul_matrix(x_bold)).e[0][0] + self.a)
            )

        graph02.plot(x, upper_variance, color='red')
        graph02.plot(x, lower_variance, color='red')

        # --------------------- graph03 ------------------------
        graph03 = axs[1][0]
        graph03.set_title('After 10 incomes')
        graph03.set_xlim(-2.0, 2.0)
        graph03.set_ylim(-14, 22)
        graph03.scatter(x_data[:10], y_data[:10], color="blue")
        x = np.linspace(-2.0, 2.0, 30)
        predict_f = np.poly1d([row[0] for row in np.flip(mean_10.e)])
        predict_y = predict_f(x)
        graph03.plot(x, predict_y, color="black")

        upper_variance = []
        lower_variance = []
        for i in range(len(x)):
            x_bold = Matrix(self.n, 1)

            basic = 1.0
            for j in range(self.n):
                x_bold.set_element(j, 0, basic)
                basic *= x[i]

            upper_variance.append(
                predict_y[i] + (x_bold.tranpose().mul_matrix(variance_10.mul_matrix(x_bold)).e[0][0] + self.a)
            )
            lower_variance.append(
                predict_y[i] - (x_bold.tranpose().mul_matrix(variance_10.mul_matrix(x_bold)).e[0][0] + self.a)
            )
        graph03.plot(x, upper_variance, color='red')
        graph03.plot(x, lower_variance, color='red')

        # -------------------- graph04 ------------------------------
        graph04 = axs[1][1]
        graph04.set_title('After 50 incomes')

        graph04.set_xlim(-2.0, 2.0)
        graph04.set_ylim(-14, 22)
        graph04.scatter(x_data[: 50], y_data[: 50], color="blue")
        x = np.linspace(-2.0, 2.0, 30)
        predict_f = np.poly1d([row[0] for row in np.flip(mean_50.e)])
        predict_y = predict_f(x)
        graph04.plot(x, predict_y, color="black")

        upper_variance = []
        lower_variance = []
        for i in range(len(x)):
            x_bold = Matrix(self.n, 1)

            basic = 1.0
            for j in range(self.n):
                x_bold.set_element(j, 0, basic)
                basic *= x[i]

            upper_variance.append(
                predict_y[i] + (x_bold.tranpose().mul_matrix(variance_50.mul_matrix(x_bold)).e[0][0] + self.a)
            )
            lower_variance.append(
                predict_y[i] - (x_bold.tranpose().mul_matrix(variance_50.mul_matrix(x_bold)).e[0][0] + self.a)
            )
        graph04.plot(x, upper_variance, color='red')
        graph04.plot(x, lower_variance, color='red')

        fig.show()


if __name__ == '__main__':
    print("Machine Learning - HW03")
    print("阮世聰 - 0860832")
    print()

    # ------------- Answer 02 --------------------- #
    print("Input (m, s): ", end=" ")
    m, s = [float(i) for i in input().split()]

    normal_distribution = UnivariateGaussianDataGenerator(m, s)
    answer02 = SequentialEstimation(normal_distribution)
    answer02.estimate()

    # ------------- Answer 03 --------------------- #
    print("Input n:", end=" ")
    n = int(input())
    print("Input a:", end=" ")
    a = int(input())
    print("Input w:", end=" ")
    elements = [[float(i)] for i in input().split()]
    w = Matrix(n, 1)
    w.set_elements(elements)
    print("Input b:", end=" ")
    b = int(input())

    answer03 = BayesianLinearRegression(b, n, a, w)
    answer03.estimate()
