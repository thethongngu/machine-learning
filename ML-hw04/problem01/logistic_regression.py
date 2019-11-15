import numpy as np
import matplotlib.pyplot as plt


class Matrix:

    def __init__(self, h, w):
        self.height = h
        self.width = w
        self.e = [[0 for j in range(self.width)] for i in range(self.height)]

    @staticmethod
    def get_identity_matrix(n):
        identity_matrix = Matrix(n, n)
        for i in range(n):
            identity_matrix.e[i][i] = 1
        return identity_matrix

    def set_elements(self, elements):
        self.e = elements

    def set_element(self, i, j, value):
        self.e[i][j] = value

    def mul_scalar(self, number):
        ans = Matrix(self.height, self.width)
        for i in range(self.height):
            for j in range(self.width):
                ans.e[i][j] = self.e[i][j] * number
        return ans

    def product_diagonal(self):
        res = self.e[0][0]
        for i in range(1, self.height):
            res *= self.e[i][i]
        return res

    def add_matrix(self, mat):
        ans = Matrix(self.height, self.width)
        for i in range(self.height):
            for j in range(self.width):
                ans.e[i][j] = self.e[i][j] + mat.e[i][j]
        return ans

    def sub_matrix(self, mat):
        ans = Matrix(self.height, self.width)
        for i in range(self.height):
            for j in range(self.width):
                ans.e[i][j] = self.e[i][j] - mat.e[i][j]
        return ans

    def tranpose(self):
        ans = Matrix(self.width, self.height)
        for i in range(self.height):
            for j in range(self.width):
                ans.e[j][i] = self.e[i][j]
        return ans

    def mul_matrix(self, mat):
        if self.width != mat.height:
            print("Cannot multiple matrix (%s, %s) x (%s, %s)" % (self.width, self.height, mat.width, mat.height))
            return self

        ans = Matrix(self.height, mat.width)

        for i in range(self.height):
            for j in range(mat.width):
                sum = 0.0
                for k in range(mat.height):
                    sum += self.e[i][k] * mat.e[k][j]
                ans.e[i][j] = sum

        return ans

    """ 
    LU decomposition for square matrix (n x n) 
    Return tuple (L, U)
    Doolittle Algorithm 
    """

    def LU_decomposition(self):
        if self.width != self.height:
            print("Does not support inverse non-square matrix")
            exit

        L = Matrix.get_identity_matrix(self.height)
        U = Matrix(self.height, self.width)

        for i in range(self.height):

            for j in range(i, self.width):
                sum = 0
                for k in range(self.width):
                    sum += L.e[i][k] * U.e[k][j]
                U.e[i][j] = self.e[i][j] - sum

            for j in range(i + 1, self.width):
                sum = 0
                for k in range(self.width):
                    sum += L.e[j][k] * U.e[k][i]
                L.e[j][i] = (self.e[j][i] - sum) / U.e[i][i]

        return L, U

    def is_invertible(self):
        L, U = self.LU_decomposition()
        return L.product_diagonal() * U.product_diagonal() != 0

    def inverse(self):

        y = Matrix(self.height, self.width)
        x = Matrix(self.height, self.width)
        I = Matrix.get_identity_matrix(self.height)
        L, U = self.LU_decomposition()

        # solve L * y = I
        for i in range(L.height):
            for j in range(y.height):
                sum = 0
                for k in range(y.height):
                    sum += L.e[i][k] * y.e[k][j]
                y.e[i][j] = I.e[i][j] - sum

        # solve U * x = y
        for i in range(U.height - 1, -1, -1):
            for j in range(x.height):
                sum = 0
                for k in range(x.height):
                    sum += U.e[i][k] * x.e[k][j]
                x.e[i][j] = (y.e[i][j] - sum) / U.e[i][i]

        return x

    def __str__(self):
        ans = ''
        for i in range(self.height):
            ans += '|'
            for j in range(self.width):
                ans += '{} '.format('%5.5f' % (self.e[i][j]))
            ans += '|\n'

        return ans


class GaussianGenerator:
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance
        self.standard_deviation = np.sqrt(variance)

    """ Z = (X - mu) * standard_deviation"""

    def sample(self):
        uniform_points = np.random.uniform(0, 1, 12)
        standard_normal_points = np.sum(uniform_points) - 6
        return standard_normal_points * self.standard_deviation + self.mean


def my_sigmoid(x, w):
    return 1.0 / (1.0 + np.exp(x.mul_scalar(-1).mul_matrix(w).e[0][0]))


def norm2(c):
    return np.sqrt(c.e[0][0] ** 2 + c.e[1][0] ** 2 + c.e[2][0] ** 2)


def show_result(D1, D2, w):
    confusion = [[0, 0], [0, 0]]
    x = Matrix(1, 3)
    for p in D1:
        x.set_elements([[p[0], p[1], 1]])
        res = my_sigmoid(x, w)
        if res >= 0.5:
            confusion[0][1] += 1
        else:
            confusion[0][0] += 1
    for p in D2:
        x.set_elements([[p[0], p[1], 1]])
        res = my_sigmoid(x, w)
        if res >= 0.5:
            confusion[1][1] += 1
        else:
            confusion[1][0] += 1
    print("Gradient descent:")
    print("w:")
    print(w)
    print("Confusion matrix: ")
    print("             | Predict cluster 1  |  Predict cluster 2")
    print("Is cluster 1 |     %s             |     %s            " % (confusion[0][0], confusion[0][1]))
    print("Is cluster 2 |     %s             |     %s            " % (confusion[1][0], confusion[1][1]))
    print(
        "Sensitivity (Successfully predict cluster 1): %s" % str(confusion[0][0] / (confusion[0][0] + confusion[0][1])))
    print(
        "Sensitivity (Successfully predict cluster 2): %s" % str(confusion[1][1] / (confusion[1][1] + confusion[1][0])))


def logistic_regression(D1, D2):
    w = Matrix(3, 1)
    x = Matrix(1, 3)

    times = 0
    while True:
        times += 1
        last_w = w
        for p in D1:
            x.set_elements([[p[0], p[1], 1]])
            gradient = x.tranpose().mul_scalar(0 - my_sigmoid(x, w))
            w = w.add_matrix(gradient)

        for p in D2:
            x.set_elements([[p[0], p[1], 1]])
            gradient = x.tranpose().mul_scalar(1 - my_sigmoid(x, w))
            w = w.add_matrix(gradient)

        change = w.sub_matrix(last_w)
        if norm2(change) < 0.01:
            break

    show_result(D1, D2, w)
    return w


def newton(D1, D2):
    A = Matrix(n * 2, 3)
    y = Matrix(n * 2, 1)
    for i in range(n):
        A.set_element(i, 0, D1[i][0])
        A.set_element(i, 1, D1[i][1])
        A.set_element(i, 2, 1)
        y.set_element(i, 0, 0)

    for i in range(n, n * 2):
        A.set_element(i, 0, D2[i - n][0])
        A.set_element(i, 1, D2[i - n][1])
        A.set_element(i, 2, 1)
        y.set_element(i, 0, 1)

    D = Matrix(n * 2, n * 2)
    x = Matrix(1, 3)
    w = Matrix(3, 1)

    diff = Matrix(2 * n, 1)
    while True:
        last_w = w
        for i in range(n * 2):
            x.set_elements([[A.e[i][0], A.e[i][1], A.e[i][2]]])
            xw = x.mul_matrix(w).e[0][0]
            diff.set_element(i, 0, y.e[i][0] - my_sigmoid(x, w))  # (2n, 1)
            D.set_element(i, i, np.exp(-xw) / ((1.0 + np.exp(-xw)) ** 2))  # (2n, 2n)

        gradient = A.tranpose().mul_matrix(diff)  # (3, 2n) x (2n, 1)
        H = A.tranpose().mul_matrix(D).mul_matrix(A)  # (3, 2n) x (2n, 2n) x (2n, 3)
        if H.is_invertible():
            w = w.add_matrix(H.inverse().mul_matrix(gradient))  # (3, 1)
        else:
            w = w.add_matrix(gradient)

        if norm2(w.sub_matrix(last_w)) < 0.01:
            break

    show_result(D1, D2, w)
    return w


def draw_data(D1, D2, w1, w2):
    fig, axs = plt.subplots(1, 3)

    # ------------- graph 01 ---------------------
    graph01 = axs[0]
    graph01.set_title('Ground truth')

    x = [D1[i][0] for i in range(n)]
    y = [D1[i][1] for i in range(n)]
    graph01.scatter(x, y, color='red')

    x = [D2[i][0] for i in range(n)]
    y = [D2[i][1] for i in range(n)]
    graph01.scatter(x, y, color='blue')

    x1, x2, y1, y2 = [], [], [], []
    x = Matrix(1, 3)
    data = D1
    data.extend(D2)

    # ------------- graph 02 ---------------------
    graph01 = axs[1]
    graph01.set_title('Gradient descent')

    for p in data:
        x.set_elements([[p[0], p[1], 1]])
        res = my_sigmoid(x, w1)
        if res >= 0.5:
            x1.append(p[0])
            y1.append(p[1])
        else:
            x2.append(p[0])
            y2.append(p[1])

    graph01.scatter(x1, y1, color='blue')
    graph01.scatter(x2, y2, color='red')

    # ------------- graph 01 ---------------------
    graph01 = axs[2]
    graph01.set_title("Newton's method")

    for p in data:
        x.set_elements([[p[0], p[1], 1]])
        res = my_sigmoid(x, w2)
        if res >= 0.5:
            x1.append(p[0])
            y1.append(p[1])
        else:
            x2.append(p[0])
            y2.append(p[1])

    graph01.scatter(x1, y1, color='blue')
    graph01.scatter(x2, y2, color='red')

    fig.show()


if __name__ == '__main__':
    print("Machine Learning - HW04")
    print("阮世聰 - 0860832")

    # print("Input N: ", end='')
    # n = int(input())
    # print("Input mx1, vx1, my1, vy1, mx2, vx2, my2, vy2: ", end='')
    # mx1, vx1, my1, vy1, mx2, vx2, my2, vy2 = [int(val) for val in input().split(' ')]
    n = 50
    mx1, vx1, my1, vy1, mx2, vx2, my2, vy2 = (1, 2, 1, 2, 10, 2, 10, 2)
    # mx1, vx1, my1, vy1, mx2, vx2, my2, vy2 = (1, 2, 1, 2, 3, 4, 3, 4)

    D1 = []
    x1_generator = GaussianGenerator(mx1, vx1)
    y1_generator = GaussianGenerator(my1, vy1)
    for i in range(n):
        D1.append([x1_generator.sample(), y1_generator.sample()])

    D2 = []
    x2_generator = GaussianGenerator(mx2, vx2)
    y2_generator = GaussianGenerator(my2, vy2)
    for i in range(n):
        D2.append([x2_generator.sample(), y2_generator.sample()])

    w1 = logistic_regression(D1, D2)
    print("--------------------------------------------------------")
    w2 = newton(D1, D2)
    draw_data(D1, D2, w1, w2)
