import numpy as np


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


def logistic_regression(D1, D2):
    w = Matrix(3, 1)
    x = Matrix(1, 3)

    times = 0
    while True:
        times += 1
        last_w = w
        for p in D1:
            x.set_elements([[p[0], p[1], 1]])
            xw = x.mul_matrix(w).e[0][0]
            gradient = x.tranpose().mul_scalar(0 - (1.0 / (1.0 + np.exp(-xw))))
            w = w.add_matrix(gradient)

        for p in D2:
            x.set_elements([[p[0], p[1], 1]])
            xw = x.mul_matrix(w).e[0][0]
            gradient = x.tranpose().mul_scalar(1 - (1.0 / (1.0 + np.exp(-xw))))
            w = w.add_matrix(gradient)

        change = w.sub_matrix(last_w)
        if np.sqrt(change.e[0][0] ** 2 + change.e[1][0] ** 2 + change.e[2][0] ** 2) < 0.01:
            break

    confusion = [[0, 0], [0, 0]]

    for p in D1:
        x.set_elements([[p[0], p[1], 1]])
        xw = x.mul_matrix(w).e[0][0]
        sigmoid = 1.0 / (1.0 + np.exp(-xw))
        if sigmoid >= 0.5:
            confusion[0][1] += 1
        else:
            confusion[0][0] += 1

    for p in D2:
        x.set_elements([[p[0], p[1], 1]])
        xw = x.mul_matrix(w).e[0][0]
        sigmoid = 1.0 / (1.0 + np.exp(-xw))
        if sigmoid >= 0.5:
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
    print("Sensitivity (Successfully predict cluster 1): %s" % str(confusion[0][0] / (confusion[0][0] + confusion[0][1])))
    print("Sensitivity (Successfully predict cluster 2): %s" % str(confusion[1][1] / (confusion[1][1] + confusion[1][0])))


if __name__ == '__main__':
    print("Machine Learning - HW04")
    print("阮世聰 - 0860832")

    # print("Input N: ", end='')
    # n = int(input())
    # print("Input mx1, vx1, my1, vy1, mx2, vx2, my2, vy2: ", end='')
    # mx1, vx1, my1, vy1, mx2, vx2, my2, vy2 = [int(val) for val in input().split(' ')]
    n = 50
    mx1, vx1, my1, vy1, mx2, vx2, my2, vy2 = (1, 2, 1, 2, 10, 2, 10, 2)

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

    logistic_regression(D1, D2)
