""" 
Answer for homework 01 of Machine Learning class 2019
Thong The Nguyen (阮世聰) 0860832
""" 

import matplotlib.pyplot as plt
import numpy as np

class Matrix:
    
    def __init__(self, h, w):
        self.height = h
        self.width = w
        self.e = [[0 for j in range(self.width)] for i in range(self.height)]

    @staticmethod
    def get_identity_matrix(n):
        identity_matrix = Matrix(n, n)
        for i in range (n):
            identity_matrix.e[i][i] = 1
        return identity_matrix

    def setElements(self, elements):
        self.e = elements

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
        if (self.width != mat.height):
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
        if (self.width != self.height):
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

        return (L, U)

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
                ans += '{} '.format( '%5.5f' % (self.e[i][j]))
            ans += '|\n'

        return ans

def test_add():
    A = Matrix(3, 3)
    A.setElements([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B = Matrix(3, 3)
    B.setElements([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(A.add_matrix(B))

def test_LU_decomposition_01():
    A = Matrix(3, 3)
    A.setElements([[2, -1, -2], [-4, 6, 3], [-4, -2, 8]])
    L, U = A.LU_decomposition()
    print(L)
    print(U)
    print(A)

def test_LU_decomposition_02():
    A = Matrix(2, 2)
    A.setElements([[4, 3], [6, 3]])
    L, U = A.LU_decomposition()
    print(L)
    print(U)
    print(A)

def test_LU_decomposition_03():
    A = Matrix(3, 3)
    A.setElements([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    L, U = A.LU_decomposition()
    print(L)
    print(U)
    print(A)    

def test_multiply():
    A = Matrix(3, 3)
    A.setElements([[2, -1, -2], [-4, 6, 3], [-4, -2, 8]])
    L, U = A.LU_decomposition()
    print(L)
    print(U)
    L.mul_matrix(U)
    print(L)

def test_tranpose():
    A = Matrix(3, 3)
    A.setElements([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B = A.tranpose()
    print(B)

def test_inverse01():
    A = Matrix(2, 2)
    A.setElements([[4, 7], [2, 6]])
    print(A.inverse())

def test_inverse02():
    A = Matrix(2, 2)
    A.setElements([[4, 3], [3, 2]])
    print(A.inverse())

# test_add()
# test_LU_decomposition_02()
# test_LU_decomposition_03()
# test_multiply()
# test_tranpose()
# test_inverse01()
# test_inverse02()

# ===================================

# Parameter
data_file_path = 'data-hw01.txt'
base = int(input("Input your number of polynomial bases (n): "))
lambda_parameter = int(input("Input your lambda: "))
print()

# Read data from file
f = open(data_file_path, 'r')
raw_data = f.readlines()
data = []
for point in raw_data:
    x, y = point.split(',')
    data.append([float(x), float(y)])

# Create design_matrix
num_data = len(data)
y = Matrix(num_data, 1)
design_matrix = Matrix(num_data, base)
for i in range(design_matrix.height):

    basic_func = 1
    x = data[i][0]
    y.e[i][0] = data[i][1]

    design_matrix.e[i][0] = basic_func
    for j in range(1, design_matrix.width):
        basic_func *= x
        design_matrix.e[i][j] = basic_func

# Answer for question a
lambda_I = Matrix.get_identity_matrix(num_data).mul_scalar(lambda_parameter)
design_matrix_T = design_matrix.tranpose()
A_T_A_plus_lambda_I_inverse = design_matrix_T.mul_matrix(design_matrix).add_matrix(lambda_I).inverse()
answer_a = A_T_A_plus_lambda_I_inverse.mul_matrix(design_matrix_T).mul_matrix(y)

loss = design_matrix.mul_matrix(answer_a).sub_matrix(y)
total_loss = 0
for i in range(len(loss.e)):
    total_loss += loss.e[i][0] * loss.e[i][0]

print('LSE:')
print('Fitting line: ', end='')
for i in range(base - 1, -1, -1):
    print('%s%s%s' % (answer_a.e[i][0], 'X^' + str(i) if i >= 1 else '', ' + ' if i != 0 else ''), end='')
print()
print('Total error: %s' % total_loss)


# Answer for question b
answer_b = Matrix(base, 1)
answer_b.setElements([[0] for i in range(answer_b.height)])

hessian_matrix = design_matrix_T.mul_matrix(design_matrix)
first = design_matrix_T.mul_matrix(design_matrix).mul_matrix(answer_b)
second = design_matrix_T.mul_matrix(y)
gradient = first.sub_matrix(second)

for i in range(10):
    answer_b = answer_b.sub_matrix(hessian_matrix.inverse().mul_matrix(gradient))
    gradient = design_matrix_T.mul_matrix(design_matrix).mul_matrix(answer_b).sub_matrix(design_matrix_T.mul_matrix(y))

loss = design_matrix.mul_matrix(answer_b).sub_matrix(y)
total_loss = 0
for i in range(len(loss.e)):
    total_loss += loss.e[i][0] * loss.e[i][0]

print()
print("Newton's Method:")
print('Fitting line: ', end='')
for i in range(base - 1, -1, -1):
    print('%s%s%s' % (answer_b.e[i][0], 'X^' + str(i) if i >= 1 else '', ' + ' if i != 0 else ''), end='')
print()
print('Total error: %s' % total_loss)


# Answer for question c
x_data = [data[i][0] for i in range(num_data)]
y_data = [data[i][1] for i in range(num_data)]
x_func = np.linspace(-5.5, 5.5, 100)
y_func_lse = []
y_func_newton = []

for i in range(len(x_func)):
    phi_x = 1
    res = answer_a.e[0][0]
    for j in range(1, base):
        phi_x *= x_func[i]
        res += answer_a.e[j][0] * phi_x
    y_func_lse.append(res)

for i in range(len(x_func)):
    phi_x = 1
    res = answer_a.e[0][0]
    for j in range(1, base):
        phi_x *= x_func[i]
        res += answer_b.e[j][0] * phi_x
    y_func_newton.append(res)

fig, axis = plt.subplots(2, 1)
axis[0].plot(x_func, y_func_lse)
axis[0].plot(x_data, y_data, 'o', color='red')
axis[0].yaxis.set_ticks([0, 20, 40, 60, 80, 100])

axis[1].plot(x_func, y_func_newton)
axis[1].plot(x_data, y_data, 'o', color='red')
axis[1].yaxis.set_ticks([0, 20, 40, 60, 80, 100])
plt.show()
