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

    def set_element(self, i, j, value):
        self.e[i][j] = value

    def set_elements(self, elements):
        self.e = elements

    def mul_scalar(self, number):
        ans = Matrix(self.height, self.width)
        for i in range(self.height):
            for j in range(self.width):
                ans.e[i][j] = self.e[i][j] * number
        return ans

    def add_matrix(self, mat):
        if self.width != mat.width or self.height != mat.height:
            print("Cannot add matrix (%s, %s) x (%s, %s)" % (self.height, self.width, mat.height, mat.width))
            return self

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
            print("Cannot multiple matrix (%s, %s) x (%s, %s)" % (self.height, self.width, mat.height, mat.width))
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
                ans += '{} '.format('%10.5f' % (self.e[i][j]))
            ans += '|\n'

        return ans
