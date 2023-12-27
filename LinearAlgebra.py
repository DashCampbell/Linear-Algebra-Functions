import math

class Matrix:
    def __init__(self, row = 0, col = 0, *args) -> None:
        self.mat= list(args)
        self.row = row
        self.col = col
    
    def __str__(self) -> str:
        mat = ""
        for r in range(self.row):
            mat += "["
            for c in range(self.col - 1):
                mat += str(self.get(r, c))+", "        
            mat += str(self.get(r,self.col-1))+"]\n"
        return mat
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def get(self, row, col):
        """
        Gets an element in the matrix.
        :row: row number
        :col: column number
        """
        return self.mat[self.col*row+col]
    
    def assign(self, row: int , col: int, val):
        """
        Assigns an element in the matrix.
        :row: row number
        :col: column number
        :val: Scalar value
        """
        self.mat[self.col*row+col] = val

    def append(self, val):
        """
        Appends an element to the end of the matrix.
        :val: Scalar value
        """
        self.mat.append(val)

    def get_row(self, row: int) -> list:
        """
        Returns a row in the matrix as a list.
        """
        return list(self.get(row, c) for c in range(self.col))
    
    def get_col(self, col: int) -> list:
        """
        Returns a column in the matrix as a list.
        """
        return list(self.get(r, col) for r in range(self.row))

    def transpose(self):
        """
        Returns the transpose of the matrix.
        """
        matrix = Matrix(self.col, self.row)
        for c in range(self.col):
            for r in range(self.row):
                matrix.append(self.get(r, c))
        return matrix
        
    def det(self):
        """
        Returns the determinate of the matrix.
        """
        assert (self.row == self.col), "ERROR: Must be a square matrix."

        # determinate of 2x2 matrix
        if self.row <= 2:
            return self.get(0,0)*self.get(1,1) - self.get(0,1)*self.get(1,0)
        det = 0
        # get dot product of first row of matrix and determinates of cofactors of first row
        for c in range(self.col):
            # if element is 0, then skip
            if (self.get(0,c) == 0):
                continue

            cofactor = Matrix(self.row - 1, self.col - 1)
            for j in range(1, self.row):
                for i in range(self.col):
                    if i != c :
                        cofactor.append(self.get(j, i))
            det += (-1)**(c) * self.get(0, c) * cofactor.det()
        return det
    
    def inverse(self):
        """
        Returns the inverse matrix of the matrix.
        """
        determinate = self.det()
        assert determinate != 0, "ERROR: Determinate is zero. Matrix cannot be inverted."
        assert (self.row == self.col), "ERROR: Must be a square matrix."
        if (self.row <= 2):
            # inverse of 2x2 matrix 
            return Matrix(2,2,
                          self.get(1,1), -self.get(0,1),
                          -self.get(1,0), self.get(0,0)) / determinate
        adjM = Matrix(self.row, self.col)
        for r in range(self.row):
            for c in range(self.col):
                # get cofactor matrix for each cell
                cofactor = Matrix(self.row-1, self.col-1)
                for j in range(self.row):
                    if j != r:
                        for i in range(self.col):
                            if i != c:
                                cofactor.append(self.get(i,j))
                adjM.append(cofactor.det() * (-1)**(r+c))
        return adjM / determinate    

    def __add__(self, m2):
        """
        Returns the sum of two matrices.
        :m2: Matrix
        """
        assert isinstance(m2, Matrix), "TYPE ERROR: Object must be a matrix."
        assert (self.row == m2.row and self.col == m2.col), "ERROR: Rows and Columns must be the same size."
        matrix = Matrix(self.row, self.col)
        for r in range(self.row):
            for c in range(self.col):
                matrix.append(self.get(r,c) + m2.get(r,c))
        return matrix
    
    def __sub__(self, m2):
        """
        Returns the difference between two matrices.
        :m2: Matrix
        """
        assert isinstance(m2, Matrix), "TYPE ERROR: Object must be a matrix."
        assert (self.row == m2.row and self.col == m2.col), "ERROR: Rows and Columns must be the same size."
        matrix = Matrix(self.row, self.col)
        for r in range(self.row):
            for c in range(self.col):
                matrix.append(self.get(r,c) - m2.get(r,c))
        return matrix
    
    def __mul__(self, rh):
        """
        If type(rh) is a matrix, returns the product of two matrices.
        :If type(rh) is a scalar, returns the product of a matrix and scalar value.
        :rh: Scalar of Matrix
        """
        if isinstance(rh, Matrix):
            assert (self.col == rh.row), "ERROR: The number of columns of LH Matrix must be equal to number of rows as RH Matrix"
            matrix = Matrix(rh.row, self.col)
            for r in range(rh.row):
                for c in range(self.col):
                    matrix.append(Vector(*self.get_row(r)) * Vector(*rh.get_col(c)))
            return matrix
        else:
            # rh type is a number
            matrix = Matrix(self.row, self.col)
            for r in range(self.row):
                for c in range(self.col):
                    matrix.append(self.get(r,c)*rh)
            return matrix
        
    def __truediv__(self, rh):
        """
        Returns a matrix divided by a scalar.
        :rh: Scalar value
        """
        # rh type is a number
        matrix = Matrix(self.row, self.col)
        for r in range(self.row):
            for c in range(self.col):
                matrix.append(self.get(r,c)/rh)
        return matrix


class Vector:
    def __init__(self, *args) -> None:
        self.vec: list = args
    
    def __str__(self) -> str:
        return str(tuple(self.vec))
    
    def __repr__(self) -> str:
        return str(tuple(self.vec))
    
    def len(self) -> int:
        """
        Returns the length of a vector.
        """
        return math.sqrt(self*self)
    
    def size(self) -> int:
        """
        Returns the dimension of the vector.
        """
        return len(self.vec)
    
    def __add__(self, rh):
        """
        Returns the sum of two vectors
        :rh: Vector
        """
        assert (isinstance(rh, Vector)), "ERROR: Must be a vector."
        assert (len(self.vec) == len(rh.vec)), "ERROR: Both vectors must be the same size."
        return Vector(*((self.vec[i] + rh.vec[i]) for i in range(len(self.vec))))
    
    def __sub__(self, rh):
        """
        Returns the difference between two vectors.
        :rh: Vector
        """
        assert (isinstance(rh, Vector)), "ERROR: Must be a vector."
        assert (len(self.vec) == len(rh.vec)), "ERROR: Both vectors must be the same size."
        return Vector(*((self.vec[i] - rh.vec[i]) for i in range(len(self.vec))))
    
    def __mul__(self, rh):
        """
        If type(rh) is a vector, it returns the dot product between both vectors.
        :If type(rh) is a scalar, it returns the product of a vector and scalar value.
        """
        if isinstance(rh, Vector):
            # Dot Product
            assert (len(self.vec) == len(rh.vec)), "ERROR: Both vectors must be the same size."
            return __builtins__.sum(list((self.vec[i] * rh.vec[i]) for i in range(len(self.vec))))
        else:
            # Rh type is scalar
            return Vector(*((c*rh) for c in self.vec))
    
    def __truediv__(self, rh):
        """
        Returns the vector divided by a scalar.
        :rh: Scalar value.
        """
        return Vector(*((c/rh) for c in self.vec))
        
    def __floordiv__(self, rh):
        """
        Returns the cross product of the two vectors.
        :rh: Vector
        """
        assert (isinstance(rh, Vector)), "ERROR: Must be a vector"
        assert (self.size() == rh.size() == 3), "ERROR: Both vectors must have length 3"

        return Vector(  Matrix(2,2,
                            rh.vec[1],rh.vec[2],
                            self.vec[1], self.vec[2]).det(),
                        -Matrix(2,2,
                                rh.vec[0],rh.vec[2],
                                self.vec[0], self.vec[2]).det(),
                        Matrix(2,2,
                                rh.vec[0],rh.vec[1],
                                self.vec[0], self.vec[1]).det()
                    )
    def normalize(self):
        """
        Returns the unit vector.
        """
        return self / self.len()

    def proj(self, rh):
        """
        Returns the the vector projected onto the rh vector.
        :rh: Vector
        """   
        return rh * ((self * rh) / (rh*rh))
    
    @staticmethod
    def orthonormalBasis(*vectors) -> list:
        """
        Generates an orthonormal basis from a list of vectors. Uses the Gram-Schmidt algorithm.
        :vectors: List[Vector]
        """
         # gram-schmidt algorithm
        orthoVectors = [vectors[0].normalize()]

        for i in range(1, len(vectors)):
            orthoVector = vectors[i]
            for v in orthoVectors:
                orthoVector = orthoVector - orthoVector.proj(v)

            if orthoVector.len() > 0:
                orthoVectors.append(orthoVector.normalize())
        return orthoVectors


# Vector Examples
v1 = Vector(1, 2, 3)
v2 = Vector(1,0,-1)
sum = v1 * 2 + v2   #Output: (3, 4, 5)
difference = v1 - v2 / 2    #Output: (0.5, 2, 3.5)
dot_product = v1 * v2   #Output: -2
cross_product = Vector(1,0,0) // Vector(0,1,0)    #Output: (0, 0, -1)

# Matrix Examples
m1 = Matrix(2,2,            #Creates a 2x2 Matrix
            1 + 1j,1 - 1j,  #compatible with complex numbers
            4 - 5j, 5 + 8j)
m2 = Matrix(2,2,
            2, 5,
            3,-3)
m3 = Matrix(2, 3,   # Creates a 2x3 Matrix
            1, 2, 3,
            4, 4, 4)
inverse_matrix = m1.inverse()
determinate = m1.det()
double = m1 * 2
_sum = m1 + m2 
product = m1 * Matrix(2,2,
                      1,0,
                      0,1)