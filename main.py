import math
from fractions import Fraction

#####################################
#   MATRIX FUNCTIONS
#####################################


def matGetColumn(M, c):
    col = []
    for i in range(len(M)):
        col.append(M[i][c])
    return col


def matMulMatrices(M1, M2):
    newMat = []
    for row in range(len(M1)):
        newMat.append([])
        for col in range(len(M2[0])):
            newMat[row].append(dot(M1[row], matGetColumn(M2, col)))
    return newMat


# # Multiply Multiple Matrices
# matrics - array of matrices
def matMulMatrices2(matrices, M):
    if len(matrices) <= 1:
        return matMulMatrices(matrices[0], M)
    return matMulMatrices2(matrices[0:-1], matMulMatrices(matrices[-1], M))


def matMulScalar(M, s):
    newMat = []
    for row in range(len(M)):
        newMat.append([])
        for col in range(len(M[0])):
            newMat[row].append(M[row][col] * s)
    return newMat


def matAddMatrices(M1, M2):
    newMat = []
    for row in range(len(M1)):
        newMat.append([])
        for col in range(len(M1[0])):
            newMat[row].append(M1[row][col] + M2[row][col])
    return newMat


def matTranspose(M):
    newMat = []
    for col in range(len(M[0])):
        newMat.append(matGetColumn(M, col))
    return newMat


def mat2x2Det(M):
    return M[0][0] * M[1][1] - M[0][1] * M[1][0]


def mat2x2Inv(M):
    return matMulScalar([[M[1][1], -M[0][1]], [-M[1][0], M[0][0]]],
                        1 / mat2x2Det(M))


def mat3x3Det(M):
    return (M[0][0] * mat2x2Det([[M[1][1], M[1][2]], [M[2][1], M[2][2]]]) -
            M[0][1] * mat2x2Det([[M[1][0], M[1][2]], [M[2][0], M[2][2]]]) +
            M[0][2] * mat2x2Det([[M[1][0], M[1][1]], [M[2][0], M[2][1]]]))


def mat4x4Det(M):
    det = 0
    for col in range(4):
        # Get cofactors of row 0
        cofactor = []
        for j in range(1, 4):
            cofactor.append([])
            for i in range(4):
                if i == col:
                    continue
                cofactor[-1].append(M[j][i])
        det += (-1)**(col) * M[0][col] * mat3x3Det(cofactor)
    return det


#Calculate the determinant of a square matrix of any size
def matDet(M, s):
    if (s <= 2):
        return mat2x2Det(M)
    det = 0
    for col in range(s):
        # Get cofactors of row 0
        cofactor = []
        for j in range(1, s):
            cofactor.append([])
            for i in range(s):
                if i == col:
                    continue
                cofactor[-1].append(M[j][i])
        #sum cofactors of row 0
        det += (-1)**(col) * M[0][col] * matDet(cofactor, len(cofactor))
    return det


def mat3x3Inv(M):
    adjM = []
    # Get adjugagte matrix
    for row in range(3):
        adjM.append([])
        for col in range(3):
            # Get 2x2 cofactor matrix
            cofactor = []
            for j in range(3):
                if j == row:
                    continue
                cofactor.append([])
                for i in range(3):
                    if i == col:
                        continue
                    # add M[i][j] to also swap diagonal cells
                    cofactor[-1].append(M[i][j])
            adjM[row].append(mat2x2Det(cofactor))
    # apply checkerboard pattern
    for row in range(3):
        for col in range(3):
            adjM[row][col] *= (-1)**(row + col)
    return matMulScalar(adjM, 1 / mat3x3Det(M))


def mat4x4Inv(M):
    adjM = []
    # Get adjugagte matrix
    for row in range(4):
        adjM.append([])
        for col in range(4):
            # Get 3x3 cofactor matrix
            cofactor = []
            for j in range(4):
                if j == row:
                    continue
                cofactor.append([])
                for i in range(4):
                    if i == col:
                        continue
                    # Use M[i][j] to swap diagonal cells
                    cofactor[-1].append(M[i][j])
            adjM[row].append(mat3x3Det(cofactor))

    # apply checkerboard pattern
    for row in range(4):
        for col in range(4):
            adjM[row][col] *= (-1)**(col + row)
    return matMulScalar(adjM, 1 / mat4x4Det(M))


# Calculate the inverse of a matrix of any size
# Algorithm: https://www.mathsisfun.com/algebra/matrix-inverse-minors-cofactors-adjugate.html
def matInverse(M):
    adjM = []
    size = len(M)
    # Get adjugagte matrix
    for row in range(size):
        adjM.append([])
        for col in range(size):
            # Get cofactor matrix
            cofactor = []
            for j in range(size):
                if j == row:
                    continue
                cofactor.append([])
                for i in range(size):
                    if i == col:
                        continue
                    # append M[i][j] instead of M[j][i] to also swap diagonal cells
                    cofactor[-1].append(M[i][j])
            adjM[row].append(matDet(cofactor, len(cofactor)))
    # apply checkerboard pattern
    for row in range(size):
        for col in range(size):
            adjM[row][col] *= (-1)**(row + col)
    return matMulScalar(adjM, 1 / matDet(M, size))


#####################################
#   VECTOR FUNCTIONS
#####################################


def dot(a, b):
    sum = 0
    for i in range(len(a)):
        sum += a[i] * b[i]
    return sum


# u,v are R3 vectors
def cross(u, v):
    return [
        mat2x2Det([[u[1], u[2]], [v[1], v[2]]]),
        -mat2x2Det([[u[0], u[2]], [v[0], v[2]]]),
        mat2x2Det([[u[0], u[1]], [v[0], v[1]]]),
    ]


def lenCross(u, v):
    return lenVec(cross(u, v))


def scalarTripleProduct(s, u, v):
    return dot(s, cross(u, v))


def mulVec(s, vec):
    newV = []
    for i in range(len(vec)):
        newV.append(vec[i] * s)
    return newV


def subVec(u, v):
    newV = []
    for i in range(len(u)):
        newV.append(u[i] - v[i])
    return newV


def addVec(u, v):
    newV = []
    for i in range(len(u)):
        newV.append(u[i] + v[i])
    return newV


def lenVec(v):
    return math.sqrt(dot(v, v))


def unitVec(v):
    return mulVec(1 / lenVec(v), v)


def proj(v1, u1):
    return mulVec((dot(v1, u1) / dot(u1, u1)), u1)


def orthogonalBasis(vectors):
    # gram-schmidt algorithm
    orthoVectors = [vectors[0]]
    for i in range(1, len(vectors)):
        orthoVector = vectors[i]
        for j in range(0, i):
            orthoVector = subVec(orthoVector, proj(orthoVector,
                                                   orthoVectors[j]))
        orthoVectors.append(orthoVector)
    return orthoVectors


def orthonormalBasis(vectors):
    orthoVectors = orthogonalBasis(vectors)
    # normalize orthogonal vectors
    for i, vec in enumerate(orthoVectors):
        orthoVectors[i] = unitVec(vec)
    return orthoVectors


# fix floating point error in 2d arrays
def cleanMat(M):
    for row in range(len(M)):
        for col in range(len(M[0])):
            M[row][col] = round(M[row][col], 2)
    return M


# fix floating point error in vectors
def cleanVec(v):
    for i in range(len(v)):
        v[i] = round(v[i], 2)
    return v


# convert to fraction
def fracMat(M):
    for row in range(len(M)):
        for col in range(len(M[0])):
            M[row][col] = Fraction(round(M[row][col], 2)).limit_denominator()
    return M


# convert to fraction
def fracVec(v):
    for i in range(len(v)):
        v[i] = Fraction(round(v[i], 2)).limit_denominator()
    return v


mat = [[2, 3, 4], [3, 0, -5], [12, 3, 4]]
print(fracMat(matInverse(mat)))
print(Fraction(1, 10))
