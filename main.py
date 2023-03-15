import math


def dot(a, b):
    sum = 0
    for i in range(len(a)):
        sum += a[i]*b[i]
    return sum


def mat2x2Det(M):
    return M[0][0] * M[1][1] - M[0][1]*M[1][0]


def mat2x2Inv(M):
    return matMulScalar([[M[1][1], -M[0][1]], [-M[1][0], M[0][0]]], 1/mat2x2Det(M))


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
                    cofactor[len(cofactor)-1].append(M[i][j])
            adjM[row].append(mat2x2Det(cofactor))
    # swap diagonal cells
    # adjM[0][1], adjM[1][0] = adjM[1][0], adjM[0][1]
    # adjM[0][2], adjM[2][0] = adjM[2][0], adjM[0][2]
    # adjM[1][2], adjM[2][1] = adjM[2][1], adjM[1][2]
    # apply checkerboard pattern
    for row in range(3):
        for col in range(3):
            if (row+col) % 2 != 0:
                adjM[row][col] *= -1
    return matMulScalar(adjM, 1/mat3x3Det(M))


def mat3x3Det(M):
    return (M[0][0]*mat2x2Det([[M[1][1], M[1][2]],
                               [M[2][1], M[2][2]]])
            - M[0][1]*mat2x2Det([[M[1][0], M[1][2]],
                                 [M[2][0], M[2][2]]])
            + M[0][2]*mat2x2Det([[M[1][0], M[1][1]],
                                 [M[2][0], M[2][1]]]))

# u,v are R3 vectors


def cross(u, v):
    return [mat2x2Det([[u[1], u[2]],
                       [v[1], v[2]]]),
            -mat2x2Det([[u[0], u[2]],
                       [v[0], v[2]]]),
            mat2x2Det([[u[0], u[1]],
                       [v[0], v[1]]])]


def lenCross(u, v):
    return lenVec(cross(u, v))


def scalarTripleProduct(s, u, v):
    return dot(s, cross(u, v))


def mulVec(s, vec):
    newV = []
    for i in range(len(vec)):
        newV.append(vec[i]*s)
    return newV


def subVec(u, v):
    newV = []
    for i in range(len(u)):
        newV.append(u[i]-v[i])
    return newV


def addVec(u, v):
    newV = []
    for i in range(len(u)):
        newV.append(u[i]+v[i])
    return newV


def lenVec(v):
    return math.sqrt(dot(v, v))


def unitVec(v):
    return mulVec(1/lenVec(v), v)


def matGetColumn(M, n):
    col = []
    for i in range(len(M)):
        col.append(M[i][n])
    return col


def matMulMatrices(M1, M2):
    newMat = []
    for row in range(len(M1)):
        newMat.append([])
        for col in range(len(M2[0])):
            newMat[row].append(dot(M1[row], matGetColumn(M2, col)))
    return newMat

# # Multiply Multiple Matrices


# def matMulMatrices(M, matrices):
#     if len(matrices) == 1:
#         return matMulMatrices(M, matrices[0])
#     return matMulMatrices(matMulMatrices(M, matrices[0]), matrices[1:])


def matMulScalar(M, s):
    newMat = []
    for row in range(len(M)):
        newMat.append([])
        for col in range(len(M[0])):
            newMat[row].append(M[row][col]*s)
    return newMat


def matAddMatrices(M1, M2):
    newMat = []
    for row in range(len(M1)):
        newMat.append([])
        for col in range(len(M1[0])):
            newMat[row].append(M1[row][col]+M2[row][col])
    return newMat


def matTranspose(M):
    newMat = []
    for col in range(len(matGetColumn(M, 0))):
        newMat.append(matGetColumn(M, col))
    return newMat


def proj(u1, v1):
    return mulVec((dot(u1, v1)/dot(u1, u1)), u1)


# print(addVec(mulVec(c1, a), mulVec(c2, b)))
m = [[1, 1, 4], [0, 2, -6], [1, 1, -1]]
print(matMulMatrices(mat3x3Inv(m), [[-5], [-18], [-22]]))
print(matMulMatrices(mat2x2Inv([[-1, 5], [1, -4]]), [[27], [-22]]))
