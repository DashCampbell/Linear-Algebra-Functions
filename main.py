import math


def dot(a, b):
    sum = 0
    for i in range(len(a)):
        sum += a[i]*b[i]
    return sum


def mat2x2Det(M):
    return M[0][0] * M[1][1] - M[0][1]*M[1][0]


def mat3x3Det(M):
    return (M[0][0]*mat2x2Det([[M[1][1], M[1][2]], [M[2][1], M[2][2]]])
            - M[0][1]*mat2x2Det([[M[1][0], M[2][0]], [M[1][2], M[2][2]]])
            + M[0][2]*mat2x2Det([[M[1][0], M[2][0]], [M[1][1], M[2][1]]]))


def mulVec(s, vec):
    newV = []
    for i in range(len(vec)):
        newV.append(vec[i]*s)
    return newV


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


def mat3x3Inverse(M):
    newMat = []


def proj(u1, v1):
    return mulVec((dot(u1, v1)/dot(u1, u1)), u1)
