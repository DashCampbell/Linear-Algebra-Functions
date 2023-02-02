
def dot(a, b):
    sum = 0
    for i in range(len(a)):
        sum += a[i]*b[i]
    return sum


def scalar_mul_vec(scalar, vec):
    new_vec = []
    for i in range(len(vec)):
        new_vec.append(vec[i]*scalar)
    return new_vec


def orthogonal(a, b):
    return dot(a, b) == 0
