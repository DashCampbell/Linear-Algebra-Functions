
u = (16, -8, 17, 4)
v = (12, 17, 4, -4)


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


def proj(u, v):
    return scalar_mul_vec(dot(u, v)/dot(u, u), u)


print(proj(u, v))
print(dot(u, v))
print(dot(u, u))
