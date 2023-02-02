import math


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


def bOrthogonal(a, b):
    return dot(a, b) == 0


def proj(u, v):
    return scalar_mul_vec(dot(u, v)/dot(u, u), u)


def length(vec):
    sum = 0
    for i in range(len(vec)):
        sum += vec[i]**2
    return math.sqrt(sum)


def sub_vec(a, b):
    vec = []
    for i in range(len(a)):
        vec.append(a[i]-b[i])
    return vec


def add_vec(a, b):
    vec = []
    for i in range(len(a)):
        vec.append(a[i]+b[i])
    return vec


def norm_vec2d(vec):
    return (-vec[1], vec[0])


a = (279, -136)
c = (-208, -328)

norm = norm_vec2d(sub_vec(a, c))

print(add_vec(norm, a))
print(add_vec(norm, c))
