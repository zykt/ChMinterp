from collections import namedtuple
from math import sin, cos, pi
import matplotlib.pyplot as plt
import numpy as np


Point = namedtuple('Point', 'x fx')


def f(x):
    return cos(x + 2)/sin(x + 2) + x**2


def frange(start, end, steps):
    return [start + i*(end-start)/(steps-1) for i in range(steps)]


def altrange(start, end, steps):
    return [((start - end) * cos((2 * i + 1) / (2 * steps) * pi) + start + end)/2 for i in range(steps)]


def interpolate(points):
    def lagrange(x):
        acc = 0
        for i in points:
            mult_acc = i.fx
            for k in points:
                if k.x != i.x:
                    mult_acc *= (x-k.x)/(i.x-k.x)
            acc += mult_acc
        return acc
    return lagrange


start, end = -1, 1
steps = 5

# uncomment for task 3: f(x)*|x|
# oldf = f
# f = lambda x: oldf(x)*abs(x)

# comparison of f and oldf
# tst = plt.figure().add_subplot(111)
# tstxs = xs1 = np.linspace(start, end, num=200)
# tst.plot(tstxs, [oldf(x) for x in tstxs], 'k', label='Interpolated')
# tst.plot(tstxs, [f(x) for x in xs1], label='cos(x + 2)/sin(x + 2) + x**2')

# task 1
print('#1')
points1 = [Point(x, f(x)) for x in frange(start, end, steps)]
interp1 = interpolate(points1)

xs1 = np.linspace(start, end, num=200)
ax1 = plt.figure().add_subplot(111)
ax1.set_title('task #1\ncos(x + 2)/sin(x + 2) + x**2')
ax1.plot(xs1, [interp1(x) for x in xs1], 'k', label='Interpolated')
ax1.plot(xs1, [f(x) for x in xs1], label='cos(x + 2)/sin(x + 2) + x**2')
ax1.plot(start, f(start), 'or')
ax1.plot(end, f(end), 'or')
for point in points1:
    ax1.plot(point.x, point.fx, 'og')
ax1.legend(loc=3)

# task2
print('#2')
points2 = [Point(x, f(x)) for x in altrange(start, end, steps)]
interp2 = interpolate(points2)

xs2 = np.linspace(start, end, num=200)
ax2 = plt.figure().add_subplot(111)
ax2.set_title('task #2\ncos(x + 2)/sin(x + 2) + x**2')
ax2.plot(xs2, [interp2(x) for x in xs2], 'k', label='Interpolated')
ax2.plot(xs2, [f(x) for x in xs2], label='cos(x + 2)/sin(x + 2) + x**2')
ax2.plot(start, f(start), 'or')
ax2.plot(end, f(end), 'or')
for point in points2:
    ax2.plot(point.x, point.fx, 'og')
ax2.legend(loc=3)

plt.show()
