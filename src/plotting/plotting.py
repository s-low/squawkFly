''' plotting.py

    A bundle of useful plotting functions using pyplot and mplot3D.

    plot3D
    plot2D
    plotOrderedBar
    plotEpilines

    Credit to Remy F on StackOverflow for the bounding box trick to simulate
    equal aspect ratio in 3D axes.

    http://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot3D(data_3d, name='3D Plot'):

    X = [point[0] for point in data_3d]
    Y = [point[1] for point in data_3d]
    Z = [point[2] for point in data_3d]

    fig = plt.figure(name)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, marker='o', c='b', zdir='y')

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array(
        [max(X) - min(X), max(Y) - min(Y), max(Z) - min(Z)]).max()
    Xb = 0.5 * max_range * \
        np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (max(X) + min(X))
    Yb = 0.5 * max_range * \
        np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (max(Y) + min(Y))
    Zb = 0.5 * max_range * \
        np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (max(Z) + min(Z))

    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_zlabel('Y (m)')

    plt.locator_params(nbins=4)
    plt.show()


# can provide an optional second set of data
def plot2D(pts1, pts2=[], name='2D Plot', lims=(0, 0)):
    onlyoneset = False
    x1 = [p[0] for p in pts1]
    y1 = [p[1] for p in pts1]

    try:
        x2 = [p[0] for p in pts2]
        y2 = [p[1] for p in pts2]
    except IndexError:
        onlyoneset = True

    fig = plt.figure(name)
    ax = plt.axes()

    if lims[0] != 0:
        ax.set_xlim(0, lims[0])
        ax.set_ylim(lims[1], 0)

    ax.scatter(x1, y1, color='r')
    if not onlyoneset:
        ax.scatter(x2, y2, color='b')
    ax.set_xlabel('X (px)')
    ax.set_ylabel('Y (px)')

    plt.show()


def plotEpilines(lines, pts, index):
    name = 'Corresponding Epilines on Image ' + str(index)
    fig = plt.figure(name)
    ax = plt.axes(xlim=(0, 1280))

    for r in lines:
        a, b, c = r[0], r[1], r[2]
        x = np.linspace(0, 1280, 5)
        y = ((-c) - (a * x)) / b
        ax.plot(x, y)

    x = []
    y = []
    for p in pts:
        x.append(p[0])
        y.append(p[1])

    ax.plot(x, y, 'r.')

    plt.show()


# given a list of numbers, visualise them to spot outliers
def plotOrderedBar(data, name='Indexed Bar', xlabel='X', ylabel='Y'):
    x = []
    y = []
    mean = sum(data) / len(data)

    for i, d in enumerate(data):
        x.append(i)
        y.append(d)

    fig = plt.figure(name)
    ax = plt.axes()
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.bar(x, y)
    ax.axhline(mean)

    plt.show()
