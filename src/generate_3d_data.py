#!/usr/local/bin/python

import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Physical constants
g = 9.8
m = 0.45
rho = 1.27
Cd = 0.25
A = math.pi * pow(0.5, 2.0)
alpha = rho * Cd * A / 2.0
beta = alpha / m

# Initial conditions
Z0 = 0.0
Y0 = 0.0
X0 = 0.0
Vx0 = 5.0
Vy0 = 20.0
Vz0 = 40.0

# Time steps
steps = 100
t_HIT = 2.0 * Vy0 / g
dt = t_HIT / steps

# With drag
X = list()
Z = list()
Y = list()
Vx = list()
Vz = list()
Vy = list()

X.append(X0)
Z.append(Z0)
Y.append(Y0)
Vz.append(Vz0)
Vy.append(Vy0)
Vx.append(Vx0)

stop = 0
for i in range(1, steps + 1):
    if stop != 1:
        speed = ((Vz[i - 1] ** 2) + (Vy[i - 1] ** 2) + (Vx[i - 1] ** 2)) ** 0.5

        # First calculate velocity
        Vx.append(Vx[i - 1] * (1.0 - beta * speed * dt))
        Vz.append(Vz[i - 1] * (1.0 - beta * speed * dt))
        Vy.append(Vy[i - 1] + (- g - beta * Vy[i - 1] * speed) * dt)

        # Now calculate position
        X.append(X[i - 1] + Vx[i - 1] * dt)
        Z.append(Z[i - 1] + Vz[i - 1] * dt)
        Y.append(Y[i - 1] + Vy[i - 1] * dt)

        # Stop if hits ground
        if Y[i] <= 0.0:
            stop = 1

# Plot results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(X, Z, Y, 'b.')
plt.show()

outfile = open('projectile_data.txt', 'w')
for x, z, y in zip(X, Z, Y):
    outfile.write(str(x) + ' ' + str(y) + ' ' + str(z) + '\n')

outfile.close()
