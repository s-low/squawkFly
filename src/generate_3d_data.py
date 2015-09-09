#!/usr/local/bin/python

''' generate_3d_data.py

Little script to generate X Y Z coordinates for a ball launched into the air.

'''


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

# Initial position: origin
z0 = 0.0
y0 = 0.0
X0 = 0.0

# Initial velocity
v_x0 = 5.0
v_y0 = 20.0
v_z0 = 40.0

X = []
Y = []
Z = []
v_x = []
v_y = []
v_z = []

# Time steps
steps = 100
t_HIT = 2.0 * vy0 / g
dt = t_HIT / steps

X.append(x0)
Y.append(y0)
Z.append(z0)
v_z.append(v_z0)
v_y.append(v_y0)
v_x.append(v_x0)

stop = 0
for i in range(1, steps + 1):
    if stop != 1:
        speed = ((v_z[i - 1] ** 2) + (v_y[i - 1] ** 2) + (v_x[i - 1] ** 2)) ** 0.5

        # First calculate velocity
        v_x.append(v_x[i - 1] * (1.0 - beta * speed * dt))
        v_z.append(v_z[i - 1] * (1.0 - beta * speed * dt))
        v_y.append(v_y[i - 1] + (- g - beta * v_y[i - 1] * speed) * dt)

        # Now calculate position
        X.append(x[i - 1] + v_x[i - 1] * dt)
        Z.append(z[i - 1] + v_z[i - 1] * dt)
        Y.append(y[i - 1] + v_y[i - 1] * dt)

        # Stop if hits ground
        if y[i] <= 0.0:
            stop = 1

# Plot results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(x, z, y, 'b.')
plt.show()

outfile = open('projectile_data.txt', 'w')
for x, z, y in zip(x, z, y):
    outfile.write(str(x) + ' ' + str(y) + ' ' + str(z) + '\n')

outfile.close()
