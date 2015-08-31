#!/usr/local/bin/python

from pylab import *
import math

# Physical constants
g = 9.8
m = 0.45
rho = 1.27
Cd = 0.25
A = math.pi * pow(0.5, 2.0)
alpha = rho * Cd * A / 2.0
beta = alpha / m

# Initial conditions
X0 = 0.0
Y0 = 0.0
Vx0 = 30.0
Vy0 = 10.0

# Time steps
steps = 100
t_HIT = 2.0 * Vy0 / g
dt = t_HIT / steps

# With drag
X_WD = list()
Y_WD = list()
Vx_WD = list()
Vy_WD = list()

X_WD.append(X0)
Y_WD.append(Y0)
Vx_WD.append(Vx0)
Vy_WD.append(Vy0)

stop = 0
for i in range(1, steps + 1):
    if stop != 1:
        speed = pow(pow(Vx_WD[i - 1], 2.0) + pow(Vy_WD[i - 1], 2.0), 0.5)

        # First calculate velocity
        Vx_WD.append(Vx_WD[i - 1] * (1.0 - beta * speed * dt))
        Vy_WD.append(Vy_WD[i - 1] + (- g - beta * Vy_WD[i - 1] * speed) * dt)

        # Now calculate position
        X_WD.append(X_WD[i - 1] + Vx_WD[i - 1] * dt)
        Y_WD.append(Y_WD[i - 1] + Vy_WD[i - 1] * dt)

        # Stop if hits ground
        if Y_WD[i] <= 0.0:
            stop = 1

# Plot results
plot(X_WD, Y_WD, 'b.')
show()

outfile = open('projectile_data.txt', 'w')
for x, y in zip(X_WD, Y_WD):
    outfile.write('0 ' + str(y) + ' ' + str(x) + '\n')

outfile.close()
