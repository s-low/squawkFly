#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as mate


with open("output.txt") as f:
    data = f.read()

data = data.split('\n')

x = [row.split(' ')[0] for row in data]
y = [row.split(' ')[1] for row in data]


fig = plt.figure()

ax1 = fig.add_subplot(111)

ax1.set_title("Object Centroids")    
ax1.set_xlabel('Graphical x')
ax1.set_ylabel('Graphical y')

ax1.plot(x ,y, 'ro', label='Detections')

leg = ax1.legend()

plt.show()