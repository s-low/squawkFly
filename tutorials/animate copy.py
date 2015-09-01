#!/usr/bin/python

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

# set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
scat, = ax.plot([], [], 'bo')

# initialization function: plot the background of each frame
def init():
    scat.set_data([], [])
    return scat,

def animate(i):
    x = np.linspace(0, 2, 50) # x = 100 numbers between 0 and 2
    y = np.sin(2 * np.pi * (x - 0.01 * i)) # y = sin(2 * pi * x * frame)
    scat.set_data(x, y)
    return scat,

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=False)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()