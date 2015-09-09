#!/usr/local/bin/python

''' beehive.py

Generate a graph of a goalmouth in the X-Y plane, with the final positions
of each ball trajectory in the session.

Input is the session folder containing all clip directories.

'''
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.image as image

# get the relevant session
session = sys.argv[1]
outfilename = None
try:
    outfilename = sys.argv[2]
except IndexError:
    pass

# get the names of all clip subdirectories in the session
clips = next(os.walk(session))[1]
shots_x = []
shots_y = []

for sub in clips:
    # get the 3D trajectory data for each clip
    t_path = os.path.join(session, sub, '3d_out.txt')
    with open(t_path) as t_file:
        trajectory = t_file.read()
        t_file.close()
    trajectory = trajectory.split('\n')

    # remove blank line if it's there
    if trajectory[-1] in ['\n', '\r\n', '']:
        trajectory.pop(-1)

    # get the last point + project into X-Y plane
    final = trajectory[-1]
    x = float(final.split()[0])
    y = float(final.split()[1])
    shots_x.append(x)
    shots_y.append(y)

session_name = os.path.basename(session)

# Plotting
fig = plt.figure('Goalmouth Hit Points for Session: ' + session_name)
ax = fig.add_subplot(111, aspect='equal')
ax.plot(shots_x, shots_y, 'ko', markersize=10, fillstyle='none')

# Axes
plt.ylim([-1, 3.5])
plt.xlim([-1, 8.3])
ax.set_xlabel('Distance from left post / m')
ax.set_ylabel('Height / m')

# Draw the goalposts b=bottom, l=left, r=right, t=top
bl_x = 0
bl_y = 0

tl_x = 0
tl_y = 2.4

tr_x = 7.3
tr_y = 2.4

br_x = 7.3
br_y = 0

ax.plot([bl_x, tl_x], [bl_y, tl_y], color='k', linewidth=2)
ax.plot([tl_x, tr_x], [tl_y, tr_y], color='k', linewidth=2)
ax.plot([tr_x, br_x], [tr_y, br_y], color='k', linewidth=2)

plt.show()

if outfilename is not None:
    print "Save:", outfilename
    fig.savefig(outfilename)
