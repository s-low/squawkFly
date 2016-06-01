#!/usr/local/bin/python

import webbrowser
import os.path
from yattag import Doc
from yattag import indent
import sys

''' generate_x3d.py

Given a 3 column space delimited file of X Y Z coordinates in the regular
coordinate system output by reconstruction, generate the X3DOM XHTML for each
ball point, and slot it into the template file.

Uses yattag python package.

> pip install yattag

Save as 3d.xhtml in the clip directory.

'''

doc, tag, text = Doc().tagtext()

# Input the clip directory
clip = sys.argv[1]
source_data = os.path.join(clip, '3d_out.txt')
html_target = os.path.join(clip, 'graphs/3d.xhtml')

template_file = open('x3d/template.xhtml', 'r')
template = template_file.readlines()
template_file.close()

with open(source_data) as datafile:
    data3d = datafile.read()
    datafile.close()

# Gobble blanks at EOF if there
data = data3d.split('\n')
if data[-1] in ['\n', '\r\n', '']:
    data.pop(-1)

# Pop off the goalposts - we don't need them later
bl = data.pop(0)
tl = data.pop(0)
tr = data.pop(0)
br = data.pop(0)

bl_z = float(bl.split()[2])
tl_z = float(tl.split()[2])
tr_z = float(tr.split()[2])
br_z = float(br.split()[2])

# Work out ditance to goal so we know how to displace traj from origin
depth = (bl_z + tl_z + tr_z + br_z) / 4
diff = 16 - float(depth)
shift_string = '0 ' + str(diff) + ' 0'

# Generate XHTML with yattag
with tag('transform', translation=shift_string):

    # first point (origin) includes the style definition for the ball
    first = data.pop(1)
    with tag('group', DEF='ball'):
        with tag('shape'):
            doc.stag('sphere', radius='0.1')
            with tag('appearance'):
                doc.stag('material', diffuseColor='0 0 1')

    for row in data:
        x = row.split()[0]
        y = row.split()[1]
        z = row.split()[2]
        coords = x + ' ' + z + ' ' + y
        with tag('transform', translation=coords):
            doc.stag('group', USE='ball')

# Add some indentation
result = indent(doc.getvalue())

# Insert into template contents
template.insert(166, result)
template = "".join(template)

# Write to graphs/3d.xhtml
outfile = open(html_target, 'w')
outfile.write(template)
outfile.close()

# Open it in the web browser automatically
webbrowser.open('file://' + os.path.realpath(html_target))
