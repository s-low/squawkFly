#!/usr/local/bin/python

'''
    Pulled the Apex synchronisation method out of reconstruct.py to test
    it in isolation
'''
import os.path
import sys
import plotting as plot

debug = False
view = False


def synchroniseAtApex(pts_1, pts_2):
    syncd1 = []
    syncd2 = []
    shorter = []
    longer = []
    short_flag = 0

    if len(pts_1) < len(pts_2):
        shorter = pts_1
        longer = pts_2
        short_flag = 1
    else:
        shorter = pts_2
        longer = pts_1
        short_flag = 2

    diff = len(longer) - len(shorter)

    # find the highest y value in each point set
    apex1 = max(float(p[1]) for p in shorter)
    apex2 = max(float(p[1]) for p in longer)

    apex1_i = [i for i, y in enumerate(shorter) if y[1] == str(apex1)]
    apex2_i = [i for i, y in enumerate(longer) if y[1] == str(apex2)]

    if debug:
        print "\n------Apexes------"
        print "> Short:", apex1, apex1_i, "of", len(shorter)
        print "> Long:", apex2, apex2_i, "of", len(longer)

    shift = apex2_i[0] - apex1_i[0]

    # remove the front end dangle
    if debug:
        print "\nShift by:", shift

    if shift > 0:
        longer = longer[shift:]
        if debug:
            print "Longer front trimmed, new length:", len(longer)
    else:
        shorter = shorter[abs(shift):]
        if debug:
            print "Shorter front trimmed, new length:", len(shorter)

    remainder = diff - shift

    # remove the rear end dangle
    if remainder >= 0:
        if debug:
            print "\nTrim longer by remainder:", remainder
        index = len(longer) - remainder
        longer = longer[:index]

    if remainder < 0:
        index = len(shorter) - abs(remainder)
        if debug:
            print "\nShift > diff in lengths, trim the shorter end to:", index
        shorter = shorter[:index]

    if debug:
        print "New length of shorter:", len(shorter)
        print "New length of longer:", len(longer)

    # find the highest y value in each point set
    apex1 = max(float(p[1]) for p in shorter)
    apex2 = max(float(p[1]) for p in longer)

    apex1_i = [i for i, y in enumerate(shorter) if y[1] == str(apex1)]
    apex2_i = [i for i, y in enumerate(longer) if y[1] == str(apex2)]

    if debug:
        print "\nNew apex positions:"
        print apex1, apex1_i
        print apex2, apex2_i

    if short_flag == 1:
        syncd1 = shorter
        syncd2 = longer
    else:
        syncd1 = longer
        syncd2 = shorter

    if view and debug:
        plot.plot2D(syncd1, name='First Synced Trajectory')
        plot.plot2D(syncd2, name='Second Synced Trajectory')

    return syncd1, syncd2


pts1 = []
pts2 = []

folder = sys.argv[1]

out1 = os.path.join(folder, 'sync1.txt')
out2 = os.path.join(folder, 'sync2.txt')

with open(os.path.join(folder, 'pts1.txt')) as datafile:
    data1 = datafile.read()
    datafile.close

with open(os.path.join(folder, 'pts2.txt')) as datafile:
    data2 = datafile.read()
    datafile.close

data1 = data1.split('\n')
data2 = data2.split('\n')
pts1 = [(p.split()[0], p.split()[1]) for p in data1]
pts2 = [(p.split()[0], p.split()[1]) for p in data2]

syncd1, syncd2 = synchroniseAtApex(pts1, pts2)

if len(syncd1) == len(pts1):
    print "WORKED"
else:
    print "FAILED"

outfile1 = open(out1, 'w')
for p in syncd1:
    outfile1.write(str(p[0]) + ' ' + str(p[1]) + '\n')
outfile1.close()

outfile2 = open(out2, 'w')
for p in syncd2:
    outfile2.write(str(p[0]) + ' ' + str(p[1]) + '\n')
outfile2.close()
