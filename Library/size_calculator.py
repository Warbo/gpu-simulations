#!/usr/bin/env python

"""Works out the best size for the grid cells, such that the average number of
particles in a cell is equal to 'density'."""

import sys

if len(sys.argv) < 2:
	print "Usage: size_calculator.py x_range y_range z_range N density"
	sys.exit()

x_range, y_range, z_range, N, density = sys.argv[1:]
volume = float(x_range) * float(y_range) * float(z_range)

print 'Size needed: ' + str( ( float(density) * (volume / float(N)) )**(1./3.) )