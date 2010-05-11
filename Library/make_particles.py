#!/usr/bin/env python
"""Creates a file of particles for the simulation to use."""

import sys
import random

def rand(low, high):
	return (random.random() * (high-low)) + low

if len(sys.argv) == 1:
	print "Usage: make_particles.py x_min x_max y_min y_max z_min z_max size number filename"
	sys.exit()

(xmin,xmax,ymin,ymax,zmin,zmax,size,number,filename) = sys.argv[1:]

outfile = open(filename, 'w')
outfile.write('ID,x,y,z,x_vel,y_vel,z_vel,x_acc,y_acc,z_acc,mass\n')
outfile.write(number+', <- Particle Number\n')
outfile.write(size+', <- Particle size\n')

xmin = float(xmin)
xmax = float(xmax)
ymin = float(ymin)
ymax = float(ymax)
zmin = float(zmin)
zmax = float(zmax)

for p in range(int(number)):
	outfile.write(str(p)+','+str(rand(xmin,xmax))+','+str(rand(ymin,ymax))+',' \
		+str(rand(zmin,zmax))+',0,0,0,0,0,0,0\n')
