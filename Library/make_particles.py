#!/usr/bin/env python
"""Creates a file of particles for the simulation to use."""

import sys
import random

# For stripping away exponential notation.
# Taken from http://code.activestate.com/recipes/358361/
def non_exp_repr(x, is_int=False):
	"""Return a floating point representation without exponential notation.

	Result is a string that satisfies:
		float(result)==float(x) and 'e' not in result.
	
	>>> non_exp_repr(1.234e-025)
	'0.00000000000000000000000012339999999999999'
	>>> non_exp_repr(-1.234e+018)
	'-1234000000000000000.0'
	
	>>> for e in xrange(-50,51):
	...	 for m in (1.234, 0.018, -0.89, -75.59, 100/7.0, -909):
	...		 x = m * 10 ** e
	...		 s = non_exp_repr(x)
	...		 assert 'e' not in s
	...		 assert float(x) == float(s)

	"""
	if is_int:
		s = repr(int(x))
	else:
		s = repr(float(x))
	e_loc = s.lower().find('e')
	if e_loc == -1:
		return s

	mantissa = s[:e_loc].replace('.', '')
	exp = int(s[e_loc+1:])

	assert s[1] == '.' or s[0] == '-' and s[2] == '.', "Unsupported format"	 
	sign = ''
	if mantissa[0] == '-':
		sign = '-'
		mantissa = mantissa[1:]

	digitsafter = len(mantissa) - 1	 # num digits after the decimal point
	if exp >= digitsafter:
		if is_int:
			return sign + mantissa + '0'*(exp - digitsafter)
		return sign + mantissa + '0' * (exp - digitsafter) + '.0'
	elif exp <= -1:
		return sign + '0.' + '0' * (-exp - 1) + mantissa
	ip = exp + 1						# insertion point
	if is_int:
		return sign + mantissa[:ip]
	return sign + mantissa[:ip] + '.' + mantissa[ip:]

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
	outfile.write(\
		non_exp_repr(p, True)+','+\
		non_exp_repr(round(rand(xmin,xmax), 5))+','+\
		non_exp_repr(round(rand(ymin,ymax), 5))+','+\
		non_exp_repr(round(rand(zmin,zmax), 5))+\
		',0,0,0,0,0,0,0\n')
