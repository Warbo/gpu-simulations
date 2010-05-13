#!/usr/bin/env python

"""Tests a particle spreadsheet for validity."""

import sys

if len(sys.argv) == 1:
	print "Usage: input_validator.py filename"
	sys.exit()

infile = open(sys.argv[1], 'r')

# Discard first line
infile.readline()
N = int(infile.readline().split(',')[0])
size = float(infile.readline().split(',')[0])

for lineno, line in enumerate(infile.xreadlines()):
	try:
		int(line.split(',')[0])
		A,B,C,D,E,F,G,H,I,J = [float(i) for i in line.split(',')[1:]]
	except:
		print 'ERROR ON LINE '+lineno

infile.close()

