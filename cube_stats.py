#!/usr/bin/env python

# Gets statistics about cubes

size = 10

print '10x10x10 cube:'

total_cubes = 10*10*10

print 'Total: '+str(total_cubes)

corners = 8

print 'Corners: 8'

total_cubes -= corners

centre = (size - 2)**3

print "Centre cubes: "+str(centre)

total_cubes -= centre

faces = 6*((size-2)**2)

print "Faces: "+str(faces)

total_cubes -= faces

print "Edges: "+str(total_cubes)

