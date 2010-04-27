#!/usr/bin/env python

# Gets statistics about cubes

x = 8
y = 6
z = 7

print str(x)+'x'+str(y)+'x'+str(z)+' cube:'

total_cubes = x*y*z

print 'Total: '+str(total_cubes)

if x < 2:
	if y < 2:
		if z < 2:
			corners = 1
		else:
			corners = 2
	elif z < 2:
		corners = 2
	else:
		corners = 4
elif y < 2:
	if z < 2:
		corners = 2
	else:
		corners = 4
elif z < 2:
	corners = 4
else:
	corners = 8

print 'Corners: '+str(corners)

total_cubes -= corners

if total_cubes < 0:
	import sys
	sys.exit()

centre = (x - 2)*(y - 2)*(z - 2)

print "Centre cubes: "+str(centre)

total_cubes -= centre

faces = 2*((x-2)*(y-2))+2*((x-2)*(z-2))+2*((y-2)*(z-2))

print "Faces: "+str(faces)

total_cubes -= faces

print "Edges: "+str(total_cubes)
