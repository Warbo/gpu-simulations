def make_grid(x, y, z):

	grid = []

	if x < 0:
		x_range = [a-1 for a in range(0,x,-1)]
	else:
		x_range = [a+1 for a in range(x)]
	
	if y < 0:
		y_range = [a-1 for a in range(0,y,-1)]
	else:
		y_range = [a+1 for a in range(y)]
	
	if z < 0:
		z_range = [a-1 for a in range(0,z,-1)]
	else:
		z_range = [a+1 for a in range(z)]

	for x in x_range:
		current_plane = []
	
		for y in y_range:
			current_row = []
		
			for z in z_range:
				current_row.append((x,y,z))
			
			current_plane.append(current_row[:])

		grid.append(current_plane[:])
	
	return grid

#print str(grid)

grids = [make_grid(10*x,10*y,10*z) for x in [-1,1] for y in [-1,1] for z in [-1,1]]

#print str(grids[0])

def slice(magnitude):
	slice = []
	for x in [a for a in range(-1*magnitude-1, magnitude+2) if a != 0]:
		for y in [a for a in range(-1*magnitude-1, magnitude+2) if a != 0]:
			for z in [a for a in range(-1*magnitude-1, magnitude+2) if a != 0]:
				if (abs(x)-1) + (abs(y)-1) + (abs(z)-1) == magnitude:
					slice.append((x,y,z))
	return slice

mem = []

count = 0

mag = 0
print "mem, x, y, z"
while mag < 4:
	#for grid in grids:
	#	#print str((slice(grid,mag))),
	#	#count += len(slice(mag))
	#print ''
	#print str(count)
	#print ''
	#count = 0
	for line in [str(a[0])+','+str(a[1])+','+str(a[2]) for a in slice(mag)]:
		print line
	print "999,999,999"
	mag += 1

#print str([(int(a[0]), int(a[1]), int(a[2]))for a in mem])
#for i,cell in enumerate(mem):
#	print str(i)+'='+str(cell)+'   '+str((abs(cell[0])+abs(cell[1])+abs(cell[2])))
