#!/usr/bin/env python

import sys
from pymeta.grammar import OMeta

def concatenate_dictionaries(original_dictionary, to_add):
	"""Returns the original_dictionary with the contents of the
	dictionaries given in the list too_add inserted."""
	for dic in to_add:
		original_dictionary.update(dic)
	return original_dictionary

dump_grammar = """

grid ::= <token "GRID:"> <address>:grid_address '{' <grid_contents>*:c '}'					=> concatenate_dictionaries({'TYPE':'GRID', 'ADDRESS':grid_address}, c)

grid_contents ::= <x_size> | <y_size> | <z_size> | <population> | <cells> | <particles>

x_size ::= <token "X:"> <integer>:size ','*									=> {'X':size}

y_size ::= <token "Y:"> <integer>:size ','*									=> {'Y':size}

z_size ::= <token "Z:"> <integer>:size ','*									=> {'Z':size}

x_pos ::= <token "X:"> <decimal>:pos ','*									=> {'X':pos}

y_pos ::= <token "Y:"> <decimal>:pos ','*									=> {'Y':pos}

z_pos ::= <token "Z:"> <decimal>:pos ','*									=> {'Z':pos}

population ::= <token "POPULATION:"> <integer>:pop ','*								=> {'POPULATION':pop}

cells ::= <token "CELLS:STARTINGAT:"> <address>:startaddress '{' <cells_contents>*:c '}' ','*			=> {'CELLS':c, 'CELLSSTART':startaddress}

cells_contents ::= <cell>:c											=> c

cell ::= <token "CELL:"> <address>:celladdress '{' <cell_contents>*:c '}' ','*					=> concatenate_dictionaries({'TYPE':'CELL', 'ADDRESS':celladdress}, c)

cell_contents ::= <neighbours>:n										=> n
                | <first_particle>:p										=> p

neighbours ::= <token "NEIGHBOURS:STARTINGAT:"> <address>:startaddress '{' <neighbour>*:n '}' ','*		=> {'NEIGHBOURS':n, 'NEIGHBOURSSTART':startaddress}

neighbour ::= <token "LOCATION:"> <address>:location ','*							=> location

first_particle ::= <token "FIRSTPARTICLE:"> <address>:particleaddress ','*					=> {"FIRSTPARTICLE":particleaddress}

particles ::= <token "PARTICLES:STARTINGAT:"> <address>:startaddress '{' <particles_contents>*:c '}' ','*	=> {'PARTICLES':c, 'PARTICLESSTART':startaddress}

particles_contents ::= <particle>:p										=> p

particle ::= <token "PARTICLE:"> <address>:particleaddress '{' <particle_contents>*:c '}' ','*			=> concatenate_dictionaries({'TYPE':'PARTICLE', 'ADDRESS':particleaddress}, c)

particle_contents ::= <x_pos> | <y_pos> | <z_pos> | <container> | <next>

container ::= <token "CONTAINER:"> <address>:containeraddress ','*						=> {'CONTAINER':containeraddress}

next ::= <token "NEXT:"> <address>:nextparticle ','*								=> {'NEXT':nextparticle}

address ::= <integer>:memaddress										=> memaddress
         | <token "NULL">											=> None

integer ::= '-' <dig>+:ds											=> -1*int(''.join(ds))
          | <dig>+:ds												=> int(''.join(ds))
      
dig ::= '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'
          
decimal ::= <integer>:whole '.' <dig>+:frac									=> float(str(whole)+'.'+''.join(frac))
          | <integer>
"""

# Make a class which reads the dump grammar
grammar = OMeta.makeGrammar(dump_grammar, globals())

# Exit if we've been given no file to check
if len(sys.argv) < 2:
	print "Usage: memchecker.py DUMP_FILE"
	sys.exit()
	
# Open the given file
infile = open(sys.argv[1], 'r')

inlines = ''.join(infile.readlines())

infile.close()

# Instantiate the dump grammar with the input
matcher = grammar(inlines)

# Try to find a dump in the input
grid = matcher.apply('grid')

## Now we should have a dictionary of the program internals, so run some
## tests

# If there are cells, do checks on them
if 'CELLS' in grid.keys():
	
	# Make sure that CELLS doesn't appear without CELLSTART
	if 'CELLSSTART' not in grid.keys():
		print "Grid has cells, but no cells start address!"
		sys.exit()
	
	# If we've more than one cell then the spacing between all cells should be equal
	if len(grid['CELLS']) > 1:
		cell_spacing =  grid['CELLS'][1]['ADDRESS'] - grid['CELLS'][0]['ADDRESS']
		
		# Make sure the first two cells follow on from each other in memory
		if cell_spacing < 1:
			print "Second cell ("+str(grid['CELLS'][1]['ADDRESS'])+") doesn't continue from first ("+str(grid['CELLS'][0]['ADDRESS'])+")!"
			print "ALL CELLS:"+str([' '+str(c['ADDRESS'])+' ' for c in grid['CELLS']])
			sys.exit()
	
	# If we've not got more than one cell then set the spacing to None so that we know not to check it
	else:
		cell_spacing = None
	
	# Do checks for each cell
	for cell in grid['CELLS']:
		
		# Make sure that the cell is not stored before the cells' start address
		if cell['ADDRESS'] < grid['CELLSSTART']:
			print "Cells start at "+grid['CELLSSTART']+' but found a cell starting at '+cell['ADDRESS']+'!'
			sys.exit()
		
		# If we've more than one cell then make sure that the cell is an integer number of spaces from the cell start address
		if cell_spacing is not None and (cell['ADDRESS'] - grid['CELLSSTART']) % cell_spacing > 0:
			print "Cell "+str(cell['ADDRESS'])+" is not an integer number of cells away from start address "+str(grid['CELLSSTART'])+" (assuming cell spacing of "+str(cell_spacing)+")"
			sys.exit()
		
		# Do checks on this cell's first particle, if it has one
		if ('FIRSTPARTICLE' in cell.keys()) and (cell['FIRSTPARTICLE'] is not None) and ('PARTICLESSTART' in grid.keys()):
			
			# Make sure that the first particle does not come before the particles' start address
			if cell['FIRSTPARTICLE'] < grid['PARTICLESSTART']:
				print "Cell contains particle "+cell['FIRSTPARTICLE']+' but particles start at '+grid['PARTICLESSTART']
			
			# Make sure that the first particle is in the list of particles
			if cell['FIRSTPARTICLE'] not in [p['ADDRESS'] for p in grid['PARTICLES']]:
				print "Cell contains particle "+cell['FIRSTPARTICLE']+" but not in particles list "+str([p['ADDRESS'] for p in grid['PARTICLES']])
				sys.exit()
		
		# If we've no particles then do some further checks
		else:
			# If there are particles in the grid then make sure none of them gives this cell as its container (since the cell thinks it's empty)
			if 'PARTICLES' in grid.keys():
				for particle in grid['PARTICLES']:
					if particle['CONTAINER'] == cell['ADDRESS']:
						print "Cell "+str(cell['ADDRESS'])+" says it has no particles, but particle "+str(particle['ADDRESS'])+" says it is in this cell!"
						sys.exit()
		
		# If we've enabled cell neighbour allocation then look for problems in the neighbour assignments			
		if 'NEIGHBOURS' in cell.keys():
			
			# Make sure we've got 26 neighbours + ourself
			if len(cell['NEIGHBOURS']) != 27:
				print "Neighbour length is not 27 for cell "+str(cell['ADDRESS'])+"!"
			
			if cell_spacing is not None:
				# Find the position of this cell
				# Our indexing scheme for the cells should be (y*z)*x_index+(z)*y_index+z_index, so let's invert that
				start = (cell['ADDRESS'] - grid['CELLSSTART']) / cell_spacing
				z = ((start % (grid['Y']*grid['Z'])) % grid['Z'])
				y = (((start - z) % (grid['Y']*grid['Z'])) / grid['Z'])
				x = (((start - z) - (grid['Z']*y)) / (grid['Y']*grid['Z']))

				# Make sure we're in the grid
				if x < 0 or x >= grid['X'] or y < 0 or y >= grid['Y'] or z < 0 or z >= grid['Z']:
					print "Cell "+str(cell['ADDRESS'])+" is at position "+str((x,y,z))+" but grid size is "+str((grid['X'],grid['Y'],grid['Z']))
					sys.exit()

				# These are the total neighbours we predict
				predicted_n = ['self', 'left', 'right', 'leftforward', 'rightforward', 'leftback', 'rightback', 'forward', 'back',
					'up','upleft','upright','upforward','upleftforward','uprightforward','upback','upleftback','uprightback',
					'down','downleft','downright','downforward','downleftforward','downrightforward','downback','downleftback','downrightback']
				
				# If we define x as 'left' and 'right' then having x as zero loses the left neighbours
				if x == 0:
					for n in ['left','leftforward','leftback','upleft','upleftforward','upleftback','downleft','downleftforward','downleftback']:
						try:
							predicted_n.remove(n)
						except ValueError:
							pass

				# and if it is a maximum then we lose right neighbours
				if x == grid['X']-1:
					for n in ['right','rightforward','rightback','upright','uprightforward','uprightback','downright','downrightforward','downrightback']:
						try:
							predicted_n.remove(n)
						except ValueError:
							pass
				
				# If we define y as 'forward' and 'back' then y=0 implies no forward neighbours
				if y == 0:
					for n in ['leftforward','rightforward','forward','upforward','upleftforward','uprightforward','downforward','downleftforward','downrightforward']:
						try:
							predicted_n.remove(n)
						except ValueError:
							pass
				
				# and if it is a maximum then we lose back neighbours
				if y == grid['Y']-1:
					for n in ['leftback','rightback','back','upback','upleftback','uprightback','downback','downleftback','downrightback']:
						try:
							predicted_n.remove(n)
						except ValueError:
							pass
				
				# If we define z as 'up' and 'down' then z=0 implies no down neighbours
				if z == 0:
					for n in ['down','downleft','downright','downforward','downleftforward','downrightforward','downback','downleftback','downrightback']:
						try:
							predicted_n.remove(n)
						except ValueError:
							pass
				
				# and if it is a maximum then we lose up neighbours
				if z == grid['Z']-1:
					for n in ['up','upleft','upright','upforward','upleftforward','uprightforward','upback','upleftback','uprightback']:
						try:
							predicted_n.remove(n)
						except ValueError:
							pass
				
				# Now make sure that we have the same number of non-zero neighbours
				if len(predicted_n) != len([n for n in cell['NEIGHBOURS'] if n is not None]):
					print str([x, y, z]) + ' has ' + str(len([n for n in cell['NEIGHBOURS'] if n is not None])) + ' neighbours, but predicted '+str(len(predicted_n)) 
					sys.exit()
				
				# If we didn't exit above then we have the right number of neighbours, so let's make sure they are the correct neighbours
				# Our neighbours are ordered as -z 0z +z..., -y -y -y 0y 0y 0y +y +y +y... and -x -x -x -x -x -x -x -x -x 0x 0x 0x 0x 0x 0x 0x 0x 0x +x +x +x +x +x +x +x +x +x, so make a list in this order of the above definitions
				all_n = ['downleftforward', 'leftforward', 'upleftforward', 'downleft', 'left', 'upleft', 'downleftback', 'leftback', 'upleftback', 'downforward', 'forward', 'upforward', 'down', 'self', 'up', 'downback', 'back', 'upback', 'downrightforward', 'rightforward', 'uprightforward', 'downright', 'right', 'upright', 'downrightback', 'rightback', 'uprightback']
				for index, n in enumerate(cell['NEIGHBOURS']):
					
					# If we have a neighbour we're not meant to then report it
					if all_n[index] not in predicted_n:
						if n is not None:
							print "Cell "+str(cell['ADDRESS'])+" has unexpected "+all_n[index]+" neighbour!"
							sys.exit()
					
					# If we don't have a neighbour we should have then report it
					else:
						if n is None:
							print "Cell "+str(cell['ADDRESS'])+" doesn't have expected "+all_n[index]+" neighbour!"
							sys.exit()
						
						# Otherwise make sure that it's the correct neighbour
						else:
							# Work out the  relative position of this neighbour
							z_rel = ((index % 9) % 3) - 1
							y_rel = (((index - z_rel) % 9)/3) - 1
							x_rel = (((index - z_rel) - (3*y_rel)) / 9) - 1
							
							# And from that the absolute position
							n_x = x + x_rel
							n_y = y + y_rel
							n_z = z + z_rel
							
							# Make sure the current neighbour points to the address of the neighbour
							if n != (((n_x*grid['Y']*grid['Z'])+(n_y*grid['Z'])+n_z)*cell_spacing)+grid['CELLSSTART']:
								print "Expected cell "+str(((n_x*(grid['Y']*grid['Z'])+n_y*(grid['Z'])+n_z)*cell_spacing)+grid['CELLSSTART'])+" but got neighbour "+str(n)
								
								nstart = (n - grid['CELLSSTART']) / cell_spacing
								nz = ((nstart % (grid['Y']*grid['Z'])) % grid['Z'])
								ny = (((nstart - nz) % (grid['Y']*grid['Z'])) / grid['Z'])
								nx = (((nstart - nz) - (grid['Z']*ny)) / (grid['Y']*grid['Z']))
								
								print "Cell is "+str(cell['ADDRESS'])+' ( '+str((x,y,z))+" ) and neighbour should be "+str((n_x,n_y,n_z))+" at relative position "+str((x_rel,y_rel,z_rel))+", but found "+str((nx,ny,nz))
								#sys.exit()
								
# If there are no cells then make sure there really aren't any
else:
	
	# Make sure that CELLSSTART doesn't appear without CELLS
	if 'CELLSSTART' in grid.keys():
		print "Grid has cell start address, but no cells!"
		sys.exit()
				
