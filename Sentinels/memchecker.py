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

grid_contents ::= <x_size> | <y_size> | <z_size> | <population> | <cells> | <particles> | <dx> | <dy> | <dz> | <x_offset> | <y_offset> | <z_offset>

x_size ::= <token "X:"> <integer>:size ','*									=> {'X':size}

y_size ::= <token "Y:"> <integer>:size ','*									=> {'Y':size}

z_size ::= <token "Z:"> <integer>:size ','*									=> {'Z':size}

x_pos ::= <token "X:"> <decimal>:pos ','*									=> {'X':pos}

y_pos ::= <token "Y:"> <decimal>:pos ','*									=> {'Y':pos}

z_pos ::= <token "Z:"> <decimal>:pos ','*									=> {'Z':pos}

dx ::= <token "DX:"> <decimal>:val ','*										=> {'DX':val}

dy ::= <token "DY:"> <decimal>:val ','*										=> {'DY':val}

dz ::= <token "DZ:"> <decimal>:val ','*										=> {'DZ':val}

x_offset ::= <token "OX:"> <decimal>:val ','*								=> {'OX':val}

y_offset ::= <token "OY:"> <decimal>:val ','*								=> {'OY':val}

z_offset ::= <token "OZ:"> <decimal>:val ','*								=> {'OZ':val}

population ::= <token "POPULATION:"> <integer>:pop ','*						=> {'POPULATION':pop}

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
print "Making grammar"
matcher = grammar(inlines)

# Try to find a dump in the input
print "Reading contents"
grid = matcher.apply('grid')

## Now we should have a dictionary of the program internals, so run some
## tests

print "Testing"

if 'DX' not in grid.keys() or 'DY' not in grid.keys() or 'DZ' not in grid.keys():
	print "dx/y/z not found!"
	sys.exit()

if grid['DX'] != grid['DY'] or grid['DX'] != grid['DZ'] or grid['DY'] != grid['DZ']:
	print "WARNING: dx, dy and dz aren't the same value."

# If there are cells, do checks on them
if 'CELLS' not in grid.keys():
	print "No cells found!"
	sys.exit(1)
	
# Make sure there's a CELLSTART
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
		sys.exit(1)
		
	# See if this cell has a first particleDo checks on this cell's first particle, if it has one
	if 'FIRSTPARTICLE' not in cell.keys():
		print "Cell "+str(cell['ADDRESS'])+" has no first particle set, not even NULL!"
		sys.exit(1)
	
	# Check the cell's particle if it has one
	if cell['FIRSTPARTICLE'] is not None:
			
		# Make sure that the first particle does not come before the particles' start address
		if cell['FIRSTPARTICLE'] < grid['PARTICLESSTART']:
			print "Cell contains particle "+cell['FIRSTPARTICLE']+' but particles start at '+grid['PARTICLESSTART']
			sys.exit(1)
			
		# Make sure that the first particle is in the list of particles
		if cell['FIRSTPARTICLE'] not in [p['ADDRESS'] for p in grid['PARTICLES']]:
			print "Cell contains particle "+cell['FIRSTPARTICLE']+" but not in particles list "+str([p['ADDRESS'] for p in grid['PARTICLES']])
			sys.exit(1)
		
	# If we've no particles then do some further checks
	else:
		# If there are particles in the grid then make sure none of them gives this cell as its container (since the cell thinks it's empty)
		for particle in grid['PARTICLES']:
			if particle['CONTAINER'] == cell['ADDRESS']:
				print "Cell "+str(cell['ADDRESS'])+" says it has no particles, but particle "+str(particle['ADDRESS'])+" says it is in this cell!"
				sys.exit(1)
		
	if cell_spacing is not None:
		# Find the position of this cell
		# Our indexing scheme for the cells should be (y*z)*x_index+(z)*y_index+z_index, so let's invert that
		start = (cell['ADDRESS'] - grid['CELLSSTART']) / cell_spacing
		cell['Z'] = ((start % (grid['Y']*grid['Z'])) % grid['Z'])
		cell['Y'] = (((start - cell['Z']) % (grid['Y']*grid['Z'])) / grid['Z'])
		cell['X'] = (((start - cell['Z']) - (grid['Z']*cell['Y'])) / (grid['Y']*grid['Z']))

		# Make sure we're in the grid
		if cell['X'] < 0 or cell['X'] >= grid['X'] or cell['Y'] < 0 or cell['Y'] >= grid['Y'] or cell['Z'] < 0 or cell['Z'] >= grid['Z']:
			print "Cell "+str(cell['ADDRESS'])+" is at position "+str((cell['X'],cell['Y'],cell['Z']))+" but grid size is "+str((grid['X'],grid['Y'],grid['Z']))
			sys.exit(1)
								
# Now run tests on the particles
if 'PARTICLES' not in grid.keys():
	print "Could not find any particles!"
	sys.exit(1)
	
# Make sure that we have a start address if we do have particles
if 'PARTICLESSTART' not in grid.keys():
	print "We have particles, but no start address"
	sys.exit(1)

# Gets the size of particles in memory if there's more than one	
if len(grid['PARTICLES']) > 1:
	particle_length = grid['PARTICLES'][1]['ADDRESS'] - \
		grid['PARTICLES'][0]['ADDRESS']
else:
	particle_length = None

# Traverse the particles
for p in grid['PARTICLES']:

	# Make sure this particle is not stored before the
	# particles array
	if p['ADDRESS'] < grid['PARTICLESSTART']:
		print "Particle found at "+str(p['ADDRESS'])+\
			" but particles start at "+str(grid['PARTICLESSTART'])
		sys.exit(1)

	# Make sure this particle is an integer number of 
	# particle_lengths away from PARTICLESSTART
	if particle_length is not None:
		if (p['ADDRESS'] - grid['PARTICLESSTART']) \
			% particle_length != 0:
			print "Particle not an integer number of particle"+\
				" lengths away from start!"
			sys.exit(1)
		
	# Make sure this particle has a container
	if 'CONTAINER' not in p.keys():
		print "Particle has no container!"
		sys.exit(1)
		
	# Make sure this particle's container is a valid cell
	valid_container = False
	current_container = None;
	for cell in grid['CELLS']:
		if cell['ADDRESS'] == p['CONTAINER']:
			valid_container = True
			current_container = cell
	if not valid_container:
		print "Particle's container is not a valid cell address!"
		sys.exit(1)
		
	# Make sure that this particle's container is the correct one
	x = (p['X'] - grid['OX']) % grid['DX']
	y = (p['Y'] - grid['OY']) % grid['DY']
	z = (p['Z'] - grid['OZ']) % grid['DZ']
	
	if current_container['X'] != x:
		print "Particle is at X "+str(x)+" but is in cell "+str(current_container['X'])
		sys.exit(1)
		
	if current_container['Y'] != y:
		print "Particle is at Y "+str(y)+" but is in cell "+str(current_container['Y'])
		sys.exit(1)
		
	if current_container['Z'] != z:
		print "Particle is at Z"+str(z)+" but is in cell "+str(current_container['Z'])
		sys.exit(1)

