/*
 * Contains all code specific to the use of an array of cell's start locations.
 * The structure of the grid's particle storage in this case is:
 * S(0,0,0), S(0,0,1) ... (0,0,0), (0,0,1)...
 * Where S(x,y,z) is a particle storing the starting index of the cell (x,y,z).
 * This is stored as the ID, but negated. Thus an ID -10 means this cell starts
 * at particles[10]
 * (x,y,z) represents the particles in cell (x,y,z) in arbitrary order.
 *
 * This guarantees no wasted storage for nonexistant particles, however the use
 * of particle structs to store ints is a bit of overhead. It does mean that we
 * can keep the same grid struct.
 * Care must be taken to keep the start locations in sync with the rest of the
 * array, but it prevents the need to loop through the array over and over which
 * happens for the sentinel implementation.
 */

// Grab everything that's not specific to sentinels
#include "common_functions.c"

// Prototype our functions first
void get_cell_contents(grid*, int, int, int, particle**, int*);
int count_cell_contents(grid*, int, int, int);
void initialise_grid(grid*, int, int, int, float, float, float,
	float, float, float, int);
void grid_particles(grid*, particle*, int, float);

// Now define them

int count_cell_contents(grid* the_grid, int x, int y, int z) {
	/*
	* This returns the number of particles in the cell at the
	* given location in the given grid.
	*/

	// PRECONDITIONS
	assert(the_grid != NULL);
	assert(x >= 0);
	assert(y >= 0);
	assert(z >= 0);
	assert(x < the_grid->x_size);
	assert(y < the_grid->y_size);
	assert(z < the_grid->z_size);

	// We simply need the difference between the cell's start location and that
	// of the next cell in storage.
	
	// First, work out where this cell's start index can be found
	int index = z + (the_grid->z_size * y) +
		(the_grid->z_size * the_grid->y_size * x);
	int value;
	// Slight complication: If we're the last cell then there is no next
	// cell to compare with. Use the size of the array instead.
	if (x-1 == the_grid->x_size &&
		   y-1 == the_grid->y_size &&
		   z-1 == the_grid->z_size) {
		value = the_grid->particle_number +
			(the_grid->x_size*the_grid->y_size*the_grid->z_size) - index;
	}
	else {
		value = (-1 * the_grid->particles[index+1].id) -
			(-1 * the_grid->particles[index].id);
	}
	return value;
}

void initialise_grid(grid* the_grid, int x, int y, int z,
					 float dx, float dy, float dz,
	  float x_offset, float y_offset, float z_offset,
   int particles) {
	/*
	 * Set up a grid at the given pointer, assigning it memory for its
	 * cells and its particles.
	 */
	
	// PRECONDITIONS
	assert(x > 0);
	assert(y > 0);
	assert(z > 0);
	assert(dx > 0);
	assert(dy > 0);
	assert(dz > 0);
	assert(particles > 0);
	
	// Store the given values in the_grid
	the_grid->x_size = x;
	the_grid->y_size = y;
	the_grid->z_size = z;
	the_grid->dx = dx;
	the_grid->dy = dy;
	the_grid->dz = dz;
	the_grid->x_offset = x_offset;
	the_grid->y_offset = y_offset;
	the_grid->z_offset = z_offset;
	the_grid->particle_number = particles;
						
	// Allocate space for the particles and sentinels
	the_grid->particles =
		(particle*) malloc( ((unsigned int)(particles+x*y*z)) *
			sizeof(particle));
	
	// POSTCONDITIONS
	assert(the_grid->particles != NULL);
	assert(the_grid->x_size > 0);
	assert(the_grid->y_size > 0);
	assert(the_grid->z_size > 0);
	assert(the_grid->dx > 0);
	assert(the_grid->dy > 0);
	assert(the_grid->dz > 0);
	assert(the_grid->particle_number > 0);
	
}

void grid_particles(grid* the_grid, particle* particles, int particle_number,
	float size) {
	/*
	 * Sets up a grid at 'the_grid' and puts the contents of 'particles', of
	 * length 'particle_number', into it, each with size 'size'.
	 */
	
	// PRECONDITIONS
	assert(the_grid != NULL);
	assert(particles != NULL);
	assert(particle_number > 0);
	assert(size > (float)0.0);

	// Get grid parameters needed to store these particles
	float x_min, x_max, y_min, y_max, z_min, z_max;
	float x_offset, y_offset, z_offset;
	int x_size, y_size, z_size;
	get_extents(particles, particle_number, &x_min, &x_max, &y_min, &y_max,
		&z_min, &z_max);
	grid_size(x_min, x_max, y_min, y_max, z_min, z_max, size, size, size,
		&x_size, &y_size, &z_size, &x_offset, &y_offset, &z_offset);
	
	// DEBUG
	assert(x_size > 0);
	assert(y_size > 0);
	assert(z_size > 0);
	assert(x_min - x_offset > (float)0.0);
	assert(x_max - x_offset > (float)0.0);
	assert(y_min - y_offset > (float)0.0);
	assert(y_max - y_offset > (float)0.0);
	assert(z_min - z_offset > (float)0.0);
	assert(z_max - z_offset > (float)0.0);
	
	// Make the grid
	initialise_grid(the_grid, x_size, y_size, z_size, size, size, size,
		x_offset, y_offset, z_offset, particle_number);
	
	// DEBUG
	assert(the_grid->x_size == x_size);

	int* cell_sizes;
	count_particles_in_cells(the_grid, particles, &cell_sizes);

	int running_total = the_grid->x_size*the_grid->y_size*the_grid->z_size;
	int index;
	for (index=0; index<the_grid->x_size*the_grid->y_size*the_grid->z_size;
		index++) {
		the_grid->particles[index].id = -1 * running_total;
		running_total = running_total + cell_sizes[index];
	}

	// Stores our particle array position (particles start after dummies)
	int array_index = the_grid->x_size * the_grid->y_size * the_grid->z_size;
	int x;		// Used to loop through cell x coordinates
	int y;		// Ditto for y
	int z;		// Ditto for z
	int px;	// Particle's cell position
	int py;	// ditto
	int pz;	// ditto
	
	// Go through every cell location
	for (x = 0; x < the_grid->x_size; x++) {
		for (y = 0; y < the_grid->y_size; y++) {
			for (z = 0; z < the_grid->z_size; z++) {

				// To begin with, save our starting position
				the_grid->particles[
					z + (the_grid->z_size * y) +
					(the_grid->z_size * the_grid->y_size * x)].id =
						-1 * array_index;

				// For each cell, loop through every given particle
				for (index = 0; index < the_grid->particle_number; index++) {
					// Initialise particle's position to a nonsense value
					px = -1;
					py = -1;
					pz = -1;

					// If the particle is in this cell, add it to the grid
					get_index_from_position(the_grid, &(particles[index]),
						&px, &py, &pz);
					
					// DEBUG
					assert(px >= 0);
					assert(py >= 0);
					assert(pz >= 0);
					
					if (x == px && y == py && z == pz) {
						
						// DEBUG
						assert(array_index < the_grid->particle_number +
							(the_grid->x_size * the_grid->y_size *
								the_grid->z_size));
						
						the_grid->particles[array_index] = particles[index];
						
						array_index++;
					}
				}
			
			}
		}
	}
	
	// POSTCONDITIONS

}

void get_cell_contents(grid* the_grid, int x, int y, int z,
	particle** start, int* length) {
	/*
	 * Assigns the pointer 'start' to an array of particles, the size of
	 * which is stored in 'length'. The contents of the array are those
	 * particles in 'the_grid' at the grid cell with coordinates (x,y,z)
	 */
	// PRECONDITIONS
	assert(the_grid != NULL);
	assert(x >= 0);
	assert(y >= 0);
	assert(z >= 0);
	assert(x < the_grid->x_size);
	assert(y < the_grid->y_size);
	assert(z < the_grid->z_size);
	assert(length != NULL);
	assert(start != NULL);

	// Our location in the dummies
	int i;
	i = z + (the_grid->z_size * y) + (the_grid->z_size * the_grid->y_size * x);

	// Now we know where to start from, so set it
	*start = &(the_grid->particles[i]);

	// Now work out how many there are
	*length = count_cell_contents(the_grid, x, y, z);

	// POSTCONDITIONS
	assert(*length >= 0);
	assert(*start != NULL);

}

