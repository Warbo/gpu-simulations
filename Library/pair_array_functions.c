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
	if (x+1 == the_grid->x_size &&
			y+1 == the_grid->y_size &&
			z+1 == the_grid->z_size) {
		value = the_grid->particle_number +
			(the_grid->x_size*the_grid->y_size*the_grid->z_size) +
			the_grid->particles[index].id;
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

int check_dummies(grid* the_grid) {
	/*
	 * Goes through the dummy particles in the given grid and does some sanity
	 * checks.
	 */
	// PRECONDITIONS
	assert(the_grid != NULL);

	// Go through each dummy
	int last_value = the_grid->particles[0].id * -1;
	int index;
	for (index=0; index<the_grid->x_size * the_grid->y_size * the_grid->z_size;
		index++) {
		// Make sure it's a dummy
		assert(the_grid->particles[index].id < 0);
		// Make sure it's pointing after the last one
		assert((-1 * the_grid->particles[index].id) >= last_value);
		// Make sure it's pointing to somewhere inside the grid
		assert((-1 * the_grid->particles[index].id) >= (
			the_grid->x_size * the_grid->y_size * the_grid->z_size
		));
		assert((-1 * the_grid->particles[index].id) < (
			the_grid->x_size * the_grid->y_size * the_grid->z_size +
				the_grid->particle_number
		));
		// Increment the previous dummy
		last_value = -1 * the_grid->particles[index].id;
	}
	// Now go through real particles
	for (index=the_grid->x_size * the_grid->y_size * the_grid->z_size;
		index < the_grid->x_size * the_grid->y_size * the_grid->z_size +
			the_grid->particle_number; index++) {
		// Make sure they're not dummies
		assert(the_grid->particles[index].id >= 0);
	}
	return 0;	
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

	// Get the size of each cell
	int* cell_sizes;
	count_particles_in_cells(the_grid, particles, &cell_sizes);

	// Fill the grid with relevant dummies
	int running_total = the_grid->x_size*the_grid->y_size*the_grid->z_size;
	int index;
	for (index=0; index<the_grid->x_size*the_grid->y_size*the_grid->z_size;
		index++) {
		the_grid->particles[index].id = -1 * running_total;
		running_total = running_total + cell_sizes[index];

		// Reuse the cell_sizes array for offsets
		cell_sizes[index] = 0;
	}
	
	int px;	// Particle's cell position
	int py;	// ditto
	int pz;	// ditto
	int lookup; // The offset of a cell in the array
	
	// Loop through every given particle
	for (index = 0; index < the_grid->particle_number; index++) {

		// Get this particle's cell
		get_index_from_position(the_grid, &(particles[index]),
			&px, &py, &pz);
					
		// DEBUG
		assert(px >= 0);
		assert(py >= 0);
		assert(pz >= 0);

		lookup = pz + (the_grid->z_size * py) +
				(the_grid->z_size * the_grid->y_size * px);

		the_grid->particles[
			(-1 * the_grid->particles[lookup].id) + cell_sizes[lookup]
		] = particles[index];

		cell_sizes[lookup]++;
		
	}

	// Clean up
	free(cell_sizes);
	
	// POSTCONDITIONS
	assert(check_dummies(the_grid) == 0);

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
	*start = &(the_grid->particles[-1 * the_grid->particles[i].id]);

	// Now work out how many there are
	*length = count_cell_contents(the_grid, x, y, z);

	// POSTCONDITIONS
	assert(*length >= 0);
	assert(*start != NULL);

}

void make_padded_array(grid* the_grid, particle* in_array, particle** out_array,
	int* length, int cellsize) {
	/*
	 * Assigns a new array of particles at out_array, then takes the particles
	 * from the_grid and inserts them into this new array, leaving space between
	 * each cell such that each is cellsize particles.
	 */

	// PRECONDITIONS
	assert(the_grid != NULL);
	assert(out_array != NULL);
	assert(length != NULL);
	assert(cellsize >= get_biggest_cell_size(the_grid, in_array));
	
	// Allocate the array
	*out_array = (particle*) malloc((unsigned int)(
		(the_grid->x_size * the_grid->y_size * the_grid->z_size) * cellsize
	) * sizeof(particle));

	// We know the length (it's the cell count * cell size)
	*length = (the_grid->x_size * the_grid->y_size * the_grid->z_size) *
		cellsize;

	// Get each cell's size, so we know when to start adding dummies
	int* cell_sizes;
	count_particles_in_cells(the_grid, in_array, &cell_sizes);

	// Now loop through every cell
	int particle_index;
	int cell_index;
	for (cell_index=0; cell_index < (
		the_grid->x_size * the_grid->y_size * the_grid->z_size
	); cell_index++) {
		// For each cell, loop through "cellsize" particles
		for (particle_index=0; particle_index < cellsize; particle_index++) {

			// Only add particles if they're not padding
			if (particle_index < cell_sizes[cell_index]) {
				// Grab the next particle from the array
				out_array[0][(cell_index * cellsize)+particle_index] =
					the_grid->particles[-1*(the_grid->particles[cell_index].id)+
						particle_index];
			}
			else {
				// Otherwise add a dummy for padding
				out_array[0][(cell_index * cellsize)+particle_index].id = -1;
			}
		}
	}

	// Garbage collection
	free(cell_sizes);

	// POSTCONDITIONS
	assert(out_array[0] != NULL);
	assert(*length > 0);
}
