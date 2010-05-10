#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// This is to aid in debugging
#include <assert.h>

// Read particles from stdin
#include "file_reading.c"

// Externally defined functions
int count_cell_contents(grid*, int, int, int);
void get_cell_contents(grid*, int, int, int, particle**, int*);
void grid_particles(grid*, particle*, int, float);
void initialise_grid(grid*, int, int, int, float, float, float,
	float, float, float, int);

// Tests useful for assertions
float get_distance(particle* p1, particle* p2) {
	/*
	 * Returns the scalar distance between two particles.
	 */

	// PRECONDITIONS
	
	double result = 0;
	result = result + (double) ((p2->x - p1->x)*(p2->x - p1->x));
	result = result + (double) ((p2->y - p1->y)*(p2->y - p1->y));
	result = result + (double) ((p2->z - p1->z)*(p2->z - p1->z));
	result = sqrt(result);
	
	// POSTCONDITIONS
	
	return (float) result;
}

int get_index(float position, float interval_size) {
	/*
	 * This does an "integer division" for doubles. It is effectively
	 * the same as (A-(A%B))/B, but % doesn't accept doubles and C
	 * operators cannot be overloaded since we're treated as second-
	 * class citizens compared to the ANSI standards body :(
	 */
	
	// PRECONDITIONS
	assert(position >= 0.0);
	assert(interval_size > 0.0);
	
	// If the given position fits inside one interval then return 0
	if (position < interval_size) {
		return 0;
	}
	
	// If not then we'll need to count the intervals
	int count = 0;
	float remaining = position;
	while (remaining > interval_size) {
		count++;
		remaining -= interval_size;
	}
	
	// POSTCONDITIONS
	assert( ((float)count) * interval_size < position);
	assert( ((float)(count+1)) * interval_size >= position);
	
	return count;
}

void get_extents(particle* p_array, int length, float* x_min, float* x_max,
	float* y_min, float* y_max, float* z_min, float* z_max) {
	/*
	 * Given an array of particles 'p_array' of size 'length', set the minimum
	 * and maximum values of x, y and z to the appropriate arguments.
	 */

	// PRECONDITIONS
	assert(p_array != NULL);
	assert(length > 0);
	assert(x_min != NULL);
	assert(x_max != NULL);
	assert(y_min != NULL);
	assert(y_max != NULL);
	assert(z_min != NULL);
	assert(z_max != NULL);

	// Initialise the values based on the first available particle
	int first = 0;
	int initialised = 0;

	// Get a non-sentinel particle
	while (initialised == 0 && first < length) {
		if (p_array[first].id >= 0) {
			*x_min = p_array[first].x;
			*x_max = p_array[first].x;
			*y_min = p_array[first].y;
			*y_max = p_array[first].y;
			*z_min = p_array[first].z;
			*z_max = p_array[first].z;
			initialised = 1;
		}
		first++;
	}

	// DEBUG
	assert(initialised == 1);

	// Now loop through them all until we establish the true maxes and mins
	for (first = 0; first < length; first++) {
		if (p_array[first].id >= 0) {

			if (p_array[first].x < *x_min) {
				*x_min = p_array[first].x;
			}
			if (p_array[first].x > *x_max) {
				*x_max = p_array[first].x;
			}
			if (p_array[first].y < *y_min) {
				*y_min = p_array[first].y;
			}
			if (p_array[first].y > *y_max) {
				*y_max = p_array[first].y;
			}
			if (p_array[first].z < *z_min) {
				*z_min = p_array[first].z;
			}
			if (p_array[first].z > *z_max) {
				*z_max = p_array[first].z;
			}
		}
	}

	// POSTCONDITIONS
	assert(initialised == 1);
	assert(*x_min <= *x_max);
	assert(*y_min <= *y_max);
	assert(*z_min <= *z_max);

}

void grid_size(float x_min, float x_max, float y_min, float y_max, float z_min,
	float z_max, float dx, float dy, float dz, int* x, int* y, int* z,
	float* x_offset, float* y_offset, float* z_offset) {
	/*
	 * Given the ranges x_min to x_max, y_min to y_max and z_min to z_max, works
	 * out how many cells of size dx*dy*dz would be needed to contain all of
	 * these. These dimensions are stored in x, y and z, and the offset between
	 * such a grid's coordinate system and the particles' coordinate system is
	 * stored in x_offset, y_offset, z_offset.
	 */

	// PRECONDITIONS
	assert(x_min <= x_max);
	assert(y_min <= y_max);
	assert(z_min <= z_max);
	assert(dx > 0.0);
	assert(dy > 0.0);
	assert(dz > 0.0);
	assert(x != NULL);
	assert(y != NULL);
	assert(z != NULL);
	assert(x_offset != NULL);
	assert(y_offset != NULL);
	assert(z_offset != NULL);
	assert(dx == dy);
	assert(dx == dz);
	assert(dy == dz);

	// Get the sizes of each range
	float delta_x = x_max - x_min;
	float delta_y = y_max - y_min;
	float delta_z = z_max - z_min;

	// DEBUG
	assert(delta_x > (float)0.0);
	assert(delta_y > (float)0.0);
	assert(delta_z > (float)0.0);

	// See how many cells we can fit into each of these ranges
	*x = (int) (delta_x / dx);
	*y = (int) (delta_y / dy);
	*z = (int) (delta_z / dz);

	// DEBUG
	assert(*x >= 0);
	assert(*y >= 0);
	assert(*z >= 0);

	// These will round down in the general case, so add one to each
	// This can give us an overhead, but it only goes up as a square, whilst the
	// cell count goes up as a cube, so it's only significant for small grids
	// which are quick anyway.
	*x = *x + 1;
	*y = *y + 1;
	*z = *z + 1;

	// DEBUG
	assert( ((float)*x) * dx > delta_x);
	assert( ((float)*y) * dy > delta_y);
	assert( ((float)*z) * dz > delta_z);

	// We may as well centre the smaller range on the bigger, to avoid particles
	// directly on the boundaries
	float leftover_x = ((float)*x * dx) - delta_x;
	float leftover_y = ((float)*y * dy) - delta_y;
	float leftover_z = ((float)*z * dz) - delta_z;

	// DEBUG
	assert(leftover_x > (float)0.0);
	assert(leftover_y > (float)0.0);
	assert(leftover_z > (float)0.0);

	*x_offset = x_min - (leftover_x / ((float)2.0));
	*y_offset = y_min - (leftover_y / ((float)2.0));
	*z_offset = z_min - (leftover_z / ((float)2.0));

	// POSTCONDITIONS
	assert(x_min - *x_offset > (float)0.0);
	assert(x_max - *x_offset < dx * ((float)*x));
	assert(y_min - *y_offset > (float)0.0);
	assert(y_max - *y_offset < dy * ((float)*y));
	assert(z_min - *z_offset > (float)0.0);
	assert(z_max - *z_offset < dz * ((float)*z));
	
}

void get_index_from_position(grid* the_grid, particle* the_particle,
	int* x, int* y, int* z) {
	/*
	 * This function works out the given particle's cell location in
	 * the given grid based on its position in space, vital for finding
	 * neighbouring cells.
	 * This is used for benchmarking methods for calculating the index.
	 */
	
	// PRECONDITIONS
	assert(the_grid != NULL);
	assert(the_particle != NULL);
	assert(x != NULL);
	assert(y != NULL);
	assert(z != NULL);
	
	*x = get_index(the_particle->x - the_grid->x_offset, the_grid->dx);
	*y = get_index(the_particle->y - the_grid->y_offset, the_grid->dy);
	*z = get_index(the_particle->z - the_grid->z_offset, the_grid->dz);

	// POSTCONDITIONS
	assert(*x >= 0);
	assert(*y >= 0);
	assert(*z >= 0);
	assert(*x < the_grid->x_size);
	assert(*y < the_grid->y_size);
	assert(*z < the_grid->z_size);

}

void get_potential_neighbours_for_particle(grid* the_grid, 
	particle* the_particle, particle** neighbour_particles,
	int* length) {
	/*
	 * This function will find every neighbour particle for the given
	 * particle, checking its cell and its neighbouring cells.
	 * Pointers to these particles are stored at "neighbour_particles",
	 * which is allocated in this function. The amount of memory
	 * allocated will be stored in "length".
	 */
	
	// PRECONDITIONS
	assert(the_grid != NULL);
	assert(the_particle != NULL);
	assert(neighbour_particles != NULL);
	assert(length != NULL);
	assert(the_particle->id >= 0);
	
	// Get the given particle's cell location
	int x;
	int y;
	int z;
	get_index_from_position(the_grid, the_particle, &x, &y, &z);

	// DEBUG
	assert(x >= 0);
	assert(y >= 0);
	assert(z >= 0);
	assert(x < the_grid->x_size);
	assert(y < the_grid->y_size);
	assert(z < the_grid->z_size);

	// Initialise the length counter
	*length = 0;
	
	// x, y and z offsets of the neighbour from the particle's cell
	int x_rel;
	int y_rel;
	int z_rel;

	// Needed for extracting the particles from the grid
	particle* dummy_array;
	int particle_count = -1;

	// Loop through neighbours
	for (x_rel = -1; x_rel < 2; x_rel++) {
		for (y_rel = -1; y_rel < 2; y_rel++) {
			for (z_rel = -1; z_rel < 2; z_rel++) {
				// Only act if this neighbour exists
				if ((x+x_rel >= 0) && (x+x_rel < the_grid->x_size) &&
					(y+y_rel >= 0) && (y+y_rel < the_grid->y_size) &&
					(z+z_rel >= 0) && (z+z_rel < the_grid->z_size)) {

					// DEBUG
					assert(x+x_rel >= 0);
					assert(y+y_rel >= 0);
					assert(z+z_rel >= 0);
					assert(x+x_rel < the_grid->x_size);
					assert(y+y_rel < the_grid->y_size);
					assert(z+z_rel < the_grid->z_size);
					
					// Get the number of particles in this cell
					get_cell_contents(the_grid, x+x_rel, y+y_rel, z+z_rel,
						&(dummy_array), &particle_count);

					// DEBUG
					assert(particle_count >= 0);
					assert(dummy_array != NULL);

					// Add it to the number of neighbours
					*length = *length + particle_count;
					
					// DEBUG
					assert(*length >= 0);
					
				}
			}
		}
	}
	
	// DEBUG
	assert(neighbour_particles != NULL);
	
	// Now we know how many neighbour particles we have we can make an array to
	// store them
	*neighbour_particles = (particle*) malloc( ((unsigned int)*length) *
		sizeof(particle));
	
	// DEBUG
	assert(*neighbour_particles != NULL);

	// Now we loop again to populate the array
	int index = 0;		// Position in the array
	int index2;		// Looping index
	for (x_rel = -1; x_rel < 2; x_rel++) {
		for (y_rel = -1; y_rel < 2; y_rel++) {
			for (z_rel = -1; z_rel < 2; z_rel++) {
				
				// Only act if this enighbour exists
				if ((x+x_rel >= 0) && (x+x_rel < the_grid->x_size) &&
					(y+y_rel >= 0) && (y+y_rel < the_grid->y_size) &&
					(z+z_rel >= 0) && (z+z_rel < the_grid->z_size)) {

					// DEBUG
					assert(x+x_rel >= 0);
					assert(y+y_rel >= 0);
					assert(z+z_rel >= 0);
					assert(x+x_rel < the_grid->x_size);
					assert(y+y_rel < the_grid->y_size);
					assert(z+z_rel < the_grid->z_size);

					// Get the particles in this cell
					get_cell_contents(the_grid, x+x_rel, y+y_rel, z+z_rel,
						&(dummy_array), &particle_count);

					// DEBUG
					assert(particle_count >= 0);
					assert(dummy_array != NULL);

					// Add them to the array
					for (index2 = 0; index2 < particle_count; index2++) {
						
						// DEBUG
						assert(index < *length);
						
						neighbour_particles[0][index] = dummy_array[index2];
						index++;
						
						// DEBUG
						assert(index <= *length);
						
					}
				}
			}
		}
	}
		
	// POSTCONDITIONS
	assert(neighbour_particles != NULL);
	assert(*neighbour_particles != NULL);
	assert(length != NULL);
	assert(*length >= 0);
	assert(*length <= the_grid->particle_number);
	
}

void get_true_neighbours_for_particle(grid* the_grid,
	particle* the_particle, particle** neighbour_particles,
	int* length) {
	/*
	 * Allocates an array of particles inside 'neighbour_particles' with
	 * its size stored in 'length'. The array is populated by those
	 * particles in 'the_grid' which are within one grid cell width of
	 * 'the_particle'.
	 */
	
	// PRECONDITIONS
	assert(the_grid != NULL);
	assert(the_particle != NULL);
	assert(neighbour_particles != NULL);
	assert(length != NULL);
	
	// Initialise the length counter
	*length = 0;
		
	// Get the potential neighbours first
	particle* potential_neighbours;
	int potential_count;

	get_potential_neighbours_for_particle(the_grid, the_particle,
		&(potential_neighbours), &potential_count);

	// DEBUG
	assert(potential_count <= the_grid->particle_number);
	assert(potential_neighbours != NULL);
	assert(potential_count >= 0);
	
	// Now see how many are real neighbours
	int index;
	for (index = 0; index < potential_count; index++) {
		if ((get_distance(&(potential_neighbours[index]),
			the_particle) <= the_grid->dx) &&
			(the_particle->id != potential_neighbours[index].id)) {
			
			// DEBUG
			assert(potential_neighbours[index].id >= 0);
			assert(get_distance(&(potential_neighbours[index]),
				the_particle) <= the_grid->dy);
			assert(get_distance(&(potential_neighbours[index]),
				the_particle) <= the_grid->dz);
			
			*length = *length + 1;
			
			// DEBUG
			assert(*length >= 0);
			assert(*length < the_grid->particle_number);
			
		}
	}

	// DEBUG
	assert(neighbour_particles != NULL);

	// Now we know how many true neighbours we've got, so allocate
	// memory
	*neighbour_particles=(particle*) malloc( ((unsigned int)*length) *
		sizeof(particle));

	// DEBUG
	assert(*neighbour_particles != NULL);

	// Now populate the array
	int position = 0;
	for (index = 0; index < potential_count; index++) {
		if ((get_distance(&(potential_neighbours[index]),
			the_particle) <= the_grid->dx) &&
			(potential_neighbours[index].id != the_particle->id)) {
			
			// DEBUG
			assert(potential_neighbours[index].id >= 0);
			assert(get_distance(&(potential_neighbours[index]),
				the_particle) <= the_grid->dy);
			assert(get_distance(&(potential_neighbours[index]), the_particle) <=
				the_grid->dz);
			assert(position < *length || *length == 0);
			
			neighbour_particles[0][position] = potential_neighbours[index];
			position++;
			
			// DEBUG
			assert(position <= *length || *length == 0);
			
		}
	}
	
	// POSTCONDITIONS
	assert(*length >= 0);
	assert(neighbour_particles != NULL);
	assert(*neighbour_particles != NULL);
	
}

void get_neighbours_brute_force(grid* the_grid, particle* the_particle,
	particle** neighbour_particles, int* length) {
	/*
	 * Looks through every particle in 'the_grid' to see whether it
	 * is less than one grid cell width away from 'the_particle'.
	 * Allocates an array in 'neighbour_particles' and populates it with
	 * those within that distance. The size of this array is stored in
	 * 'length'.
	 */

	// PRECONDITIONS
	assert(the_grid != NULL);
	assert(the_grid->particles != NULL);
	assert(the_particle != NULL);
	assert(neighbour_particles != NULL);
	assert(length != NULL);
	
	// Initialise the length counter
	*length = 0;

	// TODO: Specific to sentinels!
	int all_count = the_grid->particle_number +
		(the_grid->x_size * the_grid->y_size * the_grid->z_size);
	
	// Now see how many are real neighbours
	int index;
	for (index = 0; index < all_count; index++) {
		if ((the_grid->particles[index].id != -1) &&
			(get_distance(&(the_grid->particles[index]), the_particle) <=
				the_grid->dx) &&
			(the_particle->id != the_grid->particles[index].id)) {
	
			// DEBUG
			assert(*length < the_grid->particle_number);
			assert(get_distance(&(the_grid->particles[index]),
				the_particle) <= the_grid->dy);
			assert(get_distance(&(the_grid->particles[index]),
				the_particle) <= the_grid->dz);
			assert(the_grid->particles[index].id >= 0);
			
			*length = *length + 1;
			
			// DEBUG
			assert(*length < the_grid->particle_number);
		
		}
	}

	// DEBUG
	assert(neighbour_particles != NULL);

	// Now we know how many true neighbours we've got, so allocate memory
	*neighbour_particles = (particle*) malloc( ((unsigned int)*length) *
		sizeof(particle));

	// DEBUG
	assert(*neighbour_particles != NULL);
	assert(all_count == the_grid->particle_number +
		(the_grid->x_size * the_grid->y_size * the_grid->z_size));
	
	// Now populate the array
	int position = 0;
	for (index = 0; index < all_count; index++) {
		if ((the_grid->particles[index].id != -1) &&
				   (get_distance(&(the_grid->particles[index]), the_particle) <=
				   the_grid->dx) &&
				   (the_particle->id != the_grid->particles[index].id)) {
	
			// DEBUG
			assert(*length < the_grid->particle_number);
			assert(position < *length);
			assert(get_distance(&(the_grid->particles[index]),
				   the_particle) <= the_grid->dy);
			assert(get_distance(&(the_grid->particles[index]),
				   the_particle) <= the_grid->dz);
			assert(the_grid->particles[index].id >= 0);
			
			neighbour_particles[0][position] = the_grid->particles[index];
			position++;
			
			// DEBUG
			assert(*length < the_grid->particle_number);
		
		}
	}
	
}

float new_random() {
	/*
	 * Returns a random double between 0 and 0.999...
	 * NOTE: You may wish to change the random seed first using srand()!
	 */
	 return ((float)rand()) / ((float)RAND_MAX);
}

void populate_random(particle** p_array, int particles, int x, int y, int z,
	float dx, float dy, float dz) {
	/*
	 * This populates the given array with particles, determining their
	 * positions using the standard rand() function.
	 */
	
	// PRECONDITIONS
	assert(particles > 0);
	assert(x > 0);
	assert(y > 0);
	assert(z > 0);
	assert(dx > 0);
	assert(dy > 0);
	assert(dz > 0);
	assert(p_array != NULL);
	
	// Store the total size of the space
	float x_space, y_space, z_space;
	x_space = ((float) x) * dx;
	y_space = ((float) y) * dy;
	z_space = ((float) z) * dz;
	
	// Loop through the particle memory
	int particle_index;
	for (particle_index = 0;
		particle_index < particles;
		particle_index++) {
		// Assign the position
		p_array[0][particle_index].x =
			new_random()*x_space;
		p_array[0][particle_index].y =
			new_random()*y_space;
		p_array[0][particle_index].z =
			new_random()*z_space;
	}
	
	// POSTCONDITIONS
	//assert(check_particles(the_grid, particles) == 0);
	
}

void setup_test() {
	// This is the size of our volume elements
	float particle_size = (float)1.0;
	
	// These are the dimensions of our space as multiples of particle_size
	int x = 50;
	int y = 20;
	int z = 10;
	
	// This is the total number of particles to simulate
	int particle_number = 3750;
	
	// Make the space we are simulating
	grid the_grid;

	// Make the particles we're testing with (could be read from a file)
	particle* p_array = (particle*) malloc( ((unsigned int)particle_number) *
		sizeof(particle));
	
	// Fill the array with random particles
	populate_random(&p_array, particle_number, x, y, z,
		particle_size, particle_size, particle_size);
	
	// Allocate memory and assign neighbourhoods
	grid_particles(&the_grid, p_array, particle_number, particle_size);
		
	// DEBUGGING
	assert(the_grid.particles != NULL);

}

