#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// This is to aid in debugging
#include <assert.h>

// Define our data structures first of all
typedef struct particle particle;
typedef struct cell cell;
typedef struct grid grid;

struct particle {
	/*
	 * This particle implementation is for illustrative purposes only.
	 * It can easily be replaced with an arbitrary datastructure.
	 */
	int id;			// Useful when its memory location isn't constant
	float x;		// x position
	float y;		// y position
	float z;		// z position
	float x_vel;	// x velocity
	float y_vel;	// y velocity
	float z_vel;	// z velocity
	float x_acc;	// x acceleration
	float y_acc;	// y acceleration
	float z_acc;	// z acceleration
	float mass;		// The particle's mass
};

struct grid {
	/*
	 * A grid represents the space being simulated. It stores particles
	 * and cells, which point to the particles.
	 */
	
	// The dimensions of the space, as multiples of the cell dimensions,
	// are stored here
	int x_size;
	int y_size;
	int z_size;
	
	// These store the dimensions of the cells
	int dx;
	int dy;
	int dz;
	
	// The location of the first cell (x minimum, y minimum, z minimum)
	float x_offset;
	float y_offset;
	float z_offset;
	
	// This is the total number of particles in this space
	int particle_number;
	
	// This stores the particles which the cells contain
	particle* particles;
	
};

// Tests useful for assertions

int get_sentinel_number(grid* the_grid) {
	/*
	 * Counts the number of sentinel particles in the given grid.
	 */
	
	// Preconditions
	assert(the_grid != NULL);
	assert(the_grid->particle_number >= 0);
	assert(the_grid->particles != NULL);
	
	int sentinel_number = 0;
	int index;
	for (index=0; index < the_grid->particle_number; index++) {
		if (the_grid->particles[index].id == -1) {
			sentinel_count++;
		}
	}
	
	// Postconditions
	assert(sentinel_count >= 0);
	
	return sentinel_count;
}

float get_distance(particle* p1, particle* p2) {
	double result = 0;
	result = result + (double) ((p2->x - p1->x)*(p2->x - p1->x));
	result = result + (double) ((p2->y - p1->y)*(p2->y - p1->y));
	result = result + (double) ((p2->z - p1->z)*(p2->z - p1->z));
	result = sqrt(result);
	return (float) result;
}

// FIXME: There should be no tests in the core library, so move this
// after debugging!

int count_cell_contents(grid* the_grid, int x, int y, int z) {
	/*
	 * This returns the number of particles in the cell at the
	 * given location.
	 */
	
	// Loop through the grid's particles, counting those between the sentinels
	// we care about.
	int start_sentinel = (the_grid->y_size * the_grid->z_size)*x +
			(the_grid->z_size)*y + z;
	int sentinel_count = 0;
	int particle_count = 0;
	int index = 0;

	// Keep going until we go out the end of our cell, or run out of particles
	while (sentinel_count < start_sentinel+1 &&
		index < the_grid->particle_number +
			  (the_grid->x_size*the_grid->y_size*the_grid->z_size)) {

		// See if we've found another sentinel
		if (the_grid->particles[index].id == -1) {
			sentinel_count++;
		}

		// Otherwise count the particle if we're in the right cell
		else if (sentinel_count == start_sentinel) {
			particle_count++;
		}

		// Go to the next particle
		index++;
	}
		
	return particle_count;
}

int get_index(float position, float interval_size) {
	/*
	 * This does an "integer division" for doubles. It is effectively
	 * the same as (A-(A%B))/B, but % doesn't accept doubles and C
	 * operators cannot be overloaded since we're treated as second-
	 * class citizens compared to the ANSI standards body :(
	 */
	
	// PRECONDITIONS
	assert(position >= 0);
	assert(interval_size > 0);
	
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
	assert(count*interval_size < position);
	assert((count+1)*interval_size >= position);
	
	return count;
}

void get_index_from_position(grid* the_grid, particle* the_particle,
								int* x, int* y, int* z) {
	/*
	 * This function works out the given particle's cell location in
	 * the given grid based on its position in space, vital for finding
	 * neighbouring cells.
	 * This is used for benchmarking methods for calculating the index.
	 */
	
	*x = get_index(the_particle->x - the_grid->x_offset, the_grid->dx);
	*y = get_index(the_particle->y - the_grid->y_offset, the_grid->dy);
	*z = get_index(the_particle->z - the_grid->z_offset, the_grid->dz);

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
		(particle*)malloc((particles+x*y*z-1)*sizeof(particle));
	
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

void grid_particles(grid* the_grid, particle* particles) {
	/*
	 * Goes through every cell in the given grid and puts its particles in
	 * the right part of the array.
	 */
	// PRECONDITIONS
	assert(the_grid != NULL);
	assert(particles != NULL);

	int array_index = 0;		// Stores our particle array position
	int index;		// Used to loop through particles
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

				// For each cell, loop through every given particle
				for (index = 0; index < the_grid->particle_number; index++) {
					// Initialise particle's position to a nonsense value
					px = -1;
					py = -1;
					pz = -1;
					
					// If the particle is in this cell, add it to the grid
					get_index_from_position(the_grid, &(particles[index]),
						&px, &py, &pz);
					if (x == px && y == py && z == pz) {
						the_grid->particles[array_index] = particles[index];
						array_index++;
					}
				}

				// Now we've exhausted our particles, add a sentinel
				// (as long as we're not the last cell
				if (!((x == the_grid->x_size - 1) && (y == the_grid->y_size - 1)
					&& (z == the_grid->z_size - 1))) {
					the_grid->particles[array_index].id = -1;
					array_index++;
				}
			}
		}
	}
	
	// POSTCONDITIONS
	assert(get_sentinel_number(the_grid) == 
		the_grid->x_size * the_grid->y_size * the_grid->zsize - 1);
		
}

void get_cell_contents(grid* the_grid, int x, int y, int z, particle** start,
	int* length) {

	int index = 0;		// Stores where we're up to in the particle array
	int sentinel_count = 0;
	int start_sentinel = (the_grid->y_size * the_grid->z_size)*x +
		(the_grid->z_size)*y + z;

	// Initialise our lenght counter
	*length = 0;

	// Loop through the particles
	while ((index < the_grid->particle_number +
		(the_grid->x_size * the_grid->y_size * the_grid->z_size) - 1) &&
		(sentinel_count < start_sentinel + 1)) {

		// If we've found a sentinel, increment our sentinel counter
		if (the_grid->particles[index].id == -1) {
			sentinel_count++;

			// If we;ve just found the right cell, set the start to the next
			// particle
			if (sentinel_count == start_sentinel) {
				*start = &(the_grid->particles[index+1]);
 			}
		}

		// Otherwise, if we're in the right cell, increment the particle counter
 		else if (sentinel_count == start_sentinel) {
			*length = *length + 1;
		}

		// Move on to the next particle
		index++;
	}
	fprintf(stderr, "A %i\n", index);
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
	assert(the_particle != NULL);
	assert(neighbour_particles != NULL);
	assert(length != NULL);
	
	// Get the given particle's cell location
	int x;
	int y;
	int z;
	get_index_from_position(the_grid, the_particle, &x, &y, &z);

	// Initialise the length counter
	*length = 0;
	
	// x, y and z offsets of the neighbour from the particle's cell
	int x_rel;
	int y_rel;
	int z_rel;

	// Needed for extracting the particles from the grid
	particle* dummy_array;
	int particle_count;

	// Loop through neighbours
	for (x_rel = -1; x_rel < 2; x_rel++) {
		for (y_rel = -1; y_rel < 2; y_rel++) {
			for (z_rel = -1; z_rel < 2; z_rel++) {
				// Only act if this enighbour exists
				if ((x+x_rel > 0) && (x+x_rel < the_grid->x_size) &&
					(y+y_rel > 0) && (y+y_rel < the_grid->y_size) &&
					(z+z_rel > 0) && (z+z_rel < the_grid->z_size)) {

					// Get the number of particles in this cell
					get_cell_contents(the_grid, x+x_rel, y+y_rel, z+z_rel,
						&(dummy_array), &particle_count);

					// Add it to the number of neighbours
					*length = *length + particle_count;
					
					}
			}
		}
	}

	fprintf(stderr, "B %i\n", *length);
	
	// Now we know how many neighbour particles we have we can make an array to
	// store them
	*neighbour_particles = (particle*) malloc(*length * sizeof(particle));

	// Now we loop again to populate the array
	int index = 0;		// Position in the array
	int index2;		// Looping index
	for (x_rel = -1; x_rel < 2; x_rel++) {
		for (y_rel = -1; y_rel < 2; y_rel++) {
			for (z_rel = -1; z_rel < 2; z_rel++) {
				// Only act if this enighbour exists
				if ((x+x_rel > 0) && (x+x_rel < the_grid->x_size) &&
					(x+x_rel > 0) && (x+x_rel < the_grid->x_size) &&
					 (x+x_rel > 0) && (x+x_rel < the_grid->x_size)) {

					// Get the particles in this cell
					get_cell_contents(the_grid, x+x_rel, y+y_rel, z+z_rel,
						&(dummy_array), &particle_count);

					// Add them to the array
					for (index2 = 0; index2 < particle_count; index2++) {
						neighbour_particles[0][index] = dummy_array[index2];
						index++;
					}
				}
			}
		}
	}
		
}

void get_true_neighbours_for_particle(grid* the_grid, particle* the_particle,
	particle** neighbour_particles, int* length) {

	// Initialise the length counter
	*length = 0;
		
	// Get the potential neighbours first
	particle* potential_neighbours;
	int potential_count;

	get_potential_neighbours_for_particle(the_grid, the_particle,
		&(potential_neighbours), &potential_count);

	// Now see how many are real neighbours
	int index;
	for (index = 0; index < potential_count; index++) {
		if (get_distance(&(potential_neighbours[index]), the_particle) <=
			the_grid->dx) {
			*length = *length + 1;
		}
	}

	// Now we know how many true neighbours we've got, so allocate memory
	*neighbour_particles = (particle*) malloc(*length * sizeof(particle));

	// Now populate the array
	int position = 0;
	for (index = 0; index < potential_count; index++) {
		if ((!(potential_neighbours[index].id == -1)) &&
			(get_distance(&(potential_neighbours[index]), the_particle) <=
				the_grid->dx)) {
			neighbour_particles[0][position] = potential_neighbours[index];
			position++;
		}
	}
}

void get_neighbours_brute_force(grid* the_grid, particle* the_particle,
	particle** neighbour_particles, int* length) {

	// Initialise the length counter
	 *length = 0;
	
	int all_count = the_grid->particle_number +
		(the_grid->x_size * the_grid->y_size * the_grid->z_size) - 1;

	// Now see how many are real neighbours
	int index;
	for (index = 0; index < all_count; index++) {
		if ((!(the_grid->particles[index].id == -1)) &&
			(get_distance(&(the_grid->particles[index]), the_particle) <=
			the_grid->dx)) {
			*length = *length + 1;
		}
	}

	// Now we know how many true neighbours we've got, so allocate memory
	*neighbour_particles = (particle*) malloc(*length * sizeof(particle));

	// Now populate the array
	int position = 0;
	for (index = 0; index < all_count; index++) {
		if ((!(the_grid->particles[index].id == -1)) &&
			(get_distance(&(the_grid->particles[index]), the_particle) <=
				the_grid->dx)) {
			neighbour_particles[0][position] = the_grid->particles[index];
			position++;
		}
	}
}
