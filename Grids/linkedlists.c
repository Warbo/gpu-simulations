#include <stdio.h>
#include <stdlib.h>

// This is to aid in debugging
#include <assert.h>

// Define our data structures first of all
typedef struct particle particle;
typedef struct cell cell;
typedef struct grid grid;

typedef struct three_vector {
	/* 
	 * A very basic datatype, included for completeness. Final code
	 * should use optimised vector types from elsewhere (eg. Eigen2)
	 */
	double x;
	double y;
	double z;
} three_vector;

struct particle {
	/*
	 * This particle implementation is for illustrative purposes only.
	 * It can easily be replaced with an arbitrary datastructure.
	 */
	particle* next;
	cell* container;
	three_vector position;
	three_vector velocity;
	three_vector acceleration;
	double mass;
};

struct cell {
	/*
	 * A cell is a unit volume in the grid. It stores a pointer to a
	 * linked list of the particles it contains.
	 */
	cell** neighbours;
	
	// This points to the first particle in our linked list
	particle* first_particle;
	
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
	
	// This is the total number of particles in this space
	int particle_number;
	
	// This stores the volume units of this grid
	cell* cells;
	
	// This stores the particles which the cells point to
	particle* particles;
	
};

int get_index(double position, double interval_size) {
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
	double remaining = position;
	while (remaining > interval_size) {
		count++;
		remaining -= interval_size;
	}
	
	// POSTCONDITIONS
	assert(count*interval_size < position);
	assert((count+1)*interval_size >= position);
	
	return count;
}

/*int get_subscript_for_position(int x, int y, int z) {
	/*
	 * Given an x, y and z coordinate, calculate the index with which to
	 * subscript the cell array.
	 */
	
	/*// To get the address we must first find the layer it's in, which
	// depends on the magnitude of the coordinate abs(x)+abs(y)+abs(z)
	int start = 0;		// Indexing starts at zero
	int inc = 8;		// We initially go up by 8
	int inc_inc = 16;		// We initially increase the increment by 16
	count = 0;
	// Loop as long as we haven't found the right layer yet
	while ((abs(x)+abs(y)+abs(z))-count > 0) {
		inc_inc += 8;		// The increment's increase goes up by 8
		inc += inc_inc;		// Increase the increment
		start += inc;		// Increment the start address
		count += 1;			// Increment the loop
	}
	// Now start should be the beginning of the layer containing the
	// given coordinates
	
	//FIXME: Not implemented yet
}
*/
	

void initialise_grid(grid* current_grid, int x, int y, int z, int particles) {
	/*
	 * Set up a grid at the given pointer, assigning it memory for its
	 * cells and its particles.
	 */
	
	// PRECONDITIONS
	assert(x > 0);
	assert(y > 0);
	assert(z > 0);
	assert(particles > 0);
	
	// Give the grid room for its cells
	current_grid->cells = (cell*) malloc( (x*y*z)*sizeof(cell) );
						
	// Then the same for its particles
	current_grid->particles = (particle*) malloc(particles*sizeof(particle));
									
	// Now we can populate the cell array by filling up rows, planes
	// and eventually the whole space
	int x_index, y_index, z_index;
	int x_rel, y_rel, z_rel;
	
	// Step through each cell
	for (x_index = 0; x_index < x; x_index++) {
		for (y_index = 0; y_index < y; y_index++) {
			for (z_index = 0; z_index < z; z_index++) {
				// x, y and z are the maximum values possible, so we can
				// index our grids using them, ie.
				//current_grid->cells[(x_index*y*z)+(y_index*z)+z_index]
				
				////////////////////////////////////////////////////////
				// FIXME!
				////////////////////////////////////////////////////////
				// We want to get rid of these arrays once we've got rid
				// of every use of them!
				////////////////////////////////////////////////////////
				// We need to allocate some memory for each cell's
				// neighbour pointers
				current_grid->cells[
					(y*z)*x_index+(z)*y_index+z_index
					].neighbours = (cell**) malloc(27*sizeof(cell*));
			
				// We need to make sure our particle list pointer is
				// definitely outside the particle list to begin with
				current_grid->cells[
					(y*z)*x_index+(z)*y_index+z_index
					].first_particle = NULL;
				
				// Assign our neighbours
				for (x_rel = -1; x_rel < 2; x_rel++) {
					for (y_rel = -1; y_rel < 2; y_rel++) {
						for (z_rel = -1; z_rel < 2; z_rel++) {
							// Only act if the current x, y and z
							// positions allow a neighbour
							if ((x_index + x_rel >= 0) 
								&& (x_index + x_rel < x) 
								&& (y_index + y_rel >= 0) 
								&& (y_index + y_rel < y) 
								&& (z_index + z_rel >= 0) 
								&& (z_index + z_rel < z)) {

								// Do some sanity checks
								
								// Is the current cell inside the cell
								// list?
								assert(
									y*z*x_index+z*y_index+z_index<x*y*z
								);
								
								// Is the current neighbour position
								// inside the neighbour list?
								assert(
									9*(x_rel+1)+3*(y_rel+1)+z_rel+1 < 27
								);
								
								// Would the neighbouring cell be inside
								// the cell list?
								assert(
									y*z*(x_index+x_rel)
										+z*(y_index+y_rel)
										+(z_index+z_rel)
										<x*y*z
								);
	
								// Assign this neighbour
								current_grid->cells[
									(y*z)*x_index+(z)*y_index+z_index
								].neighbours[
									(9*(x_rel+1))+(3*(y_rel+1))+z_rel+1
								] = &(current_grid->cells[
									(y*z)*(x_index+x_rel)
										+(z)*(y_index+y_rel)
										+(z_index+z_rel)
								]);
								
							}
							else {
								assert(
									y*z*x_index+z*y_index+z_index<x*y*z
								);
								assert(
									9*(x_rel+1)+3*(y_rel+1)+z_rel+1 < 27
								);
								
								current_grid->cells[
									(y*z)*x_index+(z)*y_index+z_index
								].neighbours[
									(9*(x_rel+1))+(3*(y_rel+1))+z_rel+1
								] = NULL;
							}
						}
					}
				}
			}
		}
	}
	
	// POSTCONDITIONS
	assert(current_grid->particles != NULL);
	assert(current_grid->cells != NULL);
	
}
	
void put_particle_in_cell(particle* the_particle, cell* intended_cell) {
	/*
	 * This checks whether the given cell already contains a particle.
	 * If so then the list is prepended by the particle. If not then the
	 * particle is used to start the list.
	 */
	
	// TODO: Add some failsafes to handle cases where:
	//  * the_particle is already in the cell (in which case return)
	//  * handle the_particle already having a cell (this involves
	//    taking the_particle out of the old list and repairing it)
	
	// PRECONDITIONS
	assert(the_particle != NULL);
	assert(intended_cell != NULL);
	
	// First of all, tell the particle its cell
	the_particle->container = intended_cell;
	
	// If the cell already contains a list of particles
	if (intended_cell->first_particle != NULL) {
		// Then point the given particle to the start of it
		the_particle->next = intended_cell->first_particle;
	}
	
	// Now add ourselves to the cell as the start
	intended_cell->first_particle = the_particle;
	
	// POSTCONDITIONS
	assert(intended_cell->first_particle != NULL);
	assert(the_particle->container != NULL);
	
}

void grid_particles(grid* the_grid, int particles, int x, int y, int z, double dx, double dy, double dz) {
	/*
	 * Goes through every particle in the given grid and assigns it to
	 * the relevant cell.
	 */
	
	// PRECONDITIONS
	assert(x > 0);
	assert(y > 0);
	assert(z > 0);
	assert(dx > 0);
	assert(dy > 0);
	assert(dz > 0);
	assert(particles > 0);
	assert(the_grid != NULL);
	
	int x_position, y_position, z_position;
	int particle_index;
	cell* intended_cell;
	for (particle_index=0; particle_index<particles; particle_index++) {
		x_position = get_index(
			the_grid->particles[particle_index].position.x, dx
		);
		
		y_position = get_index(
			the_grid->particles[particle_index].position.y, dy
		);
		
		z_position = get_index(
			the_grid->particles[particle_index].position.z, dz
		);
		
		intended_cell = &(
			the_grid->cells[(y*z)*x_position+(z)*y_position+z_position]
		);
		
		put_particle_in_cell(
			&(the_grid->particles[particle_index]), intended_cell
		);
	
	}
	
	// POSTCONDITIONS
	
}

void get_neighbours_for_particle(grid* the_grid, particle* the_particle,
	particle*** neighbour_particles, int* length) {
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
	assert(the_particle->container != NULL);
	
	// We run two loops here, where we could cache the particle numbers
	// in the cells instead, but this way makes things simpler by
	// keeping related operations together
	
	// First count the neighbours
	
	// Start our counter at 0
	*length = 0;
	// This will store the particle we're up to
	particle* found_particle;
	
	// Iterate through the given particle's cell's neighbours
	// NOTE: This loop includes the particle's cell for us too
	
	// These will keep track of the location of the neighbour
	// we're indexing
	int n_x, n_y, n_z;
	
	// Loop through the planes at x-1, x and x+1
	for (n_x = -1; n_x < 2; n_x++) {
		
		// Loop through the rows at y-1, y and y+1
		for (n_y = -1; n_y < 2; n_y++) {
			
			// Loop through the cells at z-1, z and z+1 
			for (n_z = -1; n_z < 2; n_z++) {
				
				// TODO: We can compact a lot of this calculation
				// together since it involves converting pointers to
				// coordinates and back, but for now we'll leave it
				// since it's easier to see what's going on
				
				// Get the coordinates of the current particle's cell
				
				// First get z by throwing away multiples of 
				// y_size*z_size (x coordinates) then 
				// z_size (y coordinates)
				int z = (
					(
						(the_particle->container - the_grid->cells) 
						/ sizeof(cell)
					) 
					% (the_grid->y_size * the_grid->z_size)
				) % (the_grid->z_size);
				
				// Now get y by taking off z and throwing away x
				// coordinates
				int y = (
					(
						(
							(the_particle->container - the_grid->cells) 
							/ sizeof(cell)
						) - z
					) % (the_grid->y_size * the_grid->z_size)
				) / the_grid->z_size;

				// Now get x by taking off z and z_size*y then divide by
				// y_size*z_size to get x
				int x = (
					(
						(the_particle->container - the_grid->cells) 
						/ sizeof(cell)
					) - z - (the_grid->z_size * y)
				) / (the_grid->y_size * the_grid->z_size);

				// Only act if this neighbour is actually in the grid
				if ((x+n_x >= 0) && (x+n_x < the_grid->x_size) &&
					(y+n_y >= 0) && (y+n_y < the_grid->y_size) &&
					(z+n_z >= 0) && (z+n_z < the_grid->z_size)) {
					
					assert(the_particle != NULL);
					assert(the_particle->container != NULL);
		
					// Get this neighbour's first particle
					found_particle = the_grid->cells[
						(the_grid->y_size * the_grid->z_size)*(x+n_x) 
							+ (the_grid->z_size)*(y+n_y) + (z+n_z)
					].first_particle;
		
					// As long as there are still particles in this
					// cell, loop
					while (found_particle != NULL) {
						// Increment the counter as long as we're not
						// counting the given particle
						if (found_particle != the_particle) {
							*length = *length + 1;
						}
						found_particle = found_particle->next;
					}
				}
			}
		}
	}
	
	// Now we know how many neighbours we have, let's allocate memory
	neighbour_particles[0] = (particle**) malloc(
		(*length)*sizeof(particle*)
	);

	// Now loop through the neighbours again, storing pointers to them
	*length = 0;		// We can re-use this as our index
	
	found_particle = NULL;
	
	// Loop through the planes at x-1, x and x+1
	for (n_x = -1 * (the_grid->y_size*the_grid->z_size);
		n_x <= (the_grid->y_size*the_grid->z_size);
		n_x += (the_grid->y_size*the_grid->z_size)) {
		
		// Loop through the rows at y-1, y and y+1
		for (n_y = -1 * (the_grid->z_size);
			n_y <= (the_grid->z_size);
			n_y += (the_grid->z_size)) {
			
			// Loop through the cells at z-1, z and z+1 
			for (n_z = -1; n_z <= 1; n_z += 1) {
				
				// TODO: We can compact a lot of this calculation
				// together since it involves converting pointers to
				// coordinates and back, but for now we'll leave it
				// since it's easier to see what's going on
				
				// Get the coordinates of the current particle's cell
				
				// First get z by throwing away multiples of 
				// y_size*z_size (x coordinates) then 
				// z_size (y coordinates)
				int z = (
					(
						(the_particle->container - the_grid->cells) 
						/ sizeof(cell)
					) % (the_grid->y_size * the_grid->z_size)
				) % (the_grid->z_size);
				
				// Now get y by taking off z and throwing away 
				// x coordinates
				int y = (
					(
						(
							(the_particle->container 
								- the_grid->cells) 
							/ sizeof(cell)
						) - z
					) % (the_grid->y_size * the_grid->z_size)
				) / the_grid->z_size;

				// Now get x by taking off z and z_size*y then 
				// divide by y_size*z_size to get x
				int x = (
					(
						(the_particle->container 
							- the_grid->cells) 
						/ sizeof(cell)
					) - z - (the_grid->z_size * y)
				) / (the_grid->y_size * the_grid->z_size);

				// Only act if this neighbour is actually in the grid
				if ((x+n_x >= 0) && (x+n_x < the_grid->x_size) &&
					(y+n_y >= 0) && (y+n_y < the_grid->y_size) &&
					(z+n_z >= 0) && (z+n_z < the_grid->z_size)) {
	
					found_particle = the_grid->cells[
						(x+n_x)*(the_grid->y_size)*(the_grid->z_size)
							+(y+n_y)*(the_grid->z_size)
							+(z+n_z)
					].first_particle;
					while (found_particle != NULL) {
						if (found_particle != the_particle) {
							(neighbour_particles[0])[
								(*length)
							] = found_particle;
							*length = *length + 1;
						}
						found_particle = found_particle->next;
					}
				}
			}
		}
	}
	
	// Now "neighbours" should point to an array of particle pointers
	// which are the neighbouring particles, and length should point to
	// the length, so we're done
	
	// POSTCONDITIONS
	//assert();
	
}
