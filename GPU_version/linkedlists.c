#include <stdio.h>
#include <stdlib.h>

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
	
	// The following two are implementation-specific and may be removed
	particle* next;		// Pointer used in particle linked lists
	cell* container;	// Pointer to containing cell, could be inferred
};

struct cell {
	/*
	 * A cell is a unit volume in the grid. It stores a pointer to a
	 * linked list of the particles it contains.
	 */
	
	// The following may be removed
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
	
	// This stores the volume units of this grid
	cell* cells;
	
	// This stores the particles which the cells contain
	particle* particles;
	
};

// FIXME: There should be no tests in the core library, so move this
// after debugging!

int count_cell_contents(cell* the_cell) {
	/*
	 * This returns the number of particles in the given cell.
	 */
	int count = 0;
	particle* current_particle = the_cell->first_particle;
	while (current_particle != NULL) {
		count++;
		current_particle = current_particle->next;
	}
	return count;
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
	
	// Give the grid room for its cells
	the_grid->cells = (cell*) malloc( (x*y*z)*sizeof(cell) );
						
	// Then the same for its particles
	the_grid->particles = (particle*)malloc(particles*sizeof(particle));
									
	// Now we can initialise the cell array by looping over rows, planes
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
				
				// We need to make sure our particle list pointer is
				// definitely outside the particle list to begin with
				the_grid->cells[
					(y*z)*x_index+(z)*y_index+z_index
				].first_particle = NULL;
			}
		}
	}
	
	// POSTCONDITIONS
	assert(the_grid->particles != NULL);
	assert(the_grid->cells != NULL);
	assert(the_grid->x_size > 0);
	assert(the_grid->y_size > 0);
	assert(the_grid->z_size > 0);
	assert(the_grid->dx > 0);
	assert(the_grid->dy > 0);
	assert(the_grid->dz > 0);
	assert(the_grid->particle_number > 0);
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

void grid_particles(grid* the_grid) {
	/*
	 * Goes through every particle in the given grid and assigns it to
	 * the relevant cell.
	 */
	
	// PRECONDITIONS
	assert(the_grid != NULL);
	
	int x_position, y_position, z_position;
	int particle_index;
	cell* intended_cell;
	for (particle_index=0; particle_index<the_grid->particle_number;
		particle_index++) {
			
		// Find which y-z plane this particle is in
		// Remember to take off the grid's offset!
		x_position = get_index(
			the_grid->particles[particle_index].x - the_grid->x_offset,
			the_grid->dx
		);
		
		// Find which x-z plane this particle is in
		// Remember to take off the grid's offset!
		y_position = get_index(
			the_grid->particles[particle_index].y - the_grid->y_offset,
			the_grid->dy
		);
		
		// Find which x-y plane this particle is in
		// Remember to take off the grid's offset!
		z_position = get_index(
			the_grid->particles[particle_index].z - the_grid->z_offset,
			the_grid->dz
		);
		
		// Now use the intersection of all of these to get its relevant
		// cell
		intended_cell = &(
			the_grid->cells[
				((the_grid->y_size)*(the_grid->z_size))*x_position
					+(the_grid->z_size)*y_position
					+z_position
			]
		);
		
		// Check the validity of the found cell
		// Mustn't be NULL
		assert(intended_cell != NULL);
		// Must be at least as high as the cell array's location
		assert(intended_cell >= the_grid->cells);
		// Can't be more than the upper bound on the cells
		assert(
			intended_cell <= the_grid->cells + (
				sizeof(cell) *
				the_grid->x_size *
				the_grid->y_size *
				the_grid->z_size
			)
		);
		
		// Now assign the particle to the found cell
		put_particle_in_cell(
			&(the_grid->particles[particle_index]), intended_cell
		);
	
	}
	
	// POSTCONDITIONS
	
}

void get_potential_neighbours_for_particle(grid* the_grid, 
	particle* the_particle, particle*** neighbour_particles,
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

	// FIXME: Remove me after debugging
	int count = 0;
	
	// TODO: We can compact a lot of this calculation
	// together since it involves converting pointers to
	// coordinates and back, but for now we'll leave it
	// since it's easier to see what's going on
			
	//// Get the coordinates of the current particle's cell
				
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

	//// Loop
	// Loop through the planes at x-1, x and x+1
	for (n_x = -1; n_x < 2; n_x++) {
		
		// Loop through the rows at y-1, y and y+1
		for (n_y = -1; n_y < 2; n_y++) {
			
			// Loop through the cells at z-1, z and z+1 
			for (n_z = -1; n_z < 2; n_z++) {

				// Only act if this neighbour is actually in the grid
				////////////////////////////////////////////////////////
				// FIXME: This is a completely broken way of working if
				// we want to support negative positions!
				////////////////////////////////////////////////////////
				if ((x+n_x >= 0) && (x+n_x < the_grid->x_size) &&
					(y+n_y >= 0) && (y+n_y < the_grid->y_size) &&
					(z+n_z >= 0) && (z+n_z < the_grid->z_size)) {
					
					// FIXME: Remove me after debugging!
					count = count_cell_contents(&(the_grid->cells[
						(the_grid->y_size * the_grid->z_size)*(x+n_x) 
							+ (the_grid->z_size)*(y+n_y) + (z+n_z)
					]));
					//fprintf(stderr, "%i\n", count);
					
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
		//fprintf(stderr, "Neighbours counted: %i\n", count);
	}
	
	// Now we know how many neighbours we have, let's allocate memory
	neighbour_particles[0] = (particle**) malloc(
		(*length)*sizeof(particle*)
	);

	// Now loop through the neighbours again, storing pointers to them
	// in the array we just allocated
	*length = 0;		// We can re-use this as our index
	
	found_particle = NULL;
	
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
				////////////////////////////////////////////////////////
				// FIXME: This is completely broken if we allow negative
				// particle positions!
				////////////////////////////////////////////////////////
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
