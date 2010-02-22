#include "linkedlists.c"
#include <stdio.h>
#include <stdlib.h>

float new_random() {
	/*
	 * Returns a random double between 0 and 0.999...
	 * NOTE: You may wish to change the random seed first using srand()!
	 */
	 return rand()/(float)(RAND_MAX);
}

void populate_random(grid* the_grid) {
	/*
	 * This populates the given grid with particles, determining their
	 * positions using the standard rand() function.
	 */
	
	// PRECONDITIONS
	assert(the_grid != NULL);
	
	// Store the total size of the space
	int x_space, y_space, z_space;
	x_space = ((float) the_grid->x_size) * the_grid->dx;
	y_space = ((float) the_grid->y_size) * the_grid->dy;
	z_space = ((float) the_grid->z_size) * the_grid->dz;
	
	// Loop through the particle memory
	int particle_index;
	for (particle_index = 0;
		particle_index < the_grid->particle_number;
		particle_index++) {
		// Assign the position
		the_grid->particles[particle_index].x =
			(new_random() * x_space) - the_grid->x_offset;
		the_grid->particles[particle_index].y = 
			(new_random() * y_space) - the_grid->y_offset;
		the_grid->particles[particle_index].z = 
			(new_random() * z_space) - the_grid->z_offset;
		
		// Ensure that the particle's linked list pointer doesn't start
		// in the particle memory
		the_grid->particles[particle_index].next = NULL;
	}
	
	// POSTCONDITIONS
	//assert(check_particles(the_grid, particles) == 0);
	
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

void get_index_from_cell(grid* the_grid, particle* the_particle,
							int* x, int* y, int* z) {
	/*
	 * This function works out the given particle's cell location in
	 * the given grid based on the cell's memory location, vital for
	 * finding neighbouring cells.
	 * This is used for benchmarking methods for calculating the index.
	 */
	int temp;
	
	// First get the relative memory location in the array, rather than
	// the absolute memory location in RAM
	temp = (the_particle->container) - (the_grid->cells);

	// The memory is arranged as each z from z_min to z_max for each y
	// from y_min to y_max for each x from x_min to x_max.
	// Therefore the z fluctuates at every step, the y only fluctuates
	// when the relevant z locations have been exhausted (ie. every
	// z_size steps), and the x only changes once the relevant y
	// locations have been exhausted, ie. once y has changed y_size
	// times, and since y changes every z_size steps this means x
	// changes every y_size * z_size steps.
	
	// Get z by discarding everything above z_size
	*z = temp % the_grid->z_size;
	
	// We can remove z fluctuations from our location and normalise to y
	temp = temp - *z;
	temp = temp / the_grid->z_size;

	// Get y by discarding everything above y_size
	*y = temp % the_grid->y_size;
	
	// Now remove y and normalise to x
	temp = temp - *y;
	temp = temp / the_grid->y_size;

	// Now find x
	*x = temp;
		
}

void compare_index_methods(grid* the_grid) {
	/*
	 * Ensures that both methods for obtaining a particle's cell's
	 * position are valid and give the same results.
	 */
	
	// We need to store three indices for each method
	int x1, y1, z1, x2, y2, z2;
	
	// Loop through every particle
	int particle_index;
	for (particle_index=0; particle_index < the_grid->particle_number;
		particle_index++) {
		
		// Ensure that our indices are different to start with
		x1 = 0;
		x2 = 1;
		y1 = 0;
		y2 = 1;
		z1 = 0;
		z2 = 1;

		// Test the first method
		get_index_from_position(the_grid,
			&(the_grid->particles[particle_index]),
			&x1, &y1, &z1);

		// Test the second method
		get_index_from_cell(the_grid,
			&(the_grid->particles[particle_index]),
			&x2, &y2, &z2);
		
		// Debug
		//fprintf(stderr, "%f -> %i, %i; %f -> %i, %i; %f -> %i, %i\n",
		//	the_grid->particles[particle_index].x, x1, x2,
		//	the_grid->particles[particle_index].y, y1, y2,
		//	the_grid->particles[particle_index].z, z1, z2);
		
		// Make sure they give the same answer
		assert(x1 == x2);
		assert(y1 == y2);
		assert(z1 == z2);
		
	}
	
}

int main() {
	//// Setup a grid
	
	// Cell size is 1.0
	float dx, dy, dz;
	dx = dy = dz = 1.0;
	
	// Make a 10x10x10 grid
	int x, y, z;
	x = y = z = 10;
	
	// Cells start at the origin
	float x_offset, y_offset, z_offset;
	x_offset = y_offset = z_offset = 0.0;
	
	// Put 1000 particles in the grid
	int particle_number = 1000;
	
	// The grid itself (keep the_grid a pointer for consistency)
	grid main_grid;
	grid* the_grid = &(main_grid);
	
	// Allocate memory
	initialise_grid(the_grid, x, y, z, dx, dy, dz,
		x_offset, y_offset, z_offset, particle_number);
	
	// Fill with random particles for testing purposes
	populate_random(the_grid);
	
	// Assign the particles to cells
	grid_particles(the_grid);
	
	// Check particle cell indexing
	compare_index_methods(the_grid);
	
	return 0;
}
