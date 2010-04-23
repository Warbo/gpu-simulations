#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "linkedlists.c"


//void dump_grid(grid* the_grid, float dx, float dy, float dz) {
	/*
	 * This function will dump the entire memory representation of the
	 * grid at the given pointer to stdout. This can be piped in to a
	 * sanity checking program to aid in debugging.
	 * 
	 * The other arguments instruct the function to refrain or not from
	 * printing out grid properties (since it is useless to output the
	 * memory of a particle if it is yet to be defined).
	 * These values are simply 0 for no and nonzero for yes.
	 */
/*	

}*/

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

/*int check_particles(grid* the_grid, int particle_number) {
	
}*/

/*int count_all_particles(grid* the_grid, int x, int y, int z) {
	
	
}*/

//int check_for_dupes(particle** particles, int length) {
	/*
	 * This checks the given array of particles, with a length "length",
	 * to see whether it contains two pointers to the same address.
	 * If duplicates are found return 1, else return 0.
	 */
/*	
	
}*/

int main() {
	// These are the dimensions of our volume elements
	float dx = (float)1.0;
	float dy = (float)1.0;
	float dz = (float)1.0;
	
	// These are the dimensions of our space as multiples of dx, dy & dz
	int x = 50;
	int y = 75;
	int z = 100;
	
	// These are the offsets of the grid coordinates from the particles'
	float x_offset = (float)-4.0;
	float y_offset = (float)3.2;
	float z_offset = (float)18.7;
	
	// This is the total number of particles to simulate
	int particle_number = 375000;
	
	// Make the space we are simulating
	grid the_grid;
	
	// Allocate memory and assign neighbourhoods
	initialise_grid(&the_grid, x, y, z, dx, dy, dz, 
		x_offset, y_offset, z_offset, particle_number);
		
	// DEBUGGING
	assert(the_grid.particles != NULL);

	particle* p_array = (particle*) malloc( ((unsigned int)particle_number) *
		sizeof(particle));
	
	// Fill the space with particles at random positions
	populate_random(&p_array, particle_number, x, y, z, dx, dy, dz);
	
	// Sort the particles into cells
	grid_particles(&the_grid, p_array);

	// Choose a particle randomly from the group
	particle* test_particle = NULL;
	int test_probe;
	while (test_particle == NULL) {
		test_probe = (int)( ((float)particle_number) * new_random());
		if (the_grid.particles[test_probe].id > 0) {
			test_particle = &(the_grid.particles[test_probe]);
		}
	}

	// DEBUGGING
	assert(test_particle != NULL);
	assert(test_particle->id != -1);

	// Find its neighbours through the grid
	// This will point to an array of neighbours
	particle* neighbour_array;
	// This will tell us how long the array is
	int neighbour_number = -1;
	get_potential_neighbours_for_particle(&the_grid, test_particle,
		&neighbour_array, &neighbour_number);

	// DEBUGGING
	assert(neighbour_number >= 0);

	fprintf(stderr, "Potentials found: %i\n", neighbour_number);
	
	// Now find which of those are true neighbours
	particle* true_neighbour_array;
	int true_neighbour_number = -1;
	get_true_neighbours_for_particle(&the_grid, test_particle,
		&true_neighbour_array, &true_neighbour_number);
	
	fprintf(stderr, "True neighhbours: %i\n", true_neighbour_number);

	particle* brute_force_neighbours;
	int total_neighbours = -1;
	get_neighbours_brute_force(&the_grid, test_particle,
	&(brute_force_neighbours), &total_neighbours);
	
	fprintf(stderr, "True neighbours from all: %i\n", total_neighbours);

	assert(total_neighbours == true_neighbour_number);
	assert(total_neighbours < neighbour_number);

	// Check every particle found
	float delta_x, delta_y, delta_z;
	
	// Check that all of the "true neighbours" were found during 
	// brute force
	int index1;		// Used for indexing the "true" array
	int index2;		// Used for indexing the "brute" array
	int found;		// 0 means no match yet, 1 means match found
	
	for (index1 = 0; index1 < true_neighbour_number; index1++) {
		found = 0;
		delta_x = true_neighbour_array[index1].x
			- test_particle->x;
		delta_y = true_neighbour_array[index1].y
			- test_particle->y;
		delta_z = true_neighbour_array[index1].z
			- test_particle->z;

		// Check that the particle is a true neighbour
		assert((delta_x*delta_x)+(delta_y*delta_y)+(delta_z*delta_z) <=
			(the_grid.dx * the_grid.dx));
		
		for (index2 = 0; index2 < total_neighbours; index2++) {
			if (found == 0) {
				if (true_neighbour_array[index1].id ==
					brute_force_neighbours[index2].id) {
					found = 1;
				}
			}
		}

		// Ensure the grid particle was found in the brute force
		assert(found == 1);

	}
	
	// Now do the same but the other way around
	for (index2 = 0; index2 < total_neighbours; index2++) {
		found = 0;
		
		delta_x = brute_force_neighbours[index2].x
			- test_particle->x;
		delta_y = brute_force_neighbours[index2].y
			- test_particle->y;
		delta_z = brute_force_neighbours[index2].z
			- test_particle->z;
		
		for (index1 = 0; index1 < true_neighbour_number; index1++) {
			if (found == 0) {
				if (true_neighbour_array[index1].id ==
					brute_force_neighbours[index2].id) {
					found = 1;
				}
			}
		}

		// Ensure everything found through brute force was found by the grid
		assert(found == 1);
		
	}

	// Success
	return 0;
	
}
