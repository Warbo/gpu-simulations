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
	// PRECONDITIONS
	assert (the_grid != NULL);
	
	// Start with the grid, saying that we're a grid and our address
	printf("GRID:%u{", (int)the_grid);
	// Now give our size
	printf("X:%i,Y:%i,Z:%i,", the_grid->x_size, 
		the_grid->y_size, the_grid->z_size);
	// Our offset
	printf("OX:%G,OY:%G,OZ:%G,", the_grid->x_offset, the_grid->y_offset,
		the_grid->z_offset);
	// And our population
	printf("POPULATION:%i,", the_grid->particle_number);
	// And our cell dimensions
	printf("DX:%G,DY:%G,DZ:%G", dx, dy, dz);
	
	// Print cells
	// Start a collection of cells, giving the starting address
	printf(",CELLS:STARTINGAT:%u{", (int)(the_grid->cells));
	// Go through the memory which we assume is devoted to cells
	int cell_count;
	for (cell_count = 0; 
		cell_count < 
			(the_grid->x_size)
			*(the_grid->y_size)
			*(the_grid->z_size); 
		cell_count++) {
		printf("CELL:%u{", (int)(&(the_grid->cells[cell_count])));
		
		// Print our particles
		if (the_grid->cells[cell_count].first_particle == NULL){
			printf("FIRSTPARTICLE:NULL");
		}
		else {
			printf("FIRSTPARTICLE:%u",
				(int)(the_grid->cells[cell_count].first_particle));
		}
		
		printf("}");
		// For all but the last cell we need to insert a comma here
		if (cell_count < (
			(the_grid->x_size)*(the_grid->y_size)*(the_grid->z_size)
		)-1) {
			printf(",");
		}
	}
	printf("}");
	
	// Print particles
	printf(",PARTICLES:STARTINGAT:%u{", (int)(the_grid->particles));
	
	// Loop through all of the particles
	int particle_count;
	for (particle_count = 0; 
		particle_count < the_grid->particle_number; 
		particle_count++) {
			
		// Print the particle's address
		printf("PARTICLE:%u{", 
			(int)(&(the_grid->particles[particle_count])));
		
		// Position
		printf("X:%G", 
			the_grid->particles[particle_count].x);
		printf(",Y:%G", 
			the_grid->particles[particle_count].y);
		printf(",Z:%G", 
			the_grid->particles[particle_count].z);
		printf(",CONTAINER:%u", 
			(int)(the_grid->particles[particle_count].container));
		// Print the list pointer
		printf(",NEXT:%u", 
			(int)(the_grid->particles[particle_count].next));
		
		printf("}");
		if (particle_count < (the_grid->particle_number) - 1) {
			printf(",");
		}
	}
	
	// Close the braces we opened
	printf("}");
	
	printf("}");
}*/

double new_random() {
	/*
	 * Returns a random double between 0 and 0.999...
	 * NOTE: You may wish to change the random seed first using srand()!
	 */
	 return rand()/(double)(RAND_MAX);
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
	double x_space, y_space, z_space;
	x_space = ((double) x) * dx;
	y_space = ((double) y) * dy;
	z_space = ((double) z) * dz;
	
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
	float dx = 1.0;
	float dy = 1.0;
	float dz = 1.0;
	
	// These are the dimensions of our space as multiples of dx, dy & dz
	int x = 2;
	int y = 2;
	int z = 2;
	
	// These are the offsets of the grid coordinates from the particles'
	float x_offset = 0.0;
	float y_offset = 0.0;
	float z_offset = 0.0;
	
	// This is the total number of particles to simulate
	int particle_number = 500;
	
	// Make the space we are simulating
	grid the_grid;
	
	// Allocate memory and assign neighbourhoods
	initialise_grid(&the_grid, x, y, z, dx, dy, dz, 
		x_offset, y_offset, z_offset, particle_number);
		
	// DEBUGGING
	assert(the_grid.particles != NULL);

	particle* p_array = (particle*) malloc(particle_number*sizeof(particle));
	
	// Fill the space with particles at random positions
	populate_random(&p_array, particle_number, x, y, z, dx, dy, dz);
	
	// Sort the particles into cells
	grid_particles(&the_grid, p_array);

	// Choose a particle from the group
	particle* test_particle = &(the_grid.particles[1]);

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
	
	int total_neighbours = -1;
	get_neighbours_brute_force(&the_grid, test_particle,
	&(the_grid.particles), &total_neighbours);
	
	fprintf(stderr, "True neighbours from all: %i\n", total_neighbours);

	return 0;
}
	/*
	assert(neighbours_in_grid == true_neighbour_number);
	
	assert(total_neighbours == neighbours_in_grid);
	
	particle** brute_force_array;

	int total_neighbours2;
	
	find_neighbours_by_brute_force(test_particle, dx, 
		&brute_force_array, &total_neighbours2, all_pointers, 
		particle_number);
	
	assert(total_neighbours == neighbours_in_grid);
	
	// Now "true_neighbour_array" should contain "true_neighbour_number"
	// neighbours
	
	
	int brute_force_number;
	brute_force_all(test_particle, &the_grid, particle_number, dx, 
		&brute_force_array, &brute_force_number);
	
	// Now "brute_force_array" should contain "brute_force_number"
	
	// Compare the two
	assert(total_neighbours == true_neighbour_number);
	
	
	double delta_x, delta_y, delta_z;
	
	// Check that all of the "true neighbours" were found during 
	// brute force
	int index1;		// Used for indexing the "true" array
	int index2;		// Used for indexing the "brute" array
	int found;		// 0 means no match yet, 1 means match found
	
	int diff_count = 0;
	
	for (index1 = 0; index1 < true_neighbour_number; index1++) {
		found = 0;
		delta_x = true_neighbour_array[index1]->position.x 
			- test_particle->position.x;
		delta_y = true_neighbour_array[index1]->position.y 
			- test_particle->position.y;
		delta_z = true_neighbour_array[index1]->position.z 
			- test_particle->position.z;
		
		for (index2 = 0; index2 < brute_force_number; index2++) {
			if (found == 0) {
				if (true_neighbour_array[index1] ==
					brute_force_array[index2]) {
					found = 1;
				}
			}
		}
		if (found == 0) {
			diff_count++;
			return 1;
		}
	}
	
	diff_count = 0;
	
	// Now do the same but the other way around
	for (index2 = 0; index2 < brute_force_number; index2++) {
		found = 0;
		
		delta_x = brute_force_array[index2]->position.x 
			- test_particle->position.x;
		delta_y = brute_force_array[index2]->position.y 
			- test_particle->position.y;
		delta_z = brute_force_array[index2]->position.z 
			- test_particle->position.z;
		
		for (index1 = 0; index1 < true_neighbour_number; index1++) {
			if (found == 0) {
				if (true_neighbour_array[index1] ==
					brute_force_array[index2]) {
					found = 1;
				}
			}
		}
		if (found == 0) {
			diff_count++;
			return 1;
		}
	}
	
	//return 0;
}
*/
