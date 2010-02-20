#include "linkedlists.c"
#include <stdio.h>
#include <stdlib.h>

void dump_grid(grid* the_grid, double dx, double dy, double dz, 
	int print_cells, int print_particles, int print_lists) {
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
	
	// PRECONDITIONS
	assert (the_grid != NULL);
	
	// Start with the grid, saying that we're a grid and our address
	printf("GRID:%u{", the_grid);
	// Now give our size
	printf("X:%i,Y:%i,Z:%i,", the_grid->x_size, 
		the_grid->y_size, the_grid->z_size);
	// And our population
	printf("POPULATION:%i,", the_grid->particle_number);
	// And our cell dimensions
	printf("DX:%G,DY:%G,DZ:%G", dx, dy, dz);
	
	// Branch if we've been told to check the cells
	if (print_cells != 0) {
		// Start a collection of cells, giving the starting address
		printf(",CELLS:STARTINGAT:%u{", the_grid->cells);
		// Go through the memory which we assume is devoted to cells
		int cell_count;
		for (cell_count = 0; 
			cell_count < 
				(the_grid->x_size)
				*(the_grid->y_size)
				*(the_grid->z_size); 
			cell_count++) {
			printf("CELL:%u{", &(the_grid->cells[cell_count]));
			
			// Branch if we have been told to print our particles
			if (print_particles != 0) {
				if (the_grid->cells[cell_count].first_particle == NULL){
					printf("FIRSTPARTICLE:NULL");
				}
				else {
					printf("FIRSTPARTICLE:%u",
						the_grid->cells[cell_count].first_particle);
				}
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
	}
	
	// Branch if we've been told to print particles
	if (print_particles != 0) {
		printf(",PARTICLES:STARTINGAT:%u{", the_grid->particles);
		int particle_count;
		for (particle_count = 0; 
			particle_count < the_grid->particle_number; 
			particle_count++) {
			printf("PARTICLE:%u{", 
				&(the_grid->particles[particle_count]));
			printf("X:%G", 
				the_grid->particles[particle_count].position.x);
			printf(",Y:%G", 
				the_grid->particles[particle_count].position.y);
			printf(",Z:%G", 
				the_grid->particles[particle_count].position.z);
			printf(",CONTAINER:%u", 
				the_grid->particles[particle_count].container);
			// Branch if we've been told to print particle lists
			if (print_lists != 0) {
				printf(",NEXT:%u", 
					the_grid->particles[particle_count].next);
			}
			printf("}");
			if (particle_count < (the_grid->particle_number) - 1) {
				printf(",");
			}
		}
		printf("}");
	}
	
	printf("}");
}

double new_random() {
	/*
	 * Returns a random double between 0 and 0.999...
	 * NOTE: You may wish to change the random seed first using srand()!
	 */
	 return rand()/(double)(RAND_MAX);
}

void populate_random(grid* the_grid, int particles, int x, int y, int z,
	double dx, double dy, double dz) {
	/*
	 * This populates the given grid with particles, determining their
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
	assert(the_grid != NULL);
	
	// Store the total size of the space
	three_vector space;
	space.x = ((double) x) * dx;
	space.y = ((double) y) * dy;
	space.z = ((double) z) * dz;
	
	// Loop through the particle memory
	int particle_index;
	for (particle_index = 0;
		particle_index < particles;
		particle_index++) {
		// Assign the position
		the_grid->particles[particle_index].position.x =
			new_random()*space.x;
		the_grid->particles[particle_index].position.y = 
			new_random()*space.y;
		the_grid->particles[particle_index].position.z = 
			new_random()*space.z;
		
		// Ensure that the particle's linked list pointer doesn't start
		// in the particle memory
		the_grid->particles[particle_index].next = NULL;
	}
	
	// POSTCONDITIONS
	assert(check_particles(the_grid, particles) == 0);
	
}

int check_particles(grid* the_grid, int particle_number) {
	// PRECONDITIONS
	assert(the_grid != NULL);
	assert(particle_number >= 0);
	
	int c, x, y, z;
	
	int counter;
	for (counter = 0; counter < particle_number; counter++) {
		//assert(the_grid->particles[counter] != NULL);
		////////////////////////////////////////////////////////////////
		// FIXME: These tests are completely broken if we allow
		// negative positions!
		////////////////////////////////////////////////////////////////
		assert(the_grid->particles[counter].position.x >= 0);
		assert(the_grid->particles[counter].position.y >= 0);
		assert(the_grid->particles[counter].position.z >= 0);
		
		c = (the_grid->particles[counter].container-the_grid->cells)
			/ sizeof(cell);
		z = (c % (the_grid->y_size*the_grid->z_size))%the_grid->z_size;
		y = ((c - z) % (the_grid->y_size*the_grid->z_size))
			/ (the_grid->z_size);
		x = (c - z - (the_grid->z_size * y))
			/ (the_grid->y_size * the_grid->z_size);
		
		
	}
	return 0;
}

void count_neighbours(particle* current_particle, double radius, 
	particle** to_check, int length, int* result) {
	/*
	 * This will count the true neighbours of current_particle, ie.
	 * those within one radius of it, out of the array of particle
	 * pointers to_check, the length of which is given as "length".
	 * The result is written into "result".
	 */
	
	// PRECONDITIONS
	assert(current_particle != NULL);
	assert(radius >= 0);
	assert(to_check != NULL);
	assert(length >= 0);
	
	// These will store the values of our comparisons
	double delta_x, delta_y, delta_z;
	// This will store the running total of true neighbours
	//int count = 0;
	*result = 0;
	// This is simply the loop iterator
	int index;
	// Loop through the given array
	for (index = 0; index < length; index++) {
		// Don't act if we're comparing the current_particle to itself
		if (to_check[index] != current_particle) {
			// Work out this particle's distance from current_particle
			delta_x = current_particle->position.x 
				- to_check[index]->position.x;
			delta_y = current_particle->position.y 
				- to_check[index]->position.y;
			delta_z = current_particle->position.z 
				- to_check[index]->position.z;
			// Increment the count if we're within a radius
			// (use squares to save on unnecessary computation)
			if ((delta_x*delta_x) 
				+ (delta_y*delta_y) 
				+ (delta_z*delta_z) <= (radius*radius)) {
				//count++;
				*result = (*result) + 1;
			}
		}
	}
	
	// POSTCONDITIONS
	//assert(count >= 0);
	
	//return count;
}

void find_neighbours_by_brute_force(particle* current_particle, 
	double radius, particle*** neighbour_particles, int* length, 
	particle** set_to_check, int length_of_set) {
	/*
	 * This function will go through all "length_of_set" particles in
	 * the array "set_to_check", comparing their distance to the given
	 * "current_particle". If they are within "radius" distance from
	 * each other then a pointer to that particle is added to the array
	 * pointed to by "neighbour_particles". After they have all been
	 * compared, "length" will point to the number of neighbours found.
	 * 
	 * WARNING! This will modify the array it is given!
	 * 
	 */
	
	// PRECONDITIONS
	assert(current_particle != NULL);
	assert(radius > 0);
	assert(set_to_check != NULL);
	assert(length != NULL);
	assert(length_of_set >= 0);
	
	*length = 0;
	count_neighbours(current_particle, radius, set_to_check, 
		length_of_set, length);
	neighbour_particles[0] = (particle**) malloc(
		*length*sizeof(particle*)
	);
	
	int position = 0;
	
	// These will let us compare the positions
	double delta_x;
	double delta_y;
	double delta_z;
	
	// This points to the current particle in the given array
	particle* particle_being_checked;
	
	// Loop through the given array
	int index;
	for (index = 0; index < length_of_set; index++) {
		// Get the next particle
		particle_being_checked = set_to_check[index];
		
		// Make sure it exists
		assert(particle_being_checked != NULL);
		
		// If we've found the current particle then discard it
		// otherwise...
		if (particle_being_checked != current_particle) {
			// Get the difference in their positions
			delta_x = particle_being_checked->position.x 
				- current_particle->position.x;
			delta_y = particle_being_checked->position.y 
				- current_particle->position.y;
			delta_z = particle_being_checked->position.z 
				- current_particle->position.z;
			// See whether they're within one radius of each other
			if (delta_x*delta_x + delta_y*delta_y + delta_z*delta_z <= 
				radius*radius) {
				// If so then don't discard it, and increment the length
				neighbour_particles[0][position]=particle_being_checked;
				//fprintf(stderr, " %i ", particle_being_checked);
				position++;
				
			}
		}
	}
		
	// Now our array is populated, so we can return
	
	// POSTCONDITIONS
	// We should have a valid output array
	assert(neighbour_particles != NULL);
	// We shouldn't output more particles than we were given
	assert(*length <= length_of_set);
	
}

void brute_force_all(particle* current_particle, grid* the_grid, 
	int number, double radius, particle*** found, int* length) {
	int index;
	double delta_x;
	double delta_y;
	double delta_z;
	int counter = 0;
	for (index = 0; index < number; index++) {
		// Don't bother comparing the current_particle to itself
		if (&(the_grid->particles[index]) != current_particle) {
			// Get the difference in positions
			delta_x = the_grid->particles[index].position.x 
				- current_particle->position.x;
			delta_y = the_grid->particles[index].position.y 
				- current_particle->position.y;
			delta_z = the_grid->particles[index].position.z 
				- current_particle->position.z;
			// If they're within the radius then remember this pointer
			if (delta_x*delta_x + delta_y*delta_y + delta_z*delta_z <= 
				radius*radius) {
				counter++;
			}
		}
	}
	
	*length = counter;
	found[0] = (particle**) malloc( counter*sizeof(particle*) );
	
	counter = 0;
	
	for (index = 0; index < number; index++) {
		if ((&the_grid->particles[index]) != current_particle) {
			// Get the difference in positions
			delta_x = the_grid->particles[index].position.x 
				- current_particle->position.x;
			delta_y = the_grid->particles[index].position.y 
				- current_particle->position.y;
			delta_z = the_grid->particles[index].position.z 
				- current_particle->position.z;
			// If they're within the radius then remember this pointer
			if (delta_x*delta_x + delta_y*delta_y + delta_z*delta_z <= 
				radius*radius) {
				found[0][counter] = &(the_grid->particles[index]);
				counter++;
			}
		}
	}
	
	// POSTCONDITIONS
	assert(*length <= number);
	assert(found != NULL);
	
}

int count_all_particles(grid* the_grid, int x, int y, int z) {
	
	// PRECONDITIONS
	assert(the_grid != NULL);
	assert(x > 0);
	assert(y > 0);
	assert(z > 0);
	
	int count = 0;
	int index;
	particle* current_particle;
	for (index=0; index < x*y*z; index++) {
		current_particle = the_grid->cells[index].first_particle;
		while (current_particle != NULL) {
			current_particle = current_particle->next;
			count++;
		}
	}
	return count;
}

int check_for_dupes(particle** particles, int length) {
	/*
	 * This checks the given array of particles, with a length "length",
	 * to see whether it contains two pointers to the same address.
	 * If duplicates are found return 1, else return 0.
	 */
	
	// PRECONDITIONS
	assert(particles != NULL);
	assert(length >= 0);

	// There can be no duplicates if the array is empty or a singleton
	if (length < 2) {
		return 0;
	}
	
	int index1, index2;
	for (index1 = 0; index1 < length; index1++) {
		for (index2 = 0; index2 < length; index2++) {
			if (index1 != index2) {
				if (particles[index1] == particles[index2]) {
					return 1;
				}
			}
		}
	}
	return 0;
}

int main() {
	// These are the dimensions of our volume elements
	double dx = 1.0;
	double dy = 1.0;
	double dz = 1.0;
	
	// These are the dimensions of our space as multiples of dx, dy & dz
	int x = 4;
	int y = 4;
	int z = 4;
	
	// This is the total number of particles to simulate
	int particle_number = 400;
	
	// Make the space we are simulating
	grid the_grid;
	
	// Allocate memory and assign neighbourhoods
	initialise_grid(&the_grid, x, y, z, dx, dy, dz, particle_number);
	
	return 0;
}
	
	/*
	
	// DEBUGGING
	assert(the_grid.cells != NULL);
	assert(the_grid.particles != NULL);
	
	// Fill the space with particles at random positions
	populate_random(&the_grid, particle_number, x, y, z, dx, dy, dz);
	
	// Sort the particles into cells
	grid_particles(&the_grid);

	// DEBUGGING
	assert(the_grid.particles[0].container != NULL);
	assert(count_all_particles(&the_grid, x, y, z) == particle_number);

	//dump_grid(&(the_grid), dx, dy, dz, 1, 1, 1, 1);

	//return 0;

	// Choose a particle from the group
	particle* test_particle = &(the_grid.particles[0]);

	// DEBUGGING
	assert(test_particle != NULL);
	assert(test_particle->container != NULL);

	////////////////////////////////////////////////////////////////////
	// First get z by throwing away multiples of 
	// y_size*z_size (x coordinates) then 
	// z_size (y coordinates)
	int cz = (
		(
			(test_particle->container - the_grid.cells) 
			/ sizeof(cell)
		) 
		% (the_grid.y_size * the_grid.z_size)
	) % (the_grid.z_size);
			
	// Now get y by taking off z and throwing away x
	// coordinates
	int cy = (
		(
			(
				(test_particle->container - the_grid.cells) 
				/ sizeof(cell)
			) - cz
		) % (the_grid.y_size * the_grid.z_size)
	) / the_grid.z_size;

	// Now get x by taking off z and z_size*y then divide by
	// y_size*z_size to get x
	int cx = (
		(
			(test_particle->container - the_grid.cells) 
			/ sizeof(cell)
		) - cz - (the_grid.z_size * cy)
	) / (the_grid.y_size * the_grid.z_size);
	
	/*fprintf(stderr, "Particle is at (%G,%G,%G), in cell (%i,%i,%i)\n", 
		test_particle->position.x,
		test_particle->position.y,
		test_particle->position.z,
		cx, cy, cz);*//*
	int n_x, n_y, n_z;
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
				if ((cx+n_x >= 0) && (cx+n_x < the_grid.x_size) &&
					(cy+n_y >= 0) && (cy+n_y < the_grid.y_size) &&
					(cz+n_z >= 0) && (cz+n_z < the_grid.z_size)) {
					//fprintf(stderr, "Neighbour at (%i,%i,%i)\n",
					//	cx+n_x, cy+n_y, cz+n_z);
				}
			}
		}
	}

	// Find its neighbours through the grid
	// This will point to an array of neighbours
	particle** neighbour_array;
	// This will tell us how long the array is
	int neighbour_number;
	get_potential_neighbours_for_particle(&the_grid, test_particle,
		&neighbour_array, &neighbour_number);

	//fprintf(stderr, "\nNeighbour array [");
	int ix;
	for (ix=0; ix < neighbour_number; ix++) {
		
		////////////////////////////////////////////////////////////////////
		// First get z by throwing away multiples of 
		// y_size*z_size (x coordinates) then 
		// z_size (y coordinates)
		n_z = (
			(
				(neighbour_array[ix]->container - the_grid.cells) 
				/ sizeof(cell)
			) 
			% (the_grid.y_size * the_grid.z_size)
		) % (the_grid.z_size);
			
		// Now get y by taking off z and throwing away x
		// coordinates
		n_y = (
			(
				(
					(neighbour_array[ix]->container - the_grid.cells) 
					/ sizeof(cell)
				) - n_z
			) % (the_grid.y_size * the_grid.z_size)
		) / the_grid.z_size;

		// Now get x by taking off z and z_size*y then divide by
		// y_size*z_size to get x
		n_x = (
			(
				(neighbour_array[ix]->container - the_grid.cells) 
				/ sizeof(cell)
			) - n_z - (the_grid.z_size * n_y)
		) / (the_grid.y_size * the_grid.z_size);
		
		fprintf(stderr, "(%i %i %i), ", cx-n_x, cy-n_y, cz-n_z);
		
		
		//fprintf(stderr, "%u, ", 
		//	((int)neighbour_array[ix]-(int)the_grid.particles)/sizeof(particle));
	}
	//fprintf(stderr, "]\n");
	
	// DEBUGGING
	assert(neighbour_number >= 0);
	
	assert(check_for_dupes(neighbour_array, neighbour_number) == 0);

	//fprintf(stderr, "Potentials found: %i\n", neighbour_number);
	
	int neighbours_in_grid = 0;
	count_neighbours(test_particle, dx, neighbour_array, 
		neighbour_number, &neighbours_in_grid);
	
	//fprintf(stderr, "True neighbours a: %i\n", neighbours_in_grid);
	
	// Now find which of those are true neighbours
	particle** true_neighbour_array;
	int true_neighbour_number;
	find_neighbours_by_brute_force(test_particle, dx,
		&true_neighbour_array, &true_neighbour_number, neighbour_array,
		neighbour_number); 
	
	fprintf(stderr, "\nTrue neighbours [");
	for (ix=0; ix < true_neighbour_number; ix++) {
		fprintf(stderr, "%u, ", 
			(true_neighbour_array[ix] - the_grid.particles)
				/ sizeof(particle)
		);
	}
	fprintf(stderr, "]\n");
	
	return 0;
	
	//fprintf(stderr, "True neighhbours b: %i\n", true_neighbour_number);
	
	particle** all_pointers = (particle**) malloc(
		particle_number*sizeof(particle*)
	);
	int pointer_index;
	for (pointer_index = 0;
		pointer_index < particle_number;
		pointer_index++) {
		all_pointers[pointer_index] = &(
			the_grid.particles[pointer_index]
		);
	}
	
	int total_neighbours = 0;
	count_neighbours(test_particle, dx, 
		all_pointers, particle_number, &total_neighbours);
	
	fprintf(stderr, "True neighbours c: %i\n", total_neighbours);
	
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
