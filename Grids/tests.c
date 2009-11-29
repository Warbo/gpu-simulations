#include "linkedlists.c"
#include <stdio.h>
#include <stdlib.h>

void dump_grid(grid* the_grid, int print_cells, int print_particles,
	int print_neighbours, int print_lists) {
	/*
	 * This function will dump the entire memory representation of the
	 * grid at the given pointer to stdout. This can be piped in to a
	 * sanity checking program to aid in debugging.
	 * 
	 * The other arguments instruct the function to refrain or not from
	 * printing out grid properties (since it is useless to output the
	 * memory of a cell's neighbours if they are yet to be defined).
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
	printf("POPULATION:%i", the_grid->particle_number);
	
	// Branch if we've been told to check the cells
	if (print_cells != 0) {
		// Start a collection of cells, giving the starting address
		printf(",CELLS:STARTINGAT:%u{", the_grid->cells);
		// Go through the memory which we assume is devoted to cells
		int cell_count;
		for (cell_count = 0; cell_count < (the_grid->x_size)*(the_grid->y_size)*(the_grid->z_size); cell_count++) {
			printf("CELL:%u{", &(the_grid->cells[cell_count]));
			// Branch if we've been told to print neighbours
			if (print_neighbours != 0) {
				// Go through the neighbours
				printf("NEIGHBOURS:STARTINGAT:%u{", the_grid->cells[cell_count].neighbours);
				int neighbour_index;
				for (neighbour_index = 0; neighbour_index < 27; neighbour_index++) {
					// Print this neighbour
					// If there is not a neighbour at this location then we will have a NULL pointer
					if (the_grid->cells[cell_count].neighbours[neighbour_index] == NULL) {
						printf("LOCATION:NULL");
					}
					// Otherwise print the cell location
					else {
						printf("LOCATION:%u", the_grid->cells[cell_count].neighbours[neighbour_index]);
					}
					// We need a comma on all but the final neighbour
					if (neighbour_index < 26) {
						printf(",");
					}
				}
				printf("}");
			}
			// If we just printed the neighbours and we're going to print something else then we need a comma
			if ((print_neighbours != 0) && (print_particles != 0)) {
				printf(",");
			}
			// Branch if we have been told to print our particles
			if (print_particles != 0) {
				if (the_grid->cells[cell_count].first_particle == NULL) {
					printf("FIRSTPARTICLE:NULL");
				}
				else {
					printf("FIRSTPARTICLE:%u", the_grid->cells[cell_count].first_particle);
				}
			}
			printf("}");
			// For all but the last cell we need to insert a comma here
			if (cell_count < ((the_grid->x_size)*(the_grid->y_size)*(the_grid->z_size))-1) {
				printf(",");
			}
		}
		printf("}");
	}
	
	// Branch if we've been told to print particles
	if (print_particles != 0) {
		printf(",PARTICLES:STARTINGAT:%u{", the_grid->particles);
		int particle_count;
		for (particle_count = 0; particle_count < the_grid->particle_number; particle_count++) {
			printf("PARTICLE:%u{", &(the_grid->particles[particle_count]));
			printf("X:%G", the_grid->particles[particle_count].position.x);
			printf(",Y:%G", the_grid->particles[particle_count].position.y);
			printf(",Z:%G", the_grid->particles[particle_count].position.z);
			printf(",CONTAINER:%u", the_grid->particles[particle_count].container);
			// Branch if we've been told to print particle lists
			if (print_lists != 0) {
				printf(",NEXT:%u", the_grid->particles[particle_count].next);
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

void populate_random(grid* the_grid, int particles, int x, int y, int z, double dx, double dy, double dz) {
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
	for (particle_index = 0; particle_index < particles; particle_index++) {
		// Assign the position
		the_grid->particles[particle_index].position.x = new_random()*space.x;
		the_grid->particles[particle_index].position.y = new_random()*space.y;
		the_grid->particles[particle_index].position.z = new_random()*space.z;
		
		// Ensure that the particle's linked list pointer doesn't start in the particle memory
		the_grid->particles[particle_index].next = NULL;
	}
	
	// POSTCONDITIONS
	assert(check_particles(the_grid, particles) == 0);
	
}

int check_particles(grid* the_grid, int particle_number) {
	// PRECONDITIONS
	assert(the_grid != NULL);
	assert(particle_number >= 0);
	
	int counter;
	for (counter = 0; counter < particle_number; counter++) {
		//assert(the_grid->particles[counter] != NULL);
		assert(the_grid->particles[counter].position.x >= 0);
		assert(the_grid->particles[counter].position.y >= 0);
		assert(the_grid->particles[counter].position.z >= 0);
	}
	return 0;
}

void find_neighbours_by_brute_force(particle* current_particle, double radius, particle*** neighbour_particles, int* length, particle** set_to_check, int length_of_set) {
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
	
	*length = count_neighbours(current_particle, radius, set_to_check, length_of_set);
	//fprintf(stderr, "LLLL %i LLLL\n", no);
	neighbour_particles[0] = (particle**) malloc( *length*sizeof(particle*) );
	
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
		
		//fprintf(stderr, " %i ", particle_being_checked);
		
		// Make sure it exists
		assert(particle_being_checked != NULL);
		
		// If we've found the current particle then discard it
		// otherwise...
		if (particle_being_checked != current_particle) {
			// Get the difference in their positions
			delta_x = particle_being_checked->position.x - current_particle->position.x;
			delta_y = particle_being_checked->position.y - current_particle->position.y;
			delta_z = particle_being_checked->position.z - current_particle->position.z;
			// See whether they're within one radius of each other
			if (delta_x*delta_x + delta_y*delta_y + delta_z*delta_z <= radius*radius) {
				// If so then don't discard it, and increment the length
				neighbour_particles[0][position] = particle_being_checked;
				//fprintf(stderr, " %i ", particle_being_checked);
				position++;
				
			}
		}
	}
		
	// Now our array is populated, so we can return
	
	// POSTCONDITIONS
	assert(neighbour_particles != NULL);		// We should have a valid output array
	assert(*length <= length_of_set);		// We shouldn't output more particles than we were given
	
}



void brute_force_all(particle* current_particle, grid* the_grid, int number, double radius, particle*** found, int* length) {
	int index;
	double delta_x;
	double delta_y;
	double delta_z;
	int counter = 0;
	for (index = 0; index < number; index++) {
		// Don't bother comparing the current_particle to itself
		if (&(the_grid->particles[index]) != current_particle) {
			// Get the difference in positions
			delta_x = the_grid->particles[index].position.x - current_particle->position.x;
			delta_y = the_grid->particles[index].position.y - current_particle->position.y;
			delta_z = the_grid->particles[index].position.z - current_particle->position.z;
			// If they're within the radius then remember this pointer
			if (delta_x*delta_x + delta_y*delta_y + delta_z*delta_z <= radius*radius) {
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
			delta_x = the_grid->particles[index].position.x - current_particle->position.x;
			delta_y = the_grid->particles[index].position.y - current_particle->position.y;
			delta_z = the_grid->particles[index].position.z - current_particle->position.z;
			// If they're within the radius then remember this pointer
			if (delta_x*delta_x + delta_y*delta_y + delta_z*delta_z <= radius*radius) {
				found[0][counter] = &(the_grid->particles[index]);
				counter++;
			}
		}
	}
	
	// POSTCONDITIONS
	assert(*length <= number);
	assert(found != NULL);
	
}

int count_neighbours(particle* current_particle, double radius, particle** to_check, int length) {
	
	// PRECONDITIONS
	assert(current_particle != NULL);
	assert(radius >= 0);
	assert(to_check != NULL);
	assert(length >= 0);
	
	double delta_x, delta_y, delta_z;
	int count = 0;
	int index;
	for (index = 0; index < length; index++) {
		if (to_check[index] != current_particle) {
			delta_x = current_particle->position.x - to_check[index]->position.x;
			delta_y = current_particle->position.y - to_check[index]->position.y;
			delta_z = current_particle->position.z - to_check[index]->position.z;
			if (delta_x*delta_x + delta_y*delta_y + delta_z*delta_z <= radius*radius) {
				count++;
			}
		}
	}
	
	// POSTCONDITIONS
	assert(count >= 0);
	
	return count;
}

int neighbour_check(cell* current_cell) {
	int count = 0;
	int index;
	for (index = 0; index < 27; index++) {
		if (current_cell->neighbours[index] != NULL) {
			count++;
		}
	}
	return count;
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
	double dx = 5.0;
	double dy = 5.0;
	double dz = 5.0;
	
	// These are the dimensions of our space as multiples of dx, dy & dz
	int x = 8;
	int y = 8;
	int z = 8;
	
	// This is the total number of particles to simulate
	int particle_number = 1000;
	
	// Make the space we are simulating
	grid the_grid;
	
	the_grid.x_size = x;
	the_grid.y_size = y;
	the_grid.z_size = z;
	the_grid.particle_number = particle_number;
	
	// Allocate memory and assign neighbourhoods
	initialise_grid(&the_grid, x, y, z, particle_number);
	
	// DEBUGGING
	assert(the_grid.cells != NULL);
	assert(the_grid.particles != NULL);
	
	// Fill the space with particles at random positions
	populate_random(&the_grid, particle_number, x, y, z, dx, dy, dz);
	
	// Sort the particles into cells
	grid_particles(&the_grid, particle_number, x, y, z, dx, dy, dz);

	// DEBUGGING
	assert(the_grid.particles[0].container != NULL);
	assert(count_all_particles(&the_grid, x, y, z) == particle_number);

	dump_grid(&(the_grid), 1, 1, 1, 1);

	// Choose a particle from the group
	particle* test_particle = &(the_grid.particles[0]);

	// DEBUGGING
	assert(test_particle != NULL);
	assert(test_particle->container != NULL);

	// Find its neighbours through the grid
	particle** neighbour_array;		// This will point too an array of neighbours
	int neighbour_number;		// This will tell us how long the array is
	get_neighbours_for_particle(&the_grid, test_particle, &neighbour_array, &neighbour_number);
	
	// DEBUGGING
	assert(neighbour_number >= 0);
	
	assert(check_for_dupes(neighbour_array, neighbour_number) == 0);
	
	int neighbours_in_grid = count_neighbours(test_particle, dx, neighbour_array, neighbour_number);
	
	// Now find which of those are true neighbours
	particle** true_neighbour_array;
	int true_neighbour_number;
	find_neighbours_by_brute_force(test_particle, dx, &true_neighbour_array, &true_neighbour_number, neighbour_array, neighbour_number); 
	
	
	
	particle** all_pointers = (particle**) malloc( particle_number*sizeof(particle*) );
	int pointer_index;
	for (pointer_index = 0; pointer_index < particle_number; pointer_index++) {
		all_pointers[pointer_index] = &(the_grid.particles[pointer_index]);
	}
	
	int total_neighbours = count_neighbours(test_particle, dx, all_pointers, particle_number);
	
	assert(neighbours_in_grid == true_neighbour_number);
	
	assert(total_neighbours == neighbours_in_grid);
	
	particle** brute_force_array;

	int total_neighbours2;
	
	find_neighbours_by_brute_force(test_particle, dx, &brute_force_array, &total_neighbours2, all_pointers, particle_number);
	
	assert(total_neighbours == neighbours_in_grid);
	
	// Now "true_neighbour_array" should contain "true_neighbour_number"
	// neighbours
	
	
	int brute_force_number;
	brute_force_all(test_particle, &the_grid, particle_number, dx, &brute_force_array, &brute_force_number);
	
	// Now "brute_force_array" should contain "brute_force_number"
	
	// Compare the two
	assert(total_neighbours == true_neighbour_number);
	
	
	double delta_x, delta_y, delta_z;
	
	// Check that all of the "true neighbours" were found during brute force
	int index1;		// Used for indexing the "true" array
	int index2;		// Used for indexing the "brute" array
	int found;		// 0 means no match yet, 1 means match found
	
	int diff_count = 0;
	
	for (index1 = 0; index1 < true_neighbour_number; index1++) {
		found = 0;
		delta_x = true_neighbour_array[index1]->position.x - test_particle->position.x;
		delta_y = true_neighbour_array[index1]->position.y - test_particle->position.y;
		delta_z = true_neighbour_array[index1]->position.z - test_particle->position.z;
		
		for (index2 = 0; index2 < brute_force_number; index2++) {
			if (found == 0) {
				if (true_neighbour_array[index1] == brute_force_array[index2]) {
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
		
		delta_x = brute_force_array[index2]->position.x - test_particle->position.x;
		delta_y = brute_force_array[index2]->position.y - test_particle->position.y;
		delta_z = brute_force_array[index2]->position.z - test_particle->position.z;
		
		for (index1 = 0; index1 < true_neighbour_number; index1++) {
			if (found == 0) {
				if (true_neighbour_array[index1] == brute_force_array[index2]) {
					found = 1;
				}
			}
		}
		if (found == 0) {
			diff_count++;
			return 1;
		}
	}
	
	return 0;
}
