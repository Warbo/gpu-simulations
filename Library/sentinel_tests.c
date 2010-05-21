/*
 * Stores tests for the sentinels implementation.
 */

// Drag in everything we need
#include "sentinel_functions.c"

void interact(particle* p1, particle* p2) {
	/*
	 * Placeholder interaction.
	 */

	// PRECONDITIONS
	assert(p1 != NULL);
	assert(p2 != NULL);

	p2 = p2;
	p1->x_acc = p1->x_acc + 1;

	// POSTCONDITIONS
	assert(p1->x_acc > 0);
}

int main() {
	/*
	 * The tests themselves.
	 */
	particle* p_array;
	int particle_number;
	float size;
	read_particles(&p_array, &particle_number, &size);

	// DEBUG
	assert(size > (float)0.0);
	assert(particle_number > 0);
	
	grid the_grid;
	grid_particles(&the_grid, p_array, particle_number, size);

	// Clean up
	free(p_array);

	// Swap comments to switch between brute-force and grid
	particle* test_particle = NULL;
	//int test_x, test_y, test_z;
	//int neighbour_number = -1;
	int true_neighbour_number = -1;
	//particle* brute_force_neighbours;
	//int total_neighbours = -1;
	//particle* neighbour_array;
	particle* true_neighbour_array;
	//float delta_x, delta_y, delta_z;
	int index1;		// Used for indexing the "true" array
	//int index2;		// Used for indexing the "brute" array
	//int found;		// 0 means no match yet, 1 means match found
	int particle_index;

	// Loop through every particle
	for (particle_index=0; particle_index < the_grid.particle_number + (
		the_grid.x_size * the_grid.y_size * the_grid.z_size
	); particle_index++) {
	// Only act if we're not on a sentinel
	if (the_grid.particles[particle_index].id >= 0) {

		test_particle = &(the_grid.particles[particle_index]);
			
	

		// DEBUGGING ///////////////////////////////////////////////////////////////
		
		//get_index_from_position(&the_grid, test_particle,
		//	&test_x, &test_y, &test_z);
		//assert(test_x >= 0);
		//assert(test_y >= 0);
		//assert(test_z >= 0);
		//assert(test_x < the_grid.x_size);
		//assert(test_y < the_grid.y_size);
		//assert(test_z < the_grid.z_size);
		////////////////////////////////////////////////////////////////////////////

		// DEBUGGING
		assert(test_particle != NULL);
		assert(test_particle->id != -1);

		// Find its neighbours through the grid
		// This will point to an array of neighbours
		
		// This will tell us how long the array is
		
		//get_potential_neighbours_for_particle(&the_grid, test_particle,
		//	&neighbour_array, &neighbour_number);

		// DEBUGGING
		//assert(neighbour_number >= 0);

		//fprintf(stderr, "Potentials found: %i\n", neighbour_number);
	
		// Now find which of those are true neighbours
		get_true_neighbours_for_particle(&the_grid, test_particle,
			&true_neighbour_array, &true_neighbour_number);

		// Interact with neighbours
		for (index1=0; index1<true_neighbour_number; index1++) {
			interact(test_particle, &(true_neighbour_array[index1]));
		}

		// Everything below is debugging
		
		//fprintf(stderr, "True neighhbours: %i\n", true_neighbour_number);

		
		//get_neighbours_brute_force(&the_grid, test_particle,
		//	&(brute_force_neighbours), &total_neighbours);
	
		//fprintf(stderr, "True neighbours from all: %i\n", total_neighbours);

		//assert(total_neighbours == true_neighbour_number);
		//assert(total_neighbours < neighbour_number);

		// Check every particle found
		
	
		// Check that all of the "true neighbours" were found during 
		// brute force
		
	/*
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
	*/
		// Clean up
		//free(brute_force_neighbours);
		//free(neighbour_array);
		free(true_neighbour_array);
		
		}
	}
	// Success
	return 0;
	
}

