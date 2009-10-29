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
	 * linked list of the particles it contains, as well as pointers to
	 * each of its neighbours in the 3D grid (including diagonals).
	 */
	
	/*
	 * The neighbours are stored in an array, and are easily accessible
	 * based on their relative position by using the following trinary
	 * 'bitmask': 9(x+1) + 3(y+1) + (z+1). This allows us to uniquely
	 * index all 26 neighbours, along with ourselves, whilst making any
	 * all-neighbour loops straightforward (iterate through the array).
	 */
	cell** neighbours;
	
	// This points to the first particle in our linked list
	particle* first_particle;
	
};

struct grid {
	/*
	 * A grid represents the space being simulated. It stores cells
	 * which in turn store particles.
	 */
	
	// The dimensions are stored here
	int x_size;
	int y_size;
	int z_size;
	
	// And the particle number
	int particle_number;
	
	// This stores the volume units of this grid
	cell* cells;
	
	// This stores the particles which the cells contain
	particle* particles;
	
};

void dump_grid(grid* current_grid, int print_cells, int print_particles, int print_neighbours, int print_lists) {
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
	assert (current_grid != NULL);
	
	// Start with the grid, saying that we're a grid and our address
	printf("GRID:%u{", current_grid);
	// Now give our size
	printf("X:%i,Y:%i,Z:%i,", current_grid->x_size, current_grid->y_size, current_grid->z_size);
	// And our population
	printf("POPULATION:%i", current_grid->particle_number);
	
	// Branch if we've been told to check the cells
	if (print_cells != 0) {
		// Start a collection of cells, giving the starting address
		printf(",CELLS:STARTINGAT:%u{", current_grid->cells);
		// Go through the memory which we assume is devoted to cells
		int cell_count;
		for (cell_count = 0; cell_count < (current_grid->x_size)*(current_grid->y_size)*(current_grid->z_size); cell_count++) {
			printf("CELL:%u{", &(current_grid->cells[cell_count]));
			// Branch if we've been told to print neighbours
			if (print_neighbours != 0) {
				// Go through the neighbours
				printf("NEIGHBOURS:STARTINGAT:%u{", current_grid->cells[cell_count].neighbours);
				int neighbour_index;
				for (neighbour_index = 0; neighbour_index < 27; neighbour_index++) {
					// Print this neighbour
					// If there is not a neighbour at this location then we will have a NULL pointer
					if (current_grid->cells[cell_count].neighbours[neighbour_index] == NULL) {
						printf("LOCATION:NULL");
					}
					// Otherwise print the cell location
					else {
						printf("LOCATION:%u", current_grid->cells[cell_count].neighbours[neighbour_index]);
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
				if (current_grid->cells[cell_count].first_particle == NULL) {
					printf("FIRSTPARTICLE:NULL");
				}
				else {
					printf("FIRSTPARTICLE:%u", current_grid->cells[cell_count].first_particle);
				}
			}
			printf("}");
			// For all but the last cell we need to insert a comma here
			if (cell_count < ((current_grid->x_size)*(current_grid->y_size)*(current_grid->z_size))-1) {
				printf(",");
			}
		}
		printf("}");
	}
	
	// Branch if we've been told to print particles
	if (print_particles != 0) {
		printf(",PARTICLES:STARTINGAT:%u{", current_grid->particles);
		int particle_count;
		for (particle_count = 0; particle_count < current_grid->particle_number; particle_count++) {
			printf("PARTICLE:%u{", &(current_grid->particles[particle_count]));
			printf("X:%G", current_grid->particles[particle_count].position.x);
			printf(",Y:%G", current_grid->particles[particle_count].position.y);
			printf(",Z:%G", current_grid->particles[particle_count].position.z);
			printf(",CONTAINER:%u", current_grid->particles[particle_count].container);
			// Branch if we've been told to print particle lists
			if (print_lists != 0) {
				printf(",NEXT:%u", current_grid->particles[particle_count].next);
			}
			printf("}");
			if (particle_count < (current_grid->particle_number) - 1) {
				printf(",");
			}
		}
		printf("}");
	}
	
	printf("}");
}
	

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
				//current_grid->cells[(x_index*y*z)+(y_index*z)+(z_index)] = a_cell;
					
				// We need to allocate some memory for each cell's neighbour pointers
				current_grid->cells[(y*z)*x_index+(z)*y_index+z_index].neighbours = (cell**) malloc(27*sizeof(cell*));
			
				// We need to make sure our particle list pointer is definitely outside the particle list to begin with
				current_grid->cells[(y*z)*x_index+(z)*y_index+z_index].first_particle = NULL;
				
				// Assign our neighbours
				for (x_rel = -1; x_rel < 2; x_rel++) {
					for (y_rel = -1; y_rel < 2; y_rel++) {
						for (z_rel = -1; z_rel < 2; z_rel++) {
							// Only act if the current x, y and z positions allow a neighbour
							if ((x_index + x_rel >= 0) && (x_index + x_rel < x) && (y_index + y_rel >= 0) && (y_index + y_rel < y) && (z_index + z_rel >= 0) && (z_index + z_rel < z)) {
								// Do some sanity checks
								assert((y*z)*x_index+(z)*y_index+z_index < x*y*z);		// Is the current cell inside the cell list?
								assert((9*(x_rel+1))+(3*(y_rel+1))+z_rel+1 < 27);		// Is the current neighbour position inside the neighbour list?
								assert((y*z)*(x_index+x_rel)+(z)*(y_index+y_rel)+(z_index+z_rel) < x*y*z);		// Would the neighbouring cell be inside the cell list?
	
								// Assign this neighbour
								current_grid->cells[(y*z)*x_index+(z)*y_index+z_index].neighbours[(9*(x_rel+1))+(3*(y_rel+1))+z_rel+1] = &(current_grid->cells[(y*z)*(x_index+x_rel)+(z)*(y_index+y_rel)+(z_index+z_rel)]);
								
								//fprintf(stderr, "cell %i, neighbour %i %i %i is cell %i\n", ((y*z)*x_index+(z)*y_index+z_index) - (int)&(current_grid->cells), x_rel, y_rel, z_rel, &(current_grid->cells[(y*z)*x_index+(z)*y_index+z_index].neighbours[(9*(x_rel+1))+(3*(y_rel+1))+z_rel+1]));
							}
							else {
								assert((y*z)*x_index+(z)*y_index+z_index < x*y*z);
								assert((9*(x_rel+1))+(3*(y_rel+1))+z_rel+1 < 27);
								
								current_grid->cells[(y*z)*x_index+(z)*y_index+z_index].neighbours[(9*(x_rel+1))+(3*(y_rel+1))+z_rel+1] = NULL;
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

double new_random() {
	/*
	 * Returns a random double between 0 and 0.999...
	 * NOTE: You may wish to change the random seed first using srand()!
	 */
	 return rand()/(double)(RAND_MAX);
}

void populate_random(grid* current_grid, int particles, int x, int y, int z, double dx, double dy, double dz) {
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
	assert(current_grid != NULL);
	
	// Store the total size of the space
	three_vector space;
	space.x = ((double) x) * dx;
	space.y = ((double) y) * dy;
	space.z = ((double) z) * dz;
	
	// Loop through the particle memory
	int particle_index;
	for (particle_index = 0; particle_index < particles; particle_index++) {
		// Assign the position
		current_grid->particles[particle_index].position.x = new_random()*space.x;
		current_grid->particles[particle_index].position.y = new_random()*space.y;
		current_grid->particles[particle_index].position.z = new_random()*space.z;
		
		// Ensure that the particle's linked list pointer doesn't start in the particle memory
		current_grid->particles[particle_index].next = NULL;
	}
	
	// POSTCONDITIONS
	assert(check_particles(current_grid, particles) == 0);
	
}

int check_particles(grid* current_grid, int particle_number) {
	// PRECONDITIONS
	assert(current_grid != NULL);
	assert(particle_number >= 0);
	
	int counter;
	for (counter = 0; counter < particle_number; counter++) {
		//assert(current_grid->particles[counter] != NULL);
		assert(current_grid->particles[counter].position.x >= 0);
		assert(current_grid->particles[counter].position.y >= 0);
		assert(current_grid->particles[counter].position.z >= 0);
	}
	return 0;
}
	
void put_particle_in_cell(particle* current_particle, cell* intended_cell) {
	/*
	 * This checks whether the given cell already contains a particle.
	 * If so then the list is traversed to the end and the given
	 * particle is appended. If not then the given particle is turned
	 * into the head of the list.
	 */
	
	// PRECONDITIONS
	assert(current_particle != NULL);
	assert(intended_cell != NULL);
	
	// First of all, tell the particle its cell
	current_particle->container = intended_cell;
	
	// Now see if the cell is empty
	if (intended_cell->first_particle == NULL) {
		// If so then put in the given particle and finish
		intended_cell->first_particle = current_particle;
		return;
	}
	// Otherwise we'll need to traverse the list, so keep track of
	// where we are
	particle* current_list_particle = intended_cell->first_particle;
	
	// Loop until we're at the end
	while (current_list_particle->next != NULL) {
		// Go along one particle
		current_list_particle = current_list_particle->next;
	}
	
	// Now append the given particle
	current_list_particle->next = current_particle;
	
	// POSTCONDITIONS
	assert(intended_cell->first_particle != NULL);
	assert(current_particle->container != NULL);
	
	return;
}

void grid_particles(grid* current_grid, int particles, int x, int y, int z, double dx, double dy, double dz) {
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
	assert(current_grid != NULL);
	
	int x_position, y_position, z_position;
	int particle_index;
	cell* intended_cell;
	for (particle_index = 0; particle_index < particles; particle_index++) {
		x_position = get_index(current_grid->particles[particle_index].position.x, dx);
		y_position = get_index(current_grid->particles[particle_index].position.y, dy);
		z_position = get_index(current_grid->particles[particle_index].position.z, dz);
		intended_cell = &(current_grid->cells[(y*z)*x_position+(z)*y_position+z_position]);
		put_particle_in_cell(&(current_grid->particles[particle_index]), intended_cell);
	}
	
	// POSTCONDITIONS
	
}

void get_neighbours_for_particle(particle* current_particle, particle*** neighbour_particles, int* length) {
	/*
	 * This function will find every neighbour particle for the given
	 * particle, checking its cell and its neighbouring cells.
	 * Pointers to these particles are stored at "neighbour_particles",
	 * which is allocated in this function. The amount of memory
	 * allocated will be stored in "length".
	 */
	
	// PRECONDITIONS
	assert(current_particle != NULL);
	assert(neighbour_particles != NULL);
	assert(length != NULL);
	assert(current_particle->container != NULL);
	
	// We run two loops here, where we could cache the particle numbers
	// in the cells instead, but this way makes things simpler by
	// keeping related operations together
	
	// First count the neighbours
	*length = 0;		// Start our counter at 0
	particle* found_particle;		// This will store the particle we're up to
	
	// Iterate through the given particle's cell's neighbours
	// NOTE: This loop includes the particle's cell for us too
	int neighbour_index;
	for (neighbour_index = 0; neighbour_index < 27; neighbour_index++) {
		if (current_particle->container->neighbours[neighbour_index] != NULL) {
			assert(current_particle != NULL);
			assert(current_particle->container != NULL);
			assert(current_particle->container->neighbours != NULL);
			assert(current_particle->container->neighbours[neighbour_index] != NULL);
		
			// Get this neighbour's first particle
			found_particle = current_particle->container->neighbours[neighbour_index]->first_particle;
		
			// As long as there are still particles in this cell, loop
			while (found_particle != NULL) {
				// Increment the counter as long as we're not counting the
				// given particle
				if (found_particle != current_particle) {
					*length = *length + 1;
				}
				found_particle = found_particle->next;
			}
		}
	}
	
	// Now we know how many neighbours we have, let's allocate memory
	neighbour_particles[0] = (particle**) malloc( (*length)*sizeof(particle*) );

	// Now loop through the neighbours again, storing pointers to them
	*length = 0;		// We can re-use this as our index
	found_particle = NULL;
	for (neighbour_index = 0; neighbour_index < 27; neighbour_index++) {
		if (current_particle->container->neighbours[neighbour_index] != NULL) {
			found_particle = current_particle->container->neighbours[neighbour_index]->first_particle;
			while (found_particle != NULL) {
				if (found_particle != current_particle) {
					(neighbour_particles[0])[(*length)] = found_particle;
					*length = *length + 1;
				}
				found_particle = found_particle->next;
			}
		}
	}
	
	// Now "neighbours" should point to an array of particle pointers
	// which are the neighbouring particles, and length should point to
	// the length, so we're done
	
	// POSTCONDITIONS
	//assert();
	
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

void brute_force_all(particle* current_particle, grid* current_grid, int number, double radius, particle*** found, int* length) {
	int index;
	double delta_x;
	double delta_y;
	double delta_z;
	int counter = 0;
	for (index = 0; index < number; index++) {
		// Don't bother comparing the current_particle to itself
		if (&(current_grid->particles[index]) != current_particle) {
			// Get the difference in positions
			delta_x = current_grid->particles[index].position.x - current_particle->position.x;
			delta_y = current_grid->particles[index].position.y - current_particle->position.y;
			delta_z = current_grid->particles[index].position.z - current_particle->position.z;
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
		if ((&current_grid->particles[index]) != current_particle) {
			// Get the difference in positions
			delta_x = current_grid->particles[index].position.x - current_particle->position.x;
			delta_y = current_grid->particles[index].position.y - current_particle->position.y;
			delta_z = current_grid->particles[index].position.z - current_particle->position.z;
			// If they're within the radius then remember this pointer
			if (delta_x*delta_x + delta_y*delta_y + delta_z*delta_z <= radius*radius) {
				found[0][counter] = &(current_grid->particles[index]);
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

int count_all_particles(grid* current_grid, int x, int y, int z) {
	
	// PRECONDITIONS
	assert(current_grid != NULL);
	assert(x > 0);
	assert(y > 0);
	assert(z > 0);
	
	int count = 0;
	int index;
	particle* current_particle;
	for (index=0; index < x*y*z; index++) {
		current_particle = current_grid->cells[index].first_particle;
		while (current_particle != NULL) {
			current_particle = current_particle->next;
			count++;
		}
	}
	return count;
}

int check_for_dupes(particle** particles, int length) {
	
	// PRECONDITIONS
	assert(particles != NULL);
	assert(length >= 0);
	
	if (length == 0) {
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
	int x = 5;
	int y = 5;
	int z = 5;
	
	// This is the total number of particles to simulate
	int particle_number = 1000;
	
	// Make the space we are simulating
	grid current_grid;
	
	current_grid.x_size = x;
	current_grid.y_size = y;
	current_grid.z_size = z;
	current_grid.particle_number = particle_number;
	
	// Allocate memory and assign neighbourhoods
	initialise_grid(&current_grid, x, y, z, particle_number);
	
	// DEBUGGING
	assert(current_grid.cells != NULL);
	assert(current_grid.particles != NULL);
	
	// Fill the space with particles at random positions
	populate_random(&current_grid, particle_number, x, y, z, dx, dy, dz);
	
	// Sort the particles into cells
	grid_particles(&current_grid, particle_number, x, y, z, dx, dy, dz);

	// This will dump the freshly populated grid
	//dump_grid(&current_grid, 1, 1, 1, 1);

	// DEBUGGING
	assert(current_grid.particles[0].container != NULL);
	assert(count_all_particles(&current_grid, x, y, z) == particle_number);

	// Choose a particle from the group
	particle* test_particle = &(current_grid.particles[0]);

	// DEBUGGING
	assert(test_particle != NULL);
	assert(test_particle->container != NULL);

	// Print out its position
	fprintf(stderr, "Chosen particle position: %G, %G, %G\n", test_particle->position.x, test_particle->position.y, test_particle->position.z);

	fprintf(stderr, "Getting neighbours: ");

	// Find its neighbours through the grid
	particle** neighbour_array;		// This will point too an array of neighbours
	int neighbour_number;		// This will tell us how long the array is
	get_neighbours_for_particle(test_particle, &neighbour_array, &neighbour_number);
	
	fprintf(stderr, "DEBUG:(%G, %G, %G) has %i potential neighbours\n", test_particle->position.x, test_particle->position.y, test_particle->position.z, neighbour_number);
	
	// DEBUGGING
	fprintf(stderr, "NN:%i ", neighbour_number);
	assert(neighbour_number >= 0);
	
	assert(check_for_dupes(neighbour_array, neighbour_number) == 0);
	
	int neighbours_in_grid = count_neighbours(test_particle, dx, neighbour_array, neighbour_number);
	
	// Now find which of those are true neighbours
	particle** true_neighbour_array;
	int true_neighbour_number;
	find_neighbours_by_brute_force(test_particle, dx, &true_neighbour_array, &true_neighbour_number, neighbour_array, neighbour_number); 
	
	fprintf(stderr, "True neighbours through grid: %i\n", true_neighbour_number);
	
	
	
	particle** all_pointers = (particle**) malloc( particle_number*sizeof(particle*) );
	int pointer_index;
	for (pointer_index = 0; pointer_index < particle_number; pointer_index++) {
		all_pointers[pointer_index] = &(current_grid.particles[pointer_index]);
	}
	
	int total_neighbours = count_neighbours(test_particle, dx, all_pointers, particle_number);
	
	assert(neighbours_in_grid == true_neighbour_number);
	fprintf(stderr, "T:%i, N:%i ", total_neighbours, neighbours_in_grid);
	assert(total_neighbours == neighbours_in_grid);
	
	particle** brute_force_array;

	int total_neighbours2;
	
	find_neighbours_by_brute_force(test_particle, dx, &brute_force_array, &total_neighbours2, all_pointers, particle_number);
	
	fprintf(stderr, "grid1: %i, grid2: %i, total1: %i, total2: %i\n", neighbours_in_grid, true_neighbour_number, total_neighbours, total_neighbours2); 
	
	assert(total_neighbours == neighbours_in_grid);
	
	// Now "true_neighbour_array" should contain "true_neighbour_number"
	// neighbours
	printf("%i\n", true_neighbour_number);
	
	printf("Getting neighbours through brute force: ");
	
	
	int brute_force_number;
	brute_force_all(test_particle, &current_grid, particle_number, dx, &brute_force_array, &brute_force_number);
	
	// Now "brute_force_array" should contain "brute_force_number"
	printf("%i\n", total_neighbours);
	
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
		// See whether they're within one radius of each other
		//fprintf(stderr, "%G ", delta_x*delta_x + delta_y*delta_y + delta_z*delta_z);
		
		for (index2 = 0; index2 < brute_force_number; index2++) {
			if (found == 0) {
				if (true_neighbour_array[index1] == brute_force_array[index2]) {
					found = 1;
				}
			}
		}
		if (found == 0) {
			fprintf(stderr, "Particle in 'true' array not found through brute force!\n");
			diff_count++;
			return 1;
		}
	}
	
	fprintf(stderr, "\nDIFF: %i\n", diff_count);
	
	diff_count = 0;
	
	// Now do the same but the other way around
	for (index2 = 0; index2 < brute_force_number; index2++) {
		found = 0;
		
		delta_x = brute_force_array[index2]->position.x - test_particle->position.x;
		delta_y = brute_force_array[index2]->position.y - test_particle->position.y;
		delta_z = brute_force_array[index2]->position.z - test_particle->position.z;
		// See whether they're within one radius of each other
		//printf("%G ", delta_x*delta_x + delta_y*delta_y + delta_z*delta_z);
		
		//printf("(%G, %G, %G) ", brute_force_array[index2]->position.x, brute_force_array[index2]->position.y, brute_force_array[index2]->position.z);
		
		for (index1 = 0; index1 < true_neighbour_number; index1++) {
			if (found == 0) {
				if (true_neighbour_array[index1] == brute_force_array[index2]) {
					found = 1;
				}
			}
		}
		if (found == 0) {
			diff_count++;//printf("Particle from brute force search not found in 'true' array!\n");
			return 1;
		}
	}
	
	fprintf(stderr, "Difference of %i\n", diff_count);
	
	// Comment this out until it's true :P
	printf("Success!\n");
	
	return 0;
}
