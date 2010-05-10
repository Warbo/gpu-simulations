/*
 * Contains all code specific to the use of sentinels in memory.
 * The structure of the grid's particle storage in this case is:
 * S,(0,0,0),S,(0,0,1),S,(0,0,2)...S,(0,1,0),S,(0,1,1)...S,(1,0,0),S,(1,0,1)...
 * Where S is a sentinel particle, with ID -1, and (x,y,z) represents the
 * particles in cell (x,y,z) in arbitrary order.
 *
 * This allows efficient use of memory, since the overhead is linear with the
 * grid size. However, it makes accessing specific cells time consuming, as the
 * sentinels need to be counted beginning at zero, so lookup is O(N)
 */

// Grab everything that's not specific to sentinels
#include "common_functions."

int get_sentinel_number(grid* the_grid) {
	/*
	* Counts the number of sentinel particles in the given grid.
	*/
	
	// Preconditions
	assert(the_grid != NULL);
	assert(the_grid->particle_number >= 0);
	assert(the_grid->particles != NULL);
	
	int sentinel_count;
	sentinel_count = 0;
	int index;
	for (index=0; index < the_grid->particle_number +
			(the_grid->x_size * the_grid->y_size * the_grid->z_size);
			index++) {
			
				if (the_grid->particles[index].id == -1) {
					sentinel_count++;
				}
			}
	
	// Postconditions
			assert(sentinel_count >= 0);
	
			return sentinel_count;
}

int count_cell_contents(grid* the_grid, int x, int y, int z) {
	/*
	* This returns the number of particles in the cell at the
	* given location in the given grid.
	*/

	// PRECONDITIONS
	assert(the_grid != NULL);
	assert(x >= 0);
	assert(y >= 0);
	assert(z >= 0);
	assert(x < the_grid->x_size);
	assert(y < the_grid->y_size);
	assert(z < the_grid->z_size);
	
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
			  (the_grid->x_size * the_grid->y_size * the_grid->z_size)){

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
	
	// POSTCONDITIONS
			  assert(sentinel_count >= 0);
	
			  return particle_count;
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
			   (particle*) malloc( ((unsigned int)(particles+x*y*z)) *
			   sizeof(particle));
	
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

void grid_particles(grid* the_grid, particle* particles, int particle_number,
	float size) {
	/*
	 * Sets up a grid at 'the_grid' and puts the contents of 'particles', of
	 * length 'particle_number', into it, each with size 'size'.
	 */
	
	// PRECONDITIONS
	assert(the_grid != NULL);
	assert(particles != NULL);
	assert(particle_number > 0);
	assert(size > (float)0.0);

	// Get grid parameters needed to store these particles
	float x_min, x_max, y_min, y_max, z_min, z_max;
	float x_offset, y_offset, z_offset;
	int x_size, y_size, z_size;
	get_extents(particles, particle_number, &x_min, &x_max, &y_min, &y_max,
		&z_min, &z_max);
	grid_size(x_min, x_max, y_min, y_max, z_min, z_max, size, size, size,
		&x_size, &y_size, &z_size, &x_offset, &y_offset, &z_offset);

	// DEBUG
	assert(x_size > 0);
	assert(y_size > 0);
	assert(z_size > 0);
	assert(x_min - x_offset > (float)0.0);
	assert(x_max - x_offset > (float)0.0);
	assert(y_min - y_offset > (float)0.0);
	assert(y_max - y_offset > (float)0.0);
	assert(z_min - z_offset > (float)0.0);
	assert(z_max - z_offset > (float)0.0);
	
	// Make the grid
	initialise_grid(the_grid, x_size, y_size, z_size, size, size, size,
		x_offset, y_offset, z_offset, particle_number);

	// DEBUG
	assert(the_grid->x_size == x_size);
	
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

				// To begin with, add a sentinel
				the_grid->particles[array_index].id = -1;
				array_index++;

				// For each cell, loop through every given particle
				for (index = 0; index < the_grid->particle_number; index++) {
					// Initialise particle's position to a nonsense value
					px = -1;
					py = -1;
					pz = -1;

					// If the particle is in this cell, add it to the grid
					get_index_from_position(the_grid, &(particles[index]),
						&px, &py, &pz);
					
					// DEBUG
					assert(px >= 0);
					assert(py >= 0);
					assert(pz >= 0);
					
					if (x == px && y == py && z == pz) {
						
						// DEBUG
						assert(array_index < the_grid->particle_number +
							(the_grid->x_size * the_grid->y_size *
								the_grid->z_size));
						
						the_grid->particles[array_index] = particles[index];
						
						// TODO: Do we want to assign IDs here or in an
						// input file?
						the_grid->particles[array_index].id = array_index;
						
						array_index++;
					}
				}
			
			}
		}
	}
	
	// POSTCONDITIONS
	assert(get_sentinel_number(the_grid) ==
		the_grid->x_size * the_grid->y_size * the_grid->z_size);
	assert((array_index == the_grid->particle_number +
			get_sentinel_number(the_grid)) ||
		(array_index == the_grid->particle_number +
			get_sentinel_number(the_grid) + 1));

}

void get_cell_contents(grid* the_grid, int x, int y, int z,
	particle** start, int* length) {
	/*
	 * Assigns the pointer 'start' to an array of particles, the size of
	 * which is stored in 'length'. The contents of the array are those
	 * particles in 'the_grid' at the grid cell with coordinates (x,y,z)
	 */
	// PRECONDITIONS
	assert(the_grid != NULL);
	assert(x >= 0);
	assert(y >= 0);
	assert(z >= 0);
	assert(x < the_grid->x_size);
	assert(y < the_grid->y_size);
	assert(z < the_grid->z_size);
	assert(length != NULL);
	assert(start != NULL);
	
	// Initialise variables
	int index = 0;		// Stores where we're up to in the particle array
	int sentinel_count = 0;
	int start_sentinel = (the_grid->y_size * the_grid->z_size)*x +
		(the_grid->z_size)*y + z;
	*start = NULL;

	// Initialise our length counter
	*length = 0;

	// Loop through the particles
	while ((index < the_grid->particle_number +
		(the_grid->x_size * the_grid->y_size * the_grid->z_size)) &&
			(sentinel_count < start_sentinel + 2)) {

		// If we've found a sentinel, increment our sentinel counter
		if (the_grid->particles[index].id == -1) {
			// If we've just found the right cell, set the start to the next
			// particle
			if (sentinel_count == start_sentinel) {
				*start = &(the_grid->particles[index+1]);
			}
			
			sentinel_count++;
		}

		// Otherwise, if we're in the right cell, increment the particle counter
		else if (*start != NULL) {
			*length = *length + 1;
		}

		// Move on to the next particle
		index++;
	}

	// POSTCONDITIONS
	assert(*length >= 0);
	assert(*start != NULL);

}

