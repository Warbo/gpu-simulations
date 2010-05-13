/*
 * Contains tests for the CUDA implementation.
 */

// Drag in what we need
#include "cuda_functions.cu"

int main() {
	/*
	 * The tests themselves.
	 */

	// How many cycles to perform
	int interaction_number = 1;
	
	// Read in our particles
	int particle_number;
	particle* p_array;
	float particle_size;
	read_particles(&p_array, &particle_number, &particle_size);
	
	// Cell maximum size
	grid the_grid;
	grid_particles(&the_grid, p_array, particle_number, particle_size);
	int cell_size = get_biggest_cell_size(&the_grid, p_array);

	// The cell size used by CUDA must be #defined at pre-processor time, so we
	// can't change it once we've got to here. If it's not big enough, give an
	// error and exit.
	if (cell_size > CELLSIZE) {
		fprintf(stderr,
			"Error: Cells need at least %i particles, only got %i\n",
			cell_size, CELLSIZE);
		fprintf(stderr, "Change #define CELLSIZE in cuda_functions.cu\n");
		return 1;
	}
	
	// Allocate room for a padded grid
	particle* all_particles_host = (particle*)malloc((unsigned int)(
		the_grid.x_size * the_grid.y_size * the_grid.z_size * CELLSIZE
	) * sizeof(particle));

	// Allocate memory on the GPU
	particle* all_particles_device;
	cudaMalloc(
		(void**)&all_particles_device,
		(the_grid.x_size * the_grid.y_size * the_grid.z_size) *
			CELLSIZE * sizeof(particle)
	);

	// Copy across our particles
	cudaMemcpy(all_particles_device, all_particles_host,
		(the_grid.x_size * the_grid.y_size * the_grid.z_size) *
			CELLSIZE * sizeof(particle),
		cudaMemcpyHostToDevice
	);

	dim3 dimGrid(
		the_grid.x_size * the_grid.y_size * the_grid.z_size
	);

	// Run the interactions
	/*int index;
	for (index=0; index < interaction_number; index++) {
		// Calculate the interactions
		do_cell<<<dimGrid, CELLSIZE>>>(all_particles_device, CELLSIZE,
			the_grid.x_size, the_grid.y_size, the_grid.z_size);
	}*/

	// Get results back
	cudaMemcpy(all_particles_host, all_particles_device,
		(the_grid.x_size * the_grid.y_size * the_grid.z_size) *
			CELLSIZE * sizeof(particle),
		cudaMemcpyDeviceToHost
	);

	// Free up the memory
	cudaFree(all_particles_device);

	// DEBUG
	/*for (index=0; index <
		(the_grid.x_size * the_grid.y_size * the_grid.z_size) * CELLSIZE;
	index++) {
		printf("%G\n", all_particles_host[index].x_acc);
	}*/

	// Exit
	return 0;
}
