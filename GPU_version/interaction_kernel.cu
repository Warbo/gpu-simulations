/*
 * CUDA code for making particles interact.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "linkedlists.c"

// Each block needs access to its local particles
__shared__ particle local_particles[2*32];

// Define our kernel function
__device__ void particles_interact(particle* particle_A, particle* particle_B) {
	// Make two particles interact

	/*
	// Here we use a dummy gravity interaction
	float x_delta, y_delta, z_delta, distance, x_unit, y_unit, z_unit;
	x_delta = particle_B->x - particle_A->x;
	y_delta = particle_B->y - particle_A->y;
	z_delta = particle_B->z - particle_A->z;

	distance = sqrtf((x_delta*x_delta + y_delta*y_delta + z_delta*z_delta));
	
	x_unit = x_delta / distance;
	y_unit = y_delta / distance;
	z_unit = z_delta / distance;

	// F = (-G*m1*m2/r^2)*r_hat
	particle_A->x_acc += ((particle_A->mass * particle_B->mass)/(distance*distance))*x_unit;
	particle_A->y_acc += ((particle_A->mass * particle_B->mass)/(distance*distance))*y_unit;
	particle_A->z_acc += ((particle_A->mass * particle_B->mass)/(distance*distance))*z_unit;
	*/

	// Try doing a predictable interaction
	particle_A->x_acc += 1;
	
}

__device__ void calculate_all_neighbours(int cell_size) {
	// Calculate interactions between a particle and everything in the neighbour

	// Loop through the size of the neighbour
	for (unsigned int counter = 0; counter < cell_size; counter++) {
		// Make our particle interact with everything in the second half of
		// local memory
		particles_interact(
			&(local_particles[threadIdx.x]),
			&(local_particles[cell_size + counter])
						  );
	}
}

__device__ void load_neighbouring_cells(int cell_size) {
	// Loop through neighbours
	for (int x_rel = -1; x_rel < 2; x_rel++) {
		for (int y_rel = -1; y_rel < 2; y_rel++) {
			for (int z_rel = -1; z_rel < 2; z_rel++) {

				// Only act if we've got a valid neighbour
				if (
					(blockIdx.xl+x_rel >= 0) &&
					(blockIdx.x+x_rel < gridDim.x) &&
					(blockIdx.y+y_rel >= 0) &&
					(blockIdx.y+y_rel < gridDim.y) &&
					(blockIdx.z+z_rel >= 0) &&
					(blockIdx.z+z_rel < gridDim.z)
				) {

					// Load neighbouring cell to shared memory

					// This is the memory location we need to start allocating
					// from
					int offset = (gridDim.z * gridDim.y * (blockIdx.x + x_rel))
					+ (gridDim.z * (blockIdx.y + y_rel)) + (blockIdx.z + z_rel);

					// Each thread allocates a particle from the neighbour
					// Start at cell_size since we're filling the second half
					local_particles[cell_size + threadIdx.x] =
						all_particles[(cell_size * offset) + threadIdx.x];

					// Ensure all particles have been loaded into shared mem
					__syncthreads();

					// Work out the interactions with these neighbours
					calculate_all_neighbours(&(local_particles[count]),
						&(local_particles[cell_size]), cell_size);

					// Ensure everyone's finished before loading the next cell
					__syncthreads();
					
				}
			}
		}
	}
}

__global__ void do_cell(particle* all_particles, int cell_size) {
	// This begins calculation of the particle interactions

	// Work out where in the array our particles start
	int offset = (gridDim.z*gridDim.y*blockIdx.x) + (gridDim.z*blockIdx.y)
			+ blockIdx.z;

	// Each thread loads its own particle to local memory
	local_particles[threadIdx.x] =
		all_particles[(cell_size * offset) + threadIdx.x];

	// Initialise the interaction values
	local_particles[threadIdx.x].x_acc = 0.0;
	
	// Now load in our neighbours and calculate interactions
	load_neighbouring_cells(cell_size);
}

int main() {
	// Toy example for testing

	// Allocate room for a 1x1x1 grid with 10 particles
	particle* all_particles_host = malloc(10*sizeof(particle));

	// Give particles random positions
	for (int i=0; i< 10; i++) {
		all_particles_host[i].x = (float) (rand()/(float)(RAND_MAX));
		all_particles_host[i].y = (float) (rand()/(float)(RAND_MAX));
		all_particles_host[i].z = (float) (rand()/(float)(RAND_MAX));
		
		printf("%G, %G, %G\n", all_particles_host[i].x, all_particles_host[i].y, all_particles_host[i].z);
	}

	// Allocate memory on the GPU
	particle* all_particles_device;
	cudaMalloc((void**) &all_particles_device, 10*sizeof(particle));

	// Copy across our particles
	cudaMemcpy(all_particles_device, all_particles_host, 10*sizeof(particle), cudaMemcpyHostToDevice);

	// Calculate the interactions
	do_cell<<<1, 1>>>(all_particles_device, 1, 1, 1, 0, 0, 0, 10);

	// Get results back
	cudaMemcpy(all_particles_host, all_particles_device, 10*sizeof(particle), cudaMemcpyDeviceToHost);

	// Free up the memory
	cudaFree(all_particles_device);

	for (int i=0; i<10; i++) {
		printf("%G, %G, %G\n", all_particles_host[i].x, all_particles_host[i].y, all_particles_host[i].z);
	}

	// Exit
	return 0;
}
