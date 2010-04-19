/*
 * CUDA code for making particles interact.
 * In this version, the kernel function is one big function
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "linkedlists.c"

__global__ void do_cell(particle* all_particles, int cell_size) {
	// This begins calculation of the particle interactions

	// Each block needs access to its local particles and a neighbour
	// Note: We can't use 2*cell_size here since it's not constant!
	__shared__ particle local_particles[2*32];
	
	// Work out where in the array our particles start
	int offset = (gridDim.z*gridDim.y*blockIdx.x) + (gridDim.z*blockIdx.y)
			+ blockIdx.z;

	// Each thread loads its own particle to local memory
	local_particles[threadIdx.x] =
		all_particles[(cell_size * offset) + threadIdx.x];

	// Initialise the interaction values
	local_particles[threadIdx.x].x_acc = 0.0;
	
	// Now load in our neighbours and calculate interactions
	// Loop through neighbours
	for (int x_rel = -1; x_rel < 2; x_rel++) {
		for (int y_rel = -1; y_rel < 2; y_rel++) {
			for (int z_rel = -1; z_rel < 2; z_rel++) {

				// Only act if we've got a valid neighbour
				if (
					(((int)blockIdx.x)+x_rel >= 0) &&
								(((int)blockIdx.x)+x_rel < gridDim.x) &&
								(((int)blockIdx.y)+y_rel >= 0) &&
								(((int)blockIdx.y)+y_rel < gridDim.y) &&
								(((int)blockIdx.z)+z_rel >= 0) &&
								(((int)blockIdx.z)+z_rel < gridDim.z)
				   ) {

					// Load neighbouring cell to shared memory

					// This is the memory location we need to start allocating
					// from
					int offset = (gridDim.z * gridDim.y * (blockIdx.x + x_rel))
							+ (gridDim.z * (blockIdx.y + y_rel))
							+ (blockIdx.z + z_rel);

					// Each thread allocates a particle from the neighbour
					// Start at cell_size since we're filling the second half
					local_particles[cell_size + threadIdx.x] =
							all_particles[(cell_size * offset) + threadIdx.x];

					// Ensure all particles have been loaded into shared mem
					__syncthreads();

					// Work out the interactions with these neighbours
					// Calculate interactions between a particle and everything
					// in the neighbour

					// Loop through the size of the neighbour
					for (unsigned int counter = 0; counter < cell_size;
						counter++) {
						// Make our particle interact with everything in the
						// second half of local memory

						// Try doing a predictable interaction
						//particle_A->x_acc += 1;
						local_particles[threadIdx.x].x_acc += 1;
						//&(local_particles[cell_size + counter])
					}

					// Ensure everyone's finished before loading the next cell
					__syncthreads();
					
				   }
			}
		}
	}

	// Now put shared values back into global memory
	all_particles[(cell_size * offset) + threadIdx.x] =
		local_particles[threadIdx.x];

	// Make sure we're all done
	__syncthreads();
	
}

int main() {
	// Toy example for testing

	// Allocate room for a 3x3x3 grid with 32 particles each
	particle* all_particles_host = (particle*)malloc(27*32*sizeof(particle));

	// Give particles random positions
	for (int i=0; i < 27*32; i++) {
		all_particles_host[i].x = (float) (rand()/(float)(RAND_MAX));
		all_particles_host[i].y = (float) (rand()/(float)(RAND_MAX));
		all_particles_host[i].z = (float) (rand()/(float)(RAND_MAX));
	}

	// Allocate memory on the GPU
	particle* all_particles_device;
	cudaMalloc((void**) &all_particles_device, 27*32*sizeof(particle));

	// Copy across our particles
	cudaMemcpy(all_particles_device, all_particles_host, 27*32*sizeof(particle),
			   cudaMemcpyHostToDevice);

	dim3 dimGrid(3, 3, 3);
	// Calculate the interactions
	do_cell<<<dimGrid, 32>>>(all_particles_device, 32);

	// Get results back
	cudaMemcpy(all_particles_host, all_particles_device, 27*32*sizeof(particle),
cudaMemcpyDeviceToHost);

	// Free up the memory
	cudaFree(all_particles_device);

	for (int i=0; i<27*32; i++) {
		printf("%G\n", all_particles_host[i].x_acc);
	}

	// Exit
	return 0;
}
