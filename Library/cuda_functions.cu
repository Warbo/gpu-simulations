/*
 * CUDA code for making particles interact.
 * In this version, the kernel function is one big function
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "pair_array_functions.c"

#define CELLSIZE (64)

__device__ int get_global_offset(int bIdx_x, int grid_x, int grid_y,
	int grid_z) {
	// This gets the offset of the current thread in the global particle array
	/*return (int)(cell_size * (
		(bIdx_y * gDim_x*(bDim_x*bDim_y*bDim_z)) +
		(bIdx_x * (bDim_x*bDim_y*bDim_z)) +
		tIdx_z * (bDim_x*bDim_y) +
		tIdx_y * (bDim_x) +
		tIdx_x));*/
	int z = bIdx_x % grid_z;
	int y = ((bIdx_x - z) % (grid_z * grid_y)) / grid_z;
	int x = (bIdx_x - z - (grid_z * y)) / (grid_z * grid_y);
	return z+grid_z*y+(grid_z*grid_y)*x;
}

__device__ int current_global_offset(int grid_x, int grid_y, int grid_z) {
	return get_global_offset((int)blockIdx.x, grid_x, grid_y,
		grid_z);
}

__device__ int neighbour_offset(int x_rel, int y_rel, int z_rel, int grid_x,
	int grid_y, int grid_z) {
	// Gets the offset of a neighbour at relative position (x,y,z)
	// NOTE: Implementation-specific, assuming 1D grid of 1D blocks!
	int z = ((int)blockIdx.x) % grid_z;
	int y = ((((int)blockIdx.x) - z) % (grid_z * grid_y)) / grid_z;
	int x = (((int)blockIdx.x) - z - (grid_z * y)) / (grid_z * grid_y);

	x += x_rel;
	y += y_rel;
	z += z_rel;
	
	if (x < 0 || y < 0 || z < 0 || x >= grid_x || y >= grid_y || z >= grid_z) {
		return -1;
	}
	else {
		return z+grid_z*y+(grid_z*grid_y)*x;
	}
}

__device__ int get_local_offset() {
	// This gets the offset of the current thread in the local particle array
	return (int)(threadIdx.z * (blockDim.x*blockDim.y) +
		threadIdx.y * (blockDim.x) +
		threadIdx.x);
}

__global__ void brute_force(particle* all_particles, int cell_size, int grid_x,
	int grid_y, int grid_z) {
	// This begins calculation of the particle interactions
	
	// Work out where in the array our particles start
	int cell_offset = current_global_offset(grid_x, grid_y, grid_z);

	// Each thread loads its own particle to local memory
	particle self =
		all_particles[(cell_size*cell_offset) + get_local_offset()];

	// Initialise the interaction values
	self.x_acc = (float)0.0;

	particle other;
	
	// Now calculate interactions
	// Loop everything
	for (int index=0; index < grid_x * grid_y * grid_z * CELLSIZE; index++) {
		// Try doing a predictable interaction
		other = all_particles[index];
		self.x_acc += (float)1.0;
	}

	// Don't overwrite particles in use
	__syncthreads();
	
	// Now put our particle back into global memory
	all_particles[(cell_size * cell_offset) + get_local_offset()] = self;
	
}

__global__ void do_cell(particle* all_particles, int cell_size, int grid_x,
	int grid_y, int grid_z) {
	// This begins calculation of the particle interactions

	// Each block needs access to its local particles and a neighbour
	// Note: We can't use 2*cell_size here since it's not constant!
	__shared__ particle local_particles[2*CELLSIZE];
	
	// Work out where in the array our particles start
	int cell_offset = current_global_offset(grid_x, grid_y, grid_z);

	// Each thread loads its own particle to local memory
	local_particles[get_local_offset()] =
		all_particles[(cell_size*cell_offset) + get_local_offset()];

	// Initialise the interaction values
	local_particles[get_local_offset()].x_acc = (float)0.0;

	int n_offset;
	
	// Now load in our neighbours and calculate interactions
	// Loop through neighbours
	for (int x_rel = -1; x_rel < 2; x_rel++) {
		for (int y_rel = -1; y_rel < 2; y_rel++) {
			for (int z_rel = -1; z_rel < 2; z_rel++) {

				// Only act if we've got a valid neighbour
				if (neighbour_offset(x_rel, y_rel, z_rel, grid_x, grid_y,
					grid_z) >= 0) {

					// Load neighbouring cell to shared memory

					// This is the memory location we need to start allocating
					// from
					n_offset = neighbour_offset(x_rel, y_rel, z_rel,
						grid_x, grid_y, grid_z);

					// Each thread allocates a particle from the neighbour
					// Start at cell_size since we're filling the second half
					local_particles[cell_size + get_local_offset()] =
						all_particles[n_offset*cell_size + get_local_offset()];

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
						local_particles[get_local_offset()].x_acc += (float)1.0;
					}

					// Ensure everyone's finished before loading the next cell
					__syncthreads();
					
				   }
			}
		}
	}

	// Now put shared values back into global memory
	all_particles[(cell_size * cell_offset) + get_local_offset()] =
		local_particles[get_local_offset()];

	// Make sure we're all done
	__syncthreads();
	
}

