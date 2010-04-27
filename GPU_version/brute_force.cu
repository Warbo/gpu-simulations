/*
 * CUDA code for making particles interact.
 * In this version, the kernel function is one big function and the particles
 * are taken straight from memory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "linkedlists.c"

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

__device__ int get_local_offset() {
	// This gets the offset of the current thread in the local particle array
	return (int)(threadIdx.z * (blockDim.x*blockDim.y) +
		threadIdx.y * (blockDim.x) +
		threadIdx.x);
}

__global__ void do_cell(particle* all_particles, int cell_size, int grid_x,
	int grid_y, int grid_z) {
	// This begins calculation of the particle interactions

	particle neighbour;
	particle self = all_particles[current_global_offset(grid_x, grid_y, grid_z)*
		cell_size + get_local_offset];

	self.x_acc = (float)0.0;
	
	int index;
	for (index = 0; index < grid_x*grid_y*grid_z*cell_size; index++) {
		particle = all_particles[index];
		self.x_acc += (float)1.0;
	}

	// Now put shared values back into global memory
	all_particles[(cell_size * get_current_global_offset(
		grid_x, grid_y,grid_z)) + get_local_offset()] =
		self;

	// Make sure we're all done
	__syncthreads();
	
}

int main() {
	// Toy example for testing

	// Put our GPU's limits here
	// int block_size = 512;
	//int grid_size;
	
	// Cell maximum size
	int cell_size = 29;
	
	// Particle grid dimensions
	int grid_x = 8;
	int grid_y = 6;
	int grid_z = 7;
	
	// Allocate room for a 3x3x3 grid with 32 particles each
	particle* all_particles_host = (particle*)malloc(
		grid_x*grid_y*grid_z*cell_size*sizeof(particle));

	// Give particles random positions
	for (int i=0; i < grid_x*grid_y*grid_z*cell_size; i++) {
		all_particles_host[i].x = (float) (rand()/(float)(RAND_MAX));
		all_particles_host[i].y = (float) (rand()/(float)(RAND_MAX));
		all_particles_host[i].z = (float) (rand()/(float)(RAND_MAX));
	}

	// Allocate memory on the GPU
	particle* all_particles_device;
	cudaMalloc(
		(void**)&all_particles_device,
		grid_x*grid_y*grid_z*cell_size*sizeof(particle));

	// Copy across our particles
	cudaMemcpy(all_particles_device, all_particles_host,
		grid_x*grid_y*grid_z*cell_size*sizeof(particle),
			   cudaMemcpyHostToDevice);

	dim3 dimGrid(grid_x*grid_y*grid_z);
	// Calculate the interactions
	do_cell<<<dimGrid, cell_size>>>(all_particles_device, cell_size,
		grid_x, grid_y, grid_z);

	// Get results back
	cudaMemcpy(all_particles_host, all_particles_device,
		grid_x*grid_y*grid_z*cell_size*sizeof(particle),
		cudaMemcpyDeviceToHost);

	// Free up the memory
	cudaFree(all_particles_device);

	for (int i=0; i<grid_x*grid_y*grid_z*cell_size; i++) {
		printf("%G\n", all_particles_host[i].x_acc);
	}

	// Exit
	return 0;
}
