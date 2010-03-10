/*
 * CUDA code for making particles interact.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "linkedlists.c"

// Define our kernel function
__device__ void particles_interact(particle* particle_A, particle* particle_B) {
	// Make two particles interact

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

}

__device__ void calculate_from_buffer(particle* particle_A, particle* buffer, int buffer_size) {
	// Calculate interactions between a particle and everything in our buffer

	// Loop through the buffer
	for (unsigned int counter = 0; counter < buffer_size; counter++) {
		// Update the particle based on the buffer contents
		particles_interact(particle_A, &buffer[counter]);
	}
}

__device__ void do_neighbouring_cells(x_cell, y_cell, z_cell, x_size, y_size, z_size, cell_size) {
	// Loop through neighbours
        for (int x_rel = -1; x_rel < 2; x_rel++) {
                for (int y_rel = -1; y_rel < 2; y_rel++) {
                        for (int z_rel = -1; z_rel < 2; z_rel++) {

                                // Only act if we've got a valid neighbour
                                if (
                                        (x_cell+x_rel >= 0) && (x_cell+x_rel < x_size) &&
                                        (y_cell+y_rel >= 0) && (y_cell+y_rel < y_size) &&
                                        (z_cell+z_rel >= 0) && (z_cell+z_rel < z_size)
                                ) {

					// Load neighbour to shared memory
					int offset = (z_size*y_size*(x_cell+x_rel)) + (z_size*(y_cell+y_rel)) + (z_cell+z_rel);
				
					for (int count = 0; count < cell_size; count++) {
						local_particles[cell_size+count] = all_particles[(cell_size*offset)+count];
					}
	
					// TODO: DON'T LOOP HERE, SET UP SOME THREADS AND MAKE THEM SYNCHRONISE			
					for (int count = 0; count < cell_size; count++) {
						calculate_from_buffer(&(local_particles[count]), 
&(local_particles[cell_size]), cell_size);
					}

				}
			}
		}
	}
}

//extern __shared__ particle current_particles
__device__ void do_particle(particle* particle_A, int x, int y, int z) {
	// Start off with no acceleration
	particle_A->x_acc = 0.0;
	particle_A->y_acc = 0.0;
	particle_A->z_acc = 0.0;

	// Calculate all interactions	
	do_neighbouring_cells(particle_A, x, y, z);
}

__device__ __global__ void do_cell(particle* all_particles, int x_size, int y_size, int z_size, int x, int y, int z, int 
cell_size) {
	// This sorts out memory for each cell thread

	// Work out where in the array our particles start
	int offset = (z_size*y_size*x) + (z_size*y) + z;

	// Allocate some local storage for them
	// TODO: At the moment, treat all cells as having cell_size particles
	// Allocating 2*cell_size gives us our local cell and a neighbour cell
	__device__ __shared__ particle local_particles[2*cell_size];

	for (int count = 0; count < cell_size; count++) {
		local_particles[count] = all_particles[(cell_size*offset)+count];
	}
	
	// Now load in our neighbours and calculate interactions
	do_neighbouring_cells(x, y, z, x_size, y_size, z_size, cell_size);
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
