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

/*
__device__ void calculate_from_cell(particle* particle_A, particle* neighbours, int neighbour_number) {
	// Calculate the interactions between a particle and all of the neighbours in a cell
	// We need to split the neighbours into chunks to fit in our buffer
	int done = 0;
	while (done < neighbour_number) {
		
		for (unsigned int counter = 0; counter < buffer_size; counter++)
		
*/

__device__ void do_particle(particle* particle_A, grid* the_grid) {
	// Start off with no acceleration
	particle_A->x_acc = 0.0;
	particle_A->y_acc = 0.0;
	particle_A->z_acc = 0.0;
	
	// Get this particle's cell
	int x_cell, y_cell, z_cell;
	get_index_from_position(the_grid, particle_A, &x_cell, &y_cell, &z_cell);

	// Loop through neighbours
	for (int x_rel = -1; x_rel < 2; x_rel++) {
		for (int y_rel = -1; y_rel < 2; y_rel++) {
			for (int z_rel = -1; z_rel < 2; z_rel++) {

				// Only act if we've got a valid neighbour
				if (
					(x_cell+x_rel >= 0) && (x_cell+x_rel < the_grid->x_size) &&
					(y_cell+y_rel >= 0) && (y_cell+y_rel < the_grid->y_size) &&
					(z_cell+z_rel >= 0) && (z_cell+z_rel < the_grid->z_size)
				) {
					calculate_from_cell(particle_A, 
