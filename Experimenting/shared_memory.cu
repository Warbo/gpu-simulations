/*
 * Hello world for CUDA, with access to the shared memory of the multiprocessors
 */

#include <stdio.h>
#include <stdlib.h>

// Define a kernel function
__global__ void vector_sum(float* A, float* B, int length) {
	// Take a vector A of length "length" and sum it, putting the result in
	// the vector B

	// Declare some shared memory to store the sums.
	// We need enough floats for each thread to have one, so use blockDim.x
	__shared__ float sums[blockDim.x];
	
	// Our ID is unique to our thread, so use it as our index

	// Initialise our sum
	sums[threadIdx.x] = 0;

	// Calculate the sum
	for (unsigned int i = 0; i < length; i++) {
		sums[threadIdx.x] += A[i];
	}
	
	B[threadIdx.x] = sums[threadIdx.x];

}

int main() {
	// This is the size of our output vector, and the number of threads
	int N = 10;

	// This will be the length of our input vectors
	int length = 50;
	
	// These will be our vectors on the host
	float* host_A;		// This contains all input vectors
	float* host_B;
	
	// Use this for indices
	int i;

	// Define our vectors on the host
	host_A = (float*) malloc(N*length*sizeof(float));
	host_B = (float*) malloc(N*sizeof(float));
	
	// Initialise them
	for (i = 0; i < N*length; i++) {
		host_A[i] = (float)(i%length);
		//host_B[i] = 0.0;
	}

	// Define our vectors on the GPU
	float* device_A;
	float* device_B;
	cudaMalloc((void**) &device_A, sizeof(float)*N*length);
	cudaMalloc((void**) &device_B, sizeof(float)*N);

	// Transfer data to the GPU
	cudaMemcpy(device_A, host_A, sizeof(float)*N*length,
		cudaMemcpyHostToDevice);
	//cudaMemcpy(device_B, host_B, sizeof(float)*N, cudaMemcpyHostToDevice);

	// Call our function; second number is how many threads to use
	// The first number is to do with thread blocks...
	vector_sum<<<1, N>>>(device_A, device_B, length);

	// Copy memory back
	cudaMemcpy(host_B, device_B, sizeof(float)*N, cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(device_A);
	cudaFree(device_B);

	// Output our results
	printf("A = [");
	for (i = 0; i < N*length; i++) {
		if (i%length == 0) {
			printf("\n");
		}
		printf("%G,", host_A[i]);
	}
	printf("]\n");
	
	printf("B = [");
	for (i = 0; i < N; i++) {
		printf("%G,", host_B[i]);
	}
	printf("]\n");
	
	return 0;
}
