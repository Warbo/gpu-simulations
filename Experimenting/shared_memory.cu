/*
 * Hello world for CUDA, with access to the shared memory of the multiprocessors
 */

#include <stdio.h>
#include <stdlib.h>

// Define a kernel function
__global__ void vector_add(float* A, float* B, float* C) {
	// Add two vectors together and store in a third
	
	// Our ID is unique to our thread
	int i = threadIdx.x;
	
	C[i] = A[i] + B[i];

}

int main() {
	// This is the size of our vectors, and the number of threads
	int N = 10;
	
	// These will be our vectors on the host
	float* host_A;
	float* host_B;
	float* host_C;
	
	// Use this for indices
	int i;

	// Define our vectors on the host
	host_A = (float*) malloc(N*sizeof(float));
	host_B = (float*) malloc(N*sizeof(float));
	host_C = (float*) malloc(N*sizeof(float));
	
	// Initialise them
	for (i = 0; i < N; i++) {
		host_A[i] = (float)i;
		host_B[i] = 2.0 * (float)i;
		host_C[i] = 0.0;
	}

	// Define our vectors on the GPU
	float* device_A;
	float* device_B;
	float* device_C;
	cudaMalloc((void**) &device_A, sizeof(float)*N);
	cudaMalloc((void**) &device_B, sizeof(float)*N);
	cudaMalloc((void**) &device_C, sizeof(float)*N);

	// Transfer data to the GPU
	cudaMemcpy(device_A, host_A, sizeof(float)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(device_B, host_B, sizeof(float)*N, cudaMemcpyHostToDevice);
	//cudaMemcpy(device_C, host_C, sizeof(float)*N, cudaMemcpyHostToDevice);

	// Call our function; second number is how many threads to use
	// The first number is to do with thread blocks...
	vector_add<<<1, N>>>(device_A, device_B, device_C);

	// Copy memory back
	cudaMemcpy(host_C, device_C, sizeof(float)*N, cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(device_A);
	cudaFree(device_B);
	cudaFree(device_C);

	// Output our results
	printf("A = [");
	for (i = 0; i < N; i++) {
		printf("%G,", host_A[i]);
	}
	printf("]\n");
	
	printf("B = [");
	for (i = 0; i < N; i++) {
		printf("%G,", host_B[i]);
	}
	printf("]\n");
	
	printf("C = [");
	for (i = 0; i < N; i++) {
		printf("%G,", host_C[i]);
	}
	printf("]\n");
	
	return 0;
}
