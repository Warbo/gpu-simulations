// A "hello world" style CUDA program
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
	int N = 100;
	//int keep_going = 1;
	
	// These will be our vectors
	float* A;
	float* B;
	float* C;
	
	// Use this for indices
	int i;
	
	//while (keep_going == 1) {

		// Define our vectors
		A = (float*) malloc(N*sizeof(float));
		B = (float*) malloc(N*sizeof(float));
		C = (float*) malloc(N*sizeof(float));
	
		// Initialise them
		for (i = 0; i < N; i++) {
			A[i] = (float)i;
			B[i] = 2.0 * (float)i;
			C[i] = 0.0;
		}

		// Call our function; second number is how many threads to use
		// The first number is to do with thread blocks...
		vector_add<<<1, N>>>(A, B, C);

		printf("N=%i\n", N);

	//	if (C[1] == 0) {
	//		keep_going = 0;
	//	}
	//	else {
	//		N++;
	//	}
		
	//}
	
	
	//return 0;
	
//}


	// Output our results
	printf("A = [");
	for (i = 0; i < N; i++) {
		printf("%G,", A[i]);
	}
	printf("]\n");
	
	printf("B = [");
	for (i = 0; i < N; i++) {
		printf("%G,", B[i]);
	}
	printf("]\n");
	
	printf("C = [");
	for (i = 0; i < N; i++) {
		printf("%G,", C[i]);
	}
	printf("]\n");
	
	return 0;
}
