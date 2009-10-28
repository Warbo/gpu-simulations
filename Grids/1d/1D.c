#include <stdio.h>
#include <stdlib.h>

struct particle {
	double position;
	double velocity;
	double mass;
	double acceleration;
}

void do_gravity(particle* A, particle* B) {
	/* A simple gravitational force simulator which causes the given
	 * particles' accelerations to accrue the force between the two
	 */
	if (A == B) {
		return
	}
	else {
		if (A->position < B->position) {
			double a = A->mass*B->mass;
			A->acceleration += a;
			B->acceleration -= a;
		}
		else {
			if (A->position > B->position) {
				double a = A->mass*B->mass;
				A->acceleration -= a;
				B->acceleration += a;
			}
		}
	}
	

main() {
	// Allocate some memory for our simulation
	int mem = (int*) malloc( 24*sizeof(particle) );
	// Check that it has been allocated correctly, if not exit with error	
	if( mem == NULL ) {
		return 1;
	}

	particle* A = mem;
	particle* B = mem+(2*sizeof(particle));

	int position;
	for (position=0; position < 24; position++) {
		A[position] = particle;
	}

}
