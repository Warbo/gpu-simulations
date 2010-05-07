/*
 * Defines our data structures.
 */

typedef struct particle particle;
typedef struct grid grid;

struct particle {
	/*
	* This particle implementation is for illustrative purposes only.
	* It can easily be replaced with an arbitrary datastructure.
	*/
	int id;			// Useful when its memory location isn't constant
	float x;		// x position
	float y;		// y position
	float z;		// z position
	float x_vel;	// x velocity
	float y_vel;	// y velocity
	float z_vel;	// z velocity
	float x_acc;	// x acceleration
	float y_acc;	// y acceleration
	float z_acc;	// z acceleration
	float mass;		// The particle's mass
};

struct grid {
	/*
	* A grid represents the space being simulated. It stores particles
	* and cells, which point to the particles.
	*/
	
	// The dimensions of the space, as multiples of the cell dimensions,
	// are stored here
	int x_size;
	int y_size;
	int z_size;
	
	// These store the dimensions of the cells
	float dx;
	float dy;
	float dz;
	
	// The location of the first cell (x minimum, y minimum, z minimum)
	float x_offset;
	float y_offset;
	float z_offset;
	
	// This is the total number of particles in this space
	int particle_number;
	
	// This stores the particles which the cells contain
	particle* particles;
	
};