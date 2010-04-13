#include <stdio.h>
#include <stdlib.h>

// This is to aid in debugging
#include <assert.h>

// Define our data structures
typedef struct particle particle;
typedef struct tree tree;

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

struct tree {
	/*
	 * A tree stores the particles. Since trees are fractal, every node is also
	 * a tree (a sub-tree). Even leaves are trees, but have NULL children.
	 * Each tree struct stores at most one particle, although the subtrees can
	 * also store one particle each recursively.
	 */
	
	tree* left;
	tree* right;

	// This number represents the tree's contents. If 0 the tree is not a leaf
	// node. If 1 the tree is a leaf node. If 2 the tree is a dummy leaf node
	// which contains no particle.
	int leaf;

	particle p;
	
};
