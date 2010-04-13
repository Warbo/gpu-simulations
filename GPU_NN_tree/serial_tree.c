#include "structs.c"
#include <math.h>

int power_of_two(int p) {
	// Returns two to the power of the argument, using integers
	int count;
	int total = 1;
	for (count=0; count < p; count++) {
		total = total * 2;
	}
	return total;
}

void make_leaf_from_particle(particle* the_particle, tree* the_leaf) {
	// Makes a tree with no children (ie. just a root node), containing the
	// given particle
	the_leaf->left = NULL;
	the_leaf->right = NULL;

	// If the particle doesn't exist then we're type 2
	if (the_particle == NULL) {
		the_leaf->leaf = 2;
	}

	// Otherwise we're type 1 and contain a particle
	else {
		the_leaf->p = *the_particle;
		the_leaf->leaf = 1;
	}
}

void join_trees(tree* first, tree* second, tree* root) {
	// Makes a tree at root with the first and second trees as its children
	
	// Assume for the moment that which is left and which is right is
	// arbitrary
	root->left = first;
	root->right = second;
	root->leaf = 0;
	// Make a new particle from the average of the given two
	composite_particle(first, second, &(root->p));
}

void composite_particle(tree* t) {
	// Make a pseudo-particle by averaging the children of the given tree
	// We may want to generalise this by taking a function pointer in the
	// future
	float delta_x, delta_y, delta_z;
	get_deltas(t->left.p, t->right.p, &delta_x, &delta_y, &delta_z);
	t->p.x = t->left.p.x + delta_x;
	t->p.y = t->left.p.y + delta_y;
	t->p.z = t->left.p.z + delta_z;
	t->p.mass = t->left.p.mass + t->second.p.mass;

	//TODO Handle more particle properties when actual particle structs are
	// known

}

void get_deltas(particle* p1, particle* p2, float* x, float* y, float* z) {
	// Sets the given x, y and z floats to the differences in p1 and p2's
	// positions
	*x = p2->x - p1->x;
	*y = p2->y - p1->y;
	*z = p2->z - p1->z;
}

float get_distance(particle* p1, particle* p2) {
	// Returns the scalar distance between the two given particles
	float x, y, z;
	get_deltas(p1, p2, &x, &y, &z);
	return (float) sqrt((double) (x*x + y*y + z*z));
}

void initialise_tree(tree** bin_tree, int length) {
	// Sets every node in the given tree to a dummy node
	int i;
	for (i=0; i < length; i++) {
		bin_tree[0][i].left = NULL;
		bin_tree[0][i].right = NULL;
		bin_tree[0][i].leaf = 2;
	}
}

unsigned int tree_particles(particle* particle_array, int number) {
	// The number of leaf nodes is the smallest power of two which is greater
	// than or equal to the particle number
	int leaf_number = 1;
	int leaf_power = 0
	while (leaf_number < number) {
		leaf_power++;
		leaf_number = poweroftwo(leaf_power);
	}
	
	// The number of nodes in a binary tree is 2*leaf_nodes - 1
	tree* bin_tree = (tree*) malloc(((2*leaf_number) - 1) * sizeof(tree));

	// Initialise to NULL
	initialise_tree(&bin_tree, 2*leaf_number - 1);
	
	// This is *NOT* the final tree, it is just a place to store nodes until
	// we coalesce them. Thus the order doesn't matter.

	// First insert leaf nodes
	int i;
	for (i=0; i < number; i++) {
		make_leaf_from_particle(
			particle_array[i], bin_tree[i]);
		}
	}
	
	// Now pair up the mutual nearest neighbours until we get a root node
	while (has_root(&bin_tree) == 0) {
		pair_off(&bin_tree, 2*leaf_number - 1);
	}

	// Now the particles are paired into a nearest neighbour tree via their
	// pointers. Coalesce this into a left-leaning binary tree array.
	
	//return (unsigned int) bin_tree;
}

int is_already_paired(tree** bin_tree, int length, tree* node) {
	// Returns 1 if the given node is already paired up in a tree, 0 otherwise
	int i;
	for (i=0; i < length; i++) {
		if (bin_tree[0][i].leaf == 0) {
			if (bin_tree[0][i].left == node || bin_tree[0][i].right == node) {
				return 1;
			}
		}
	}
	return 0;
}

void pair_off(tree** bin_tree, int length) {
	// Sort the tree to get nearest-neighbours together
	int i;
	for (i = 0; i < length; i++) {
		if (bin_tree[0][i].leaf == 2) {
			// Skip dummies
		}
		else {
			// We don't care about nodes which are already paired up
			if (is_already_paired(bin_tree, length, &(bin_tree[0][i])) == 0) {
				// find_neighbour will find the nearest neighbour, if there is
				// one, and pair up the two in a new node.
				find_neighbour(bin_tree, length, &bin_tree[0][i]);
			}
		}
	}
}

void pair_mutual_neighbours(tree** bin_tree, int length, int position) {
	// Looks through the bin_tree of length length to see if there is a
	// mutual nearest neighbour. If so it puts a new node in bin_tree pairing
	// them up, if not it leaves the tree alone.

	// Get the neighbour of the node at the given position
	int neighbour_index;
	neighbour_index = find_neighbour(bin_tree, length, position);

	// Now get the neighbour of the node at the found position
	int others_neighbour;
	others_neighbour = find_neighbour(bin_tree, length, position);

	// If other_neighbour is the same as the position we were given, it's mutual
	if (others_neighbour == position) {
		add_node(bin_tree, length, position, neighbour_index);
	}
}

void add_node(tree** bin_tree, int length, int p1, int p2) {
	// Adds a new node to the given tree, pairing up the nodes at positions p1
	// and p2. It overwrites an invalid node to store the new node.
	int i;
	int done = 0;
	for (i=0, i<length; i++) {
		if (done == 0) {
			if (bin_tree[0][i].leaf == 2) {
				bin_tree[0][i].left = &(bin_tree[0][p1]);
				bin_tree[0][i].right = &(bin_tree[0][p2]);
				bin_tree[0][i].leaf = 0;
				composite_particle(&(bin_tree[0][i]));
				done = 1;
			}
		}
	}
}

int find_neighbour(tree** bin_tree, int length, int position) {
	// Finds the nearest unpaired particle to the given one
	int i;
	for (i=0; i<length; i++) {
		// Don't look in invalid nodes
		if (bin_tree[0][i].leaf != 2) {
			// Don't look at paired nodes
			if (is_already_paired(bin_tree, length, &(bin_tree[0][i])) == 0) {
				// If we've not found anything yet, just take what we can
				if (smallest == -1) {
					smallest = get_distance(&(bin_tree[0][position].p),
						&(bin_tree[0][i].p));
					neighbour_index = i;
				}
				// Otherwise compare it to our current best find
				else {
					if (get_distance(&(node->p), &(bin_tree[0][i].p))<smallest){
						smallest = get_distance(&(bin_tree[0][position].p),
										&(bin_tree[0][i].p));
						neighbour_index = i;
					}
				}
			}
		}
	}
	return neighbour_index;
}
	// Now we should have the closest unpaired particle
	// See if it's mutual