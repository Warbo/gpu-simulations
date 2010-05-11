#include <stdio.h>
#include <stdlib.h>
#include "datatypes.c"
#include <assert.h>

/*
 * Facilities for reading particle data from STDIN
 */

int count_commas(char* c_array, int max_length) {
	/*
	* Counts the number of commas in the given array of characters
	*/

	// PRECONDITIONS
	assert(c_array != NULL);
	assert(max_length > 0);

	// These are the only characters we need to find
	char comma = (char)44;
	char linefeed = (char)10;

	// Our running total
	int count = 0;
	// Our loop index
	int index;
	
	// Look through the entire array, exiting if we find a newline
	for (index=0; index < max_length; index++) {
		if (c_array[index] == comma) {
			count++;
		}
		else if (c_array[index] == linefeed) {
			// DEBUG
			assert(count >= 0);
			return count;
		}
	}

	// If we're here then there was no newline, so just return

	// POST CONDITIONS
	assert(count >= 0);
	
	return count;
}

int read_natural(char* c_array, int max_length) {
	/*
	* Reads a decimal natural number from an array of characters,
	* encoded as ASCII.
	*/

	// PRECONDITIONS
	assert(c_array != NULL);
	assert(max_length > 0);

	// Define ASCII codes for numbers, in decimal
	char zero, one, two, three, four, five, six, seven, eight, nine, linefeed,
 		comma, null_end, dot;
	zero = (char)48;
	one = (char)49;
	two = (char)50;
	three = (char)51;
	four = (char)52;
	five = (char)53;
	six = (char)54;
	seven = (char)55;
	eight = (char)56;
	nine = (char)57;
	linefeed = (char)10;
	comma = (char)44;
	null_end = (char)0;
	dot = (char)46;
	
	// This keeps track of the number we've read
	int stored_number = 0;
	int index;
	for (index=0; index<max_length; index++) {
		if (c_array[index] == zero) {
			stored_number = (stored_number * 10) + 0;
		}
		else if (c_array[index] == one) {
			stored_number = (stored_number * 10) + 1;
		}
		else if (c_array[index] == two) {
			stored_number = (stored_number * 10) + 2;
		}
		else if (c_array[index] == three) {
			stored_number = (stored_number * 10) + 3;
		}
		else if (c_array[index] == four) {
			stored_number = (stored_number * 10) + 4;
		}
		else if (c_array[index] == five) {
			stored_number = (stored_number * 10) + 5;
		}
		else if (c_array[index] == six) {
			stored_number = (stored_number * 10) + 6;
		}
		else if (c_array[index] == seven) {
			stored_number = (stored_number * 10) + 7;
		}
		else if (c_array[index] == eight) {
			stored_number = (stored_number * 10) + 8;
		}
		else if (c_array[index] == nine) {
			stored_number = (stored_number * 10) + 9;
		}
		else if (c_array[index] == linefeed ||
			c_array[index] == comma ||
			c_array[index] == null_end ||
			c_array[index] == dot) {
			return stored_number;
		}
		else {
			// ERROR
			assert(c_array[index] == linefeed);
		}
	}

	return stored_number;
	
}

int power_of_ten(int power) {
	/*
	 * Returns 10 to the power of the argument.
	 */

	// PRECONDITIONS
	assert(power >= 0);
	
	int result = 1;

	int index;
	for (index=0; index < power; index++) {
		result = result * 10;
	}

	return result;
}

float read_fraction(char* c_array, int max_length) {
	/*
	 * Reads a decimal natural number from an array of characters,
	 * encoded as ASCII.
	 */

	// PRECONDITIONS
	assert(c_array != NULL);
	assert(max_length > 0);

	// Define ASCII codes for numbers, in decimal
	char zero, one, two, three, four, five, six, seven, eight, nine, linefeed,
		comma, null_end, dot;
	zero = (char)48;
	one = (char)49;
	two = (char)50;
	three = (char)51;
	four = (char)52;
	five = (char)53;
	six = (char)54;
	seven = (char)55;
	eight = (char)56;
	nine = (char)57;
	linefeed = (char)10;
	comma = (char)44;
	null_end = (char)0;
	dot = (char)46;
	
	// This keeps track of the number we've read
	float stored_number = (float)0.0;
	int index;
	for (index=0; index<max_length; index++) {
		if (c_array[index] == zero) {
			// No change
		}
		else if (c_array[index] == one) {
			stored_number = stored_number +
				(((float)1)/(float)power_of_ten(index+1));
		}
		else if (c_array[index] == two) {
			stored_number = stored_number +
				(((float)2)/(float)power_of_ten(index+1));
		}
		else if (c_array[index] == three) {
			stored_number = stored_number +
				(((float)3)/(float)power_of_ten(index+1));
		}
		else if (c_array[index] == four) {
			stored_number = stored_number +
				(((float)4)/(float)power_of_ten(index+1));
		}
		else if (c_array[index] == five) {
			stored_number = stored_number +
				(((float)5)/(float)power_of_ten(index+1));
		}
		else if (c_array[index] == six) {
			stored_number = stored_number +
				(((float)6)/(float)power_of_ten(index+1));
		}
		else if (c_array[index] == seven) {
			stored_number = stored_number +
				(((float)7)/(float)power_of_ten(index+1));
		}
		else if (c_array[index] == eight) {
			stored_number = stored_number +
				(((float)8)/(float)power_of_ten(index+1));
		}
		else if (c_array[index] == nine) {
			stored_number = stored_number +
				(((float)9)/(float)power_of_ten(index+1));
		}
		else if (c_array[index] == linefeed ||
			c_array[index] == comma ||
			c_array[index] == null_end ||
			c_array[index] == dot) {

			return stored_number;
		}
		else {
			// ERROR
			assert(c_array[index] == linefeed);
		}
	}

	return stored_number;
}
	
float read_decimal(char* c_array, int max_length) {
	/*
	*
	*/

	// PRECONDITIONS
	assert(c_array != NULL);
	assert(max_length > 0);

	// We're interested in full stops, commas, line feeds and string terminators
	char dot = (char)46;
	char linefeed = (char)10;
	char comma = (char)44;
	char null_end = (char)0;

	// Whether we have a decimal
	int decimal = -1;

	// Look for a dot in the first number
	int index;
	for (index = 0; index < max_length; index++) {
		if (c_array[index] == comma ||
			c_array[index] == null_end ||
			c_array[index] == linefeed) {

			index = max_length;
		}
		else if (c_array[index] == dot) {
			decimal = index;
		}
	}

	// If decimal is still -1 then we found no decimal point, so look for a
	// natural number
	if (decimal == -1) {
		return (float)read_natural(c_array, max_length);
	}
	else {
		// Otherwise, we've got a decimal so we need to look for two natural
		// numbers
		int whole = read_natural(c_array, max_length);
		float fraction = read_fraction(
			&(c_array[decimal+1]), max_length-decimal
		);
		return (float)whole + fraction;
	}
}

float read_positive_or_negative(char* c_array, int max_length) {
	/*
	 * Reads a decimal number from the start of c_array, only checking at most
	 * max_length characters, and allowing a minus sign in front. Stops at
	 * linefeeds and commas, fails at anything else.
	 */

	// PRECONDITIONS
	assert(c_array != NULL);
	assert(max_length > 0);

	// We need to catch negatives here
	char minus = (char)45;
	if (c_array[0] == minus) {

		// DEBUG
		assert(max_length > 1);
		
		return -1 * read_decimal(&(c_array[1]), max_length - 1);
	}
	
	return read_decimal(c_array, max_length);

}

void read_particle_line(char* c_array, int max_length, int* ID,
	float* x, float* y, float* z, float* x_vel, float* y_vel, float* z_vel,
	float* x_acc, float* y_acc, float* z_acc, float* mass) {
	/*
	 * Reads a line of input containing:
	 * ID, x, y, z, x_vel, y_vel, z_vel, x_acc, y_acc, z_acc, mass
	 * and assigns the values it finds to the given numbers.
	 */

	// PRECONDITIONS
	assert(c_array != NULL);
	assert(max_length > 0);
	assert(ID != NULL);
	assert(x != NULL);
	assert(y != NULL);
	assert(z != NULL);
	assert(x_vel != NULL);
	assert(y_vel != NULL);
	assert(z_vel != NULL);
	assert(x_acc != NULL);
	assert(y_acc != NULL);
	assert(z_acc != NULL);
	assert(mass != NULL);

	// We need to have 10 commas since we have 11 values
	assert(count_commas(c_array, max_length) == 10);

	// Take note of where the commas are
	char comma = (char)44;
	int* comma_positions = (int*) malloc((unsigned int)10 * sizeof(int));
	int comma_index;	// Which comma we're looking for
	int index;
	int start_at = 0;	// Start looking immediately after the last one
	for (comma_index=0; comma_index < 10; comma_index++) {

		// Initialise to a meaningless number for debugging
		comma_positions[comma_index] = -1;
		
		// Look through the entire array for a comma
		for (index=start_at; index < max_length; index++) {
			if (c_array[index] == comma) {
				// Make a note of it
				comma_positions[comma_index] = index;
				// Start looking for the next one at position +1
				start_at = index + 1;
				// Exit the loop
				index = max_length;
			}
		}

		// DEBUG
		assert(comma_positions[comma_index] != -1);
		
	}

	// Now read each value we need
	*ID = read_natural(c_array, max_length);
	*x = read_positive_or_negative(&(c_array[comma_positions[0]+1]),
		max_length - comma_positions[0]+1);
	*y = read_positive_or_negative(&(c_array[comma_positions[1]+1]),
		max_length - comma_positions[1]+1);
	*z = read_positive_or_negative(&(c_array[comma_positions[2]+1]),
		max_length - comma_positions[2]+1);
	*x_vel = read_positive_or_negative(&(c_array[comma_positions[3]+1]),
		max_length - comma_positions[3]+1);
	*y_vel = read_positive_or_negative(&(c_array[comma_positions[4]+1]),
		max_length - comma_positions[4]+1);
	*z_vel = read_positive_or_negative(&(c_array[comma_positions[5]+1]),
		max_length - comma_positions[5]+1);
	*x_acc = read_positive_or_negative(&(c_array[comma_positions[6]+1]),
		max_length - comma_positions[6]+1);
	*y_acc = read_positive_or_negative(&(c_array[comma_positions[7]+1]),
		max_length - comma_positions[7]+1);
	*z_acc = read_positive_or_negative(&(c_array[comma_positions[8]+1]),
		max_length - comma_positions[8]+1);
	*mass = read_positive_or_negative(&(c_array[comma_positions[9]+1]),
		max_length - comma_positions[9]+1);

	// Cleanup
	free(comma_positions);
	
}

void assign_particle(char* c_array, int max_length, particle* p) {
	/*
	 *
	 */

	// PRECONDITIONS
	assert(c_array != NULL);
	assert(max_length > 0);
	assert(p != NULL);

	read_particle_line(c_array, max_length, &(p->id),
		&(p->x), &(p->y), &(p->z), &(p->x_vel), &(p->y_vel), &(p->z_vel),
		&(p->x_acc), &(p->y_acc), &(p->z_acc), &(p->mass));
}

void read_particles(particle** p_array, int* length, float* size) {
	/*
	 * Reads data from stdin. These values are put into
	 * an array of particles created at *p_array of length *length.
	 */

	/*
	 * FILE FORMAT IS THE FOLLOWING:
	 * First line is ignored
	 * Second line contains a positive ASCII decimal integer, giving particle
	 * number.
	 * Third line contains a positive ASCII decimal float, giving the maximum
	 * particle size.
	 * Afterwards, CSV format with each line representing a particle:
	 * ID, x, y, z, x_vel, y_vel, z_vel, x_acc, y_acc, z_acc, mass
	 */

	// PRECONDITIONS
	assert(p_array != NULL);
	assert(length != NULL);

	// 64K ought to be enough for anybody....
	int line_length = 1024;
	
	// This will store a line of text from our input.
	char* line = (char*) malloc((unsigned int)line_length * sizeof(char));

	// Read the first line
	fgets(line, line_length, stdin);

	// Discard it. Now read the second
	fgets(line, line_length, stdin);

	// Read the input size
	*length = read_natural(line, line_length);

	// Now read the third
	fgets(line, line_length, stdin);

	// Read the particle size
	*size = read_decimal(line, line_length);
	
	// Now that we know the number of particles, allocate memory for them
	*p_array = (particle*) malloc((unsigned int)(*length) * sizeof(particle));

	// Now fill the array
	int index = 0;
	while (index < *length) {
		// Read in a line
		fgets(line, line_length, stdin);
		assign_particle(line, line_length, &(p_array[0][index]));
		index++;
	}

	// Cleanup
	free(line);
	
}