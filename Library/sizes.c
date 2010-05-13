/*
 * Useful for finding RAM requirements.
 */

#include "file_reading.c"

int main() {	

	// 64K ought to be enough for anybody....
	int line_length = 1024;
	
	// This will store a line of text from our input.
	char* line = (char*) malloc((unsigned int)line_length * sizeof(char));

	// Read the first line
	fgets(line, line_length, stdin);

	// Discard it. Now read the second
	fgets(line, line_length, stdin);

	// Read the input size
	int N;
	N = read_natural(line, line_length);

	// Now read the third
	fgets(line, line_length, stdin);
	
	float size;
	// Read the particle size
	size = read_decimal(line, line_length);

	// Grid dimensions
	float x = 100.0;
	float y = 100.0;
	float z = 100.0;
	
	float cells = (x*y*z)/(size*size*size);
	fprintf(stderr, "Need %i bytes for particles\n",
		(((int)cells)+N)*sizeof(particle));
	return 0;
}
