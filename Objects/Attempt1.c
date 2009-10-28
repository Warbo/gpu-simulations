#include <stdio.h>
#include <stdarg.h>

int read(int key1, ...) {
	/*
	 * This function reads a value from memory, given an integer set of
	 * keys.
	 */
	// Initialise our variable length argument list
	va_list keys;
	va_start(keys, key1);
	
	// Loop through the keys
	int current_key;
	for (current_key = key1; current_key != -1; current_key = va_arg(keys, int)) {
		//loop
	}
	
	// Clean up the argument list
	va_end(keys);
}

void write(int value, ...) {
	/*
	 * This function reads a value from memory, given a set of integer
	 * keys.
	 */
	// Initialise our variable length argument list
	va_list keys;
	va_start(keys, key1);
	
	// Loop through the keys
	int current_key;
	for (current_key = key1; current_key != -1; current_key = va_arg(keys, int)) {
		//loop
	}
	
	// Clean up the argument list
	va_end(keys);
}

int int_of (char* start, length)

int main(void)
{
   printargs(5, 2, 14, 84, 97, 15, 24, 48, -1);
   printargs(84, 51, -1);
   printargs(-1);
   printargs(1, -1);
   return 0;
}
