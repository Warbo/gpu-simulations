#!/usr/bin/env sh

# This builds the code. Each backend can be specified via an argument

case "$1" in

	# Builds only the common files, not the specific implementations
	core)
		gcc -c datatypes.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -o datatypes.o
		gcc -c file_reading.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -o file_reading.o
		gcc -c common_functions.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -o common_functions.o
		;;

	# Builds the common files, but with GNU Mudflap support (catches many errors
	# at runtime)
	core-debug)
		gcc -c -fmudflap -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror datatypes.c -lm -o datatypes.o
		gcc -c -fmudflap -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror file_reading.c -lm -o file_reading.o
		gcc -c -fmudflap -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm common_functions.c -o common_functions.o
		;;

	# Builds the sentinels implementation, ie. using one array with special
	# particles to mark cell boundaries
	sentinels)
		gcc -c datatypes.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -o datatypes.o
		gcc -c file_reading.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -o file_reading.o
		gcc -c common_functions.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -o common_functions.o
		gcc -c sentinel_functions.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -o sentinel_functions.o
		gcc -c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm sentinel_tests.c -o sentinel_tests.o
		gcc sentinel_tests.o -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -I/usr/include -o sentinel_tests
		;;

	sentinels-debug)
		gcc -c -fmudflap -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror datatypes.c -lm -o datatypes.o
		gcc -c -fmudflap -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm common_functions.c -o common_functions.o
		gcc -c -fmudflap sentinel_functions.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -o sentinel_functions.o
		gcc -c -fmudflap -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm sentinel_tests.c -o sentinel_tests.o
		gcc -lmudflap sentinel_tests.o -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -I/usr/include -o sentinel_tests
		;;

	cuda)
		gcc -c -Wall -W -Wextra -Wshadow -Wconversion -Wcast-qual -Wwrite-strings -Werror linkedlists.c -lm -o linkedlists.o
		nvcc -c macro_kernel.cu -o macro_kernel.o
		gcc -c tests.c -Wall -W -Wextra -Wshadow -Wconversion -Wcast-qual -Wwrite-strings -Werror -lm -o tests.o
		nvcc tests.o -lm -o tests
		;;
	
	*)
		gcc -c linkedlists.c -lm -o linkedlists.o
		;;

esac

exit 0
