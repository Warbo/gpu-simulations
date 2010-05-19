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

	core-final)
		gcc -c datatypes.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -DNDEBUG -o datatypes.o
		gcc -c file_reading.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -DNDEBUG -o file_reading.o
		gcc -c common_functions.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -DNDEBUG -o common_functions.o
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

	sentinels-final)
		gcc -c datatypes.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -DNDEBUG -o datatypes.o
		gcc -c file_reading.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -DNDEBUG -o file_reading.o
		gcc -c common_functions.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -DNDEBUG -o common_functions.o
		gcc -c sentinel_functions.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -DNDEBUG -o sentinel_functions.o
		gcc -c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm sentinel_tests.c -DNDEBUG -o sentinel_tests.o
		gcc sentinel_tests.o -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -I/usr/include -DNDEBUG -o sentinel_tests
		;;

	# Builds the sentinels implementation, ie. using one array with special
	# particles to mark cell boundaries
	pairarray)
		gcc -c datatypes.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -o datatypes.o
		gcc -c file_reading.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -o file_reading.o
		gcc -c common_functions.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -o common_functions.o
		gcc -c pair_array_functions.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -o pair_array_functions.o
		gcc -c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm pair_array_tests.c -o pair_array_tests.o
		gcc pair_array_tests.o -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -I/usr/include -o pair_array_tests
		;;

	pairarray-debug)
		gcc -c -fmudflap -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror datatypes.c -lm -o datatypes.o
		gcc -c -fmudflap -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm common_functions.c -o common_functions.o
		gcc -c -fmudflap pair_array_functions.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -o pair_array_functions.o
		gcc -c -fmudflap -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm pair_array_tests.c -o pair_array_tests.o
		gcc -lmudflap pair_array_tests.o -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -I/usr/include -o pair_array_tests
		;;

	pairarray-final)
		gcc -c datatypes.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -DNDEBUG -o datatypes.o
		gcc -c file_reading.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -DNDEBUG -o file_reading.o
		gcc -c common_functions.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -DNDEBUG -o common_functions.o
		gcc -c pair_array_functions.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -DNDEBUG -o pair_array_functions.o
		gcc -c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm pair_array_tests.c -DNDEBUG -o pair_array_tests.o
		gcc pair_array_tests.o -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -I/usr/include -DNDEBUG -o pair_array_tests
		;;

	cuda)
		gcc -c datatypes.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -o datatypes.o
		gcc -c file_reading.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -o file_reading.o
		gcc -c common_functions.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -o common_functions.o
		nvcc -c cuda_functions.cu -o cuda_functions.o
		nvcc -c cuda_tests.cu -o cuda_tests.o
		nvcc cuda_tests.o -o cuda_tests
		;;

	cuda-debug)
		gcc -c -fmudflap datatypes.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -o datatypes.o
		gcc -c -fmudflap file_reading.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -o file_reading.o
		gcc -c -fmudflap common_functions.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -o common_functions.o
		nvcc -c -fmudflap cuda_functions.cu -o cuda_functions.o
		nvcc -c -fmudflap cuda_tests.cu -o cuda_tests.o
		nvcc -lmudflap cuda_tests.o -o cuda_tests
		;;

	cuda-final)
		gcc -c datatypes.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -DNDEBUG -o datatypes.o
		gcc -c file_reading.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -DNDEBUG -o file_reading.o
		gcc -c common_functions.c -Wall -W -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -Werror -lm -DNDEBUG -o common_functions.o
		nvcc -c cuda_functions.cu -DNDEBUG -o cuda_functions.o
		nvcc -c cuda_tests.cu -DNDEBUG -o cuda_tests.o
		nvcc cuda_tests.o -DNDEBUG -o cuda_tests
		;;
	
	*)
		echo "Usage: BUILD.sh (core|sentinels|pairarray|cuda)[-debug|-final]"
		;;

esac

exit 0
