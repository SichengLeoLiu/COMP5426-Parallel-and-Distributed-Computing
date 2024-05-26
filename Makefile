# Makefile for assignment2_1 and assignment2_2

# Compiler
CC = gcc

# Compiler flags
CFLAGS = -O3 -I C:\MPI\Include\

# Linker flags
LDFLAGS = -L C:\MPI\Lib\x64\ -lmsmpi

# Targets
TARGETS = assignment2_1 assignment2_2

# Source files
SRCS = assignment2_1.c assignment2_2.c

# Object files
OBJS = $(SRCS:.c=.o)

# Default target
all: $(TARGETS)

# Rule to build assignment2_1
assignment2_1: assignment2_1.o
	$(CC) -o $@ $^ $(LDFLAGS)

# Rule to build assignment2_2
assignment2_2: assignment2_2.o
	$(CC) -o $@ $^ $(LDFLAGS)

# Rule to build object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJS) $(TARGETS)

# Phony targets
.PHONY: all clean
