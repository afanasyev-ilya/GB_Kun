# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/afanasyev/LAGraph

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/afanasyev/LAGraph/build

# Include any dependencies generated for this target.
include src/test/CMakeFiles/test_SingleSourceShortestPath.dir/depend.make

# Include the progress variables for this target.
include src/test/CMakeFiles/test_SingleSourceShortestPath.dir/progress.make

# Include the compile flags for this target's objects.
include src/test/CMakeFiles/test_SingleSourceShortestPath.dir/flags.make

src/test/CMakeFiles/test_SingleSourceShortestPath.dir/test_SingleSourceShortestPath.c.o: src/test/CMakeFiles/test_SingleSourceShortestPath.dir/flags.make
src/test/CMakeFiles/test_SingleSourceShortestPath.dir/test_SingleSourceShortestPath.c.o: ../src/test/test_SingleSourceShortestPath.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/afanasyev/LAGraph/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object src/test/CMakeFiles/test_SingleSourceShortestPath.dir/test_SingleSourceShortestPath.c.o"
	cd /home/afanasyev/LAGraph/build/src/test && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/test_SingleSourceShortestPath.dir/test_SingleSourceShortestPath.c.o -c /home/afanasyev/LAGraph/src/test/test_SingleSourceShortestPath.c

src/test/CMakeFiles/test_SingleSourceShortestPath.dir/test_SingleSourceShortestPath.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/test_SingleSourceShortestPath.dir/test_SingleSourceShortestPath.c.i"
	cd /home/afanasyev/LAGraph/build/src/test && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/afanasyev/LAGraph/src/test/test_SingleSourceShortestPath.c > CMakeFiles/test_SingleSourceShortestPath.dir/test_SingleSourceShortestPath.c.i

src/test/CMakeFiles/test_SingleSourceShortestPath.dir/test_SingleSourceShortestPath.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/test_SingleSourceShortestPath.dir/test_SingleSourceShortestPath.c.s"
	cd /home/afanasyev/LAGraph/build/src/test && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/afanasyev/LAGraph/src/test/test_SingleSourceShortestPath.c -o CMakeFiles/test_SingleSourceShortestPath.dir/test_SingleSourceShortestPath.c.s

# Object files for target test_SingleSourceShortestPath
test_SingleSourceShortestPath_OBJECTS = \
"CMakeFiles/test_SingleSourceShortestPath.dir/test_SingleSourceShortestPath.c.o"

# External object files for target test_SingleSourceShortestPath
test_SingleSourceShortestPath_EXTERNAL_OBJECTS =

src/test/test_SingleSourceShortestPath: src/test/CMakeFiles/test_SingleSourceShortestPath.dir/test_SingleSourceShortestPath.c.o
src/test/test_SingleSourceShortestPath: src/test/CMakeFiles/test_SingleSourceShortestPath.dir/build.make
src/test/test_SingleSourceShortestPath: lib/liblagraphx.so.0.9.9
src/test/test_SingleSourceShortestPath: lib/liblagraphtest.so.0.9.9
src/test/test_SingleSourceShortestPath: /usr/local/lib64/libgraphblas.so.6.0.2
src/test/test_SingleSourceShortestPath: lib/liblagraph.so.0.9.9
src/test/test_SingleSourceShortestPath: /usr/local/lib64/libgraphblas.so.6.0.2
src/test/test_SingleSourceShortestPath: /usr/lib/gcc/aarch64-redhat-linux/8/libgomp.so
src/test/test_SingleSourceShortestPath: /usr/lib64/libpthread.so
src/test/test_SingleSourceShortestPath: src/test/CMakeFiles/test_SingleSourceShortestPath.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/afanasyev/LAGraph/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable test_SingleSourceShortestPath"
	cd /home/afanasyev/LAGraph/build/src/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_SingleSourceShortestPath.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/test/CMakeFiles/test_SingleSourceShortestPath.dir/build: src/test/test_SingleSourceShortestPath

.PHONY : src/test/CMakeFiles/test_SingleSourceShortestPath.dir/build

src/test/CMakeFiles/test_SingleSourceShortestPath.dir/clean:
	cd /home/afanasyev/LAGraph/build/src/test && $(CMAKE_COMMAND) -P CMakeFiles/test_SingleSourceShortestPath.dir/cmake_clean.cmake
.PHONY : src/test/CMakeFiles/test_SingleSourceShortestPath.dir/clean

src/test/CMakeFiles/test_SingleSourceShortestPath.dir/depend:
	cd /home/afanasyev/LAGraph/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/afanasyev/LAGraph /home/afanasyev/LAGraph/src/test /home/afanasyev/LAGraph/build /home/afanasyev/LAGraph/build/src/test /home/afanasyev/LAGraph/build/src/test/CMakeFiles/test_SingleSourceShortestPath.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/test/CMakeFiles/test_SingleSourceShortestPath.dir/depend

