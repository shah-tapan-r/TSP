#!/bin/bash

g++ -O3 -DNDEBUG -I ../SOURCES/DATASTRUCTURES/ -c ../SOURCES/collect.cpp; g++ -O3 -DNDEBUG -I ../SOURCES/DATASTRUCTURES -c ../SOURCES/DATASTRUCTURES/general_includes.cpp; g++ -o ../RUN/SCRIPTS/coll collect.o general_includes.o
