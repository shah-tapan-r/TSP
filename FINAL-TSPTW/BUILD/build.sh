#! /bin/bash

chmod 700 ../RUN/SCRIPTS/*.sh
../RUN/SCRIPTS/build-collect.sh

g++ -c -std=c++0x -O3 -DSEEDFILE -DNMINOROBJ -DPROGRESS -DNDEBUG -DCOLLABFILE -DNPRUNE -DPARAM -I ../SOURCES/boost_1_76_0/ -I ../SOURCES/DATASTRUCTURES/ ../SOURCES/tsptw-hegel.cpp ;  g++ -c -std=c++0x -O3 -DNDEBUG -I ../SOURCES/boost_1_76_0/ -I ../SOURCES/DATASTRUCTURES/ ../SOURCES/DATASTRUCTURES/general_includes.cpp ;  g++ -o ../RUN/BIN/search-direct -std=c++0x -O3 -DNDEBUG -I ../SOURCES/boost_1_76_0/ -I ../SOURCES/DATASTRUCTURES/ tsptw-hegel.o general_includes.o

g++ -c -std=c++0x -O3 -DSEEDFILE -DMINOROBJ -DPROGRESS -DNDEBUG -DCOLLABFILE -DNPRUNE -DPARAM -I ../SOURCES/boost_1_76_0/ -I ../SOURCES/DATASTRUCTURES/ ../SOURCES/tsptw-hegel.cpp ;  g++ -c -std=c++0x -O3 -DNDEBUG -I ../SOURCES/boost_1_76_0/ -I ../SOURCES/DATASTRUCTURES/ ../SOURCES/DATASTRUCTURES/general_includes.cpp ;  g++ -o ../RUN/BIN/search-probobj -std=c++0x -O3 -DNDEBUG -I ../SOURCES/boost_1_76_0/ -I ../SOURCES/DATASTRUCTURES/ tsptw-hegel.o general_includes.o

g++ -c -std=c++0x -O3 -DSEEDFILE -DNMINOROBJ -DPROGRESS -DNDEBUG -DCOLLABFILE -DPRUNE -DPARAM -I ../SOURCES/boost_1_76_0/ -I ../SOURCES/DATASTRUCTURES/ ../SOURCES/tsptw-hegel.cpp ;  g++ -c -std=c++0x -O3 -DNDEBUG -I ../SOURCES/boost_1_76_0/ -I ../SOURCES/DATASTRUCTURES/ ../SOURCES/DATASTRUCTURES/general_includes.cpp ;  g++ -o ../RUN/BIN/searchprune-direct -std=c++0x -O3 -DNDEBUG -I ../SOURCES/boost_1_76_0/ -I ../SOURCES/DATASTRUCTURES/ tsptw-hegel.o general_includes.o

g++ -c -std=c++0x -O3 -DSEEDFILE -DMINOROBJ -DPROGRESS -DNDEBUG -DCOLLABFILE -DPRUNE -DPARAM -I ../SOURCES/boost_1_76_0/ -I ../SOURCES/DATASTRUCTURES/ ../SOURCES/tsptw-hegel.cpp ;  g++ -c -std=c++0x -O3 -DNDEBUG -I ../SOURCES/boost_1_76_0/ -I ../SOURCES/DATASTRUCTURES/ ../SOURCES/DATASTRUCTURES/general_includes.cpp ;  g++ -o ../RUN/BIN/searchprune-probobj -std=c++0x -O3 -DNDEBUG -I ../SOURCES/boost_1_76_0/ -I ../SOURCES/DATASTRUCTURES/ tsptw-hegel.o general_includes.o


g++ -c -std=c++0x -O3 -DSEEDFILE -DEXPENSIVE -DNMINOROBJ -DPROGRESS -DNDEBUG -DCOLLABFILE -DNPRUNE -DPARAM -I ../SOURCES/boost_1_76_0/ -I ../SOURCES/DATASTRUCTURES/ ../SOURCES/tsptw-hegel.cpp ;  g++ -c -std=c++0x -O3 -DNDEBUG -I ../SOURCES/boost_1_76_0/ -I ../SOURCES/DATASTRUCTURES/ ../SOURCES/DATASTRUCTURES/general_includes.cpp ;  g++ -o ../RUN/BIN/expsearch-direct -std=c++0x -O3 -DNDEBUG -I ../SOURCES/boost_1_76_0/ -I ../SOURCES/DATASTRUCTURES/ tsptw-hegel.o general_includes.o

g++ -c -std=c++0x -O3 -DSEEDFILE -DMINOROBJ -DEXPENSIVE -DPROGRESS -DNDEBUG -DCOLLABFILE -DNPRUNE -DPARAM -I ../SOURCES/boost_1_76_0/ -I ../SOURCES/DATASTRUCTURES/ ../SOURCES/tsptw-hegel.cpp ;  g++ -c -std=c++0x -O3 -DNDEBUG -I ../SOURCES/boost_1_76_0/ -I ../SOURCES/DATASTRUCTURES/ ../SOURCES/DATASTRUCTURES/general_includes.cpp ;  g++ -o ../RUN/BIN/expsearch-probobj -std=c++0x -O3 -DNDEBUG -I ../SOURCES/boost_1_76_0/ -I ../SOURCES/DATASTRUCTURES/ tsptw-hegel.o general_includes.o

g++ -c -std=c++0x -O3 -DSEEDFILE -DNMINOROBJ -DPROGRESS -DEXPENSIVE -DNDEBUG -DCOLLABFILE -DPRUNE -DPARAM -I ../SOURCES/boost_1_76_0/ -I ../SOURCES/DATASTRUCTURES/ ../SOURCES/tsptw-hegel.cpp ;  g++ -c -std=c++0x -O3 -DNDEBUG -I ../SOURCES/boost_1_76_0/ -I ../SOURCES/DATASTRUCTURES/ ../SOURCES/DATASTRUCTURES/general_includes.cpp ;  g++ -o ../RUN/BIN/expsearchprune-direct -std=c++0x -O3 -DNDEBUG -I ../SOURCES/boost_1_76_0/ -I ../SOURCES/DATASTRUCTURES/ tsptw-hegel.o general_includes.o

g++ -c -std=c++0x -O3 -DSEEDFILE -DMINOROBJ -DPROGRESS -DNDEBUG -DEXPENSIVE -DCOLLABFILE -DPRUNE -DPARAM -I ../SOURCES/boost_1_76_0/ -I ../SOURCES/DATASTRUCTURES/ ../SOURCES/tsptw-hegel.cpp ;  g++ -c -std=c++0x -O3 -DNDEBUG -I ../SOURCES/boost_1_76_0/ -I ../SOURCES/DATASTRUCTURES/ ../SOURCES/DATASTRUCTURES/general_includes.cpp ;  g++ -o ../RUN/BIN/expsearchprune-probobj -std=c++0x -O3 -DNDEBUG -I ../SOURCES/boost_1_76_0/ -I ../SOURCES/DATASTRUCTURES/ tsptw-hegel.o general_includes.o

