#!/bin/bash

projectname=$1
numberofnodes=$2
instancename=$3
adjacencyname=$4
scenarioseed=1776
seed=1376
numexpprob=2
numexpdirect=2
numprob=2
numdirect=2
moveafter=3590
timout=$(($moveafter+110))
mkdir EXECUTION-${projectname}
cd EXECUTION-${projectname}
mkdir COLLAB-IN
mkdir COLLAB-OUT
mkdir SURROGATE-IN
mkdir SURROGATE-OUT
cp ../BIN/search*-* .
cp ../BIN/expsearch*-* .

# nohup nice -n 15 ../SCRIPTS/aggregator.sh COLLAB-OUT/ COLLAB-IN/seed.txt&
parameterset=0
while true; do
echo "../SCRIPTS/parallel.sh ${numberofnodes} ../../INSTANCES/${instancename} ../../INSTANCES/${adjacencyname} ${timeout}  ${scenarioseed} ${seed} ${numprob} ${numdirect} ${parameterset}"
../SCRIPTS/parallel.sh ${numberofnodes} ../../INSTANCES/${instancename} ../../INSTANCES/${adjacencyname} ${timeout}  ${scenarioseed} ${seed} ${numprob} ${numdirect} ${numexpprob} ${numexpdirect} ${parameterset}
sleep 10
rm core.*
sleep ${moveafter}
# ccskill --all
rm search*.*.*
rm COLLAB-OUT/*
scenarioseed=$((17+${scenarioseed}))
seed=$((17+${seed}))
if [[ ${parameterset} -le 1 ]]
then parameterset=$((1+${parameterset}))
else parameterset=0
fi
done





