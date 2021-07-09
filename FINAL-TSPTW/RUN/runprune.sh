#!/bin/bash

projectname=$1
numberofnodes=$2
instancename=$3
adjacencyname=$4
scenarioseed=13
seed=2176
numexpprob=10
numexpdirect=10
numprob=10
numdirect=10
moveafter=3590
timeout=$(($moveafter+110))
mkdir EXECUTION-${projectname}
cd EXECUTION-${projectname}
mkdir COLLAB-IN
mkdir COLLAB-OUT
mkdir SURROGATE-IN
mkdir SURROGATE-OUT
cp ../BIN/search*-* .
# nohup nice -n 15 ../SCRIPTS/aggregator.sh COLLAB-OUT/ COLLAB-IN/seed.txt&
parameterset=0
while true; do
../SCRIPTS/parallelpru.sh ${numberofnodes} ../../INSTANCES/${instancename} ../../INSTANCES/${adjacencyname} ${timeout}  ${scenarioseed} ${seed} ${numprob} ${numdirect} ${numexpprob} ${numexpdirect} ${parameterset}
sleep 10
rm core.*
sleep ${moveafter}
# ccskill --all
rm search*.*.*
pkill -f search
rm COLLAB-OUT/*
scenarioseed=$((17+${scenarioseed}))
seed=$((17+${seed}))
if [[ ${parameterset} -le 1 ]]
then parameterset=$((1+${parameterset}))
else parameterset=0
fi
done





