#!/bin/bash

for f in $1/*.txt; do
  echo ../SCRIPTS/coll $f $2; 
  ../SCRIPTS/coll $f $2
done
