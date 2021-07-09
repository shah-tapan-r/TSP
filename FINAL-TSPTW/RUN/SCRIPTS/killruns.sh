#!/bin/bash

for ((i=$1;i<=$2;i++)); 
do
  echo "ccskill $i"
  ccskill $i&
  rm search*.${i}.*
done

