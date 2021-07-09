#!/bin/bash

while true; do
 tail -n 2 search*.*.out | grep "At time"
 sleep 60
 clear
done
