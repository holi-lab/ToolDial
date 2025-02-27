#!/bin/bash

n=$1
category=$2

for i in $(seq 0 $((n-1)))
do
  tmux new-session -d -s s$i "python ${category}.py $i"
done
