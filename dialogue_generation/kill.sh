#!/bin/bash

n=$1

for i in $(seq 0 $n)
do
  tmux kill-session -t s$i
done
