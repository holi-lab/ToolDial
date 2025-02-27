#!/bin/bash

NUM_SESSIONS=$1
category=$2

SESSIONS=()
for ((i=0; i<NUM_SESSIONS; i++)); do
  SESSIONS+=("s${i}")
done

PYTHON_SCRIPT="dial_generate.py"

for i in "${!SESSIONS[@]}"; do
  tmux new-session -d -s "${SESSIONS[i]}" "python3 $PYTHON_SCRIPT $i $category"
done

echo "All tmux sessions have been started and Python scripts are running."
