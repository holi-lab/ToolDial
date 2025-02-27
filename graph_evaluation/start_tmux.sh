#!/bin/bash

# 인자로 개수를 받음
NUM_SESSIONS=$1
threshold=$2

# 입력받은 개수만큼 tmux 세션을 생성
for ((i=0; i<NUM_SESSIONS; i++))
do
    SESSION_NAME="s$i"
    tmux new-session -d -s $SESSION_NAME "python stable_graph_metric/grapheval/run_eval.py $i $threshold"
    echo "Started tmux session: $SESSION_NAME with arg: $i"
done
