#!/bin/bash

# 입력받은 인자에서 1을 뺌
first=$1
first=$((first - 1))

# 0부터 first까지의 숫자로 루프를 돌며 tmux 세션을 종료
for i in $(seq 0 $first)
do
    SESSION_NAME="s$i"
    tmux kill-session -t $SESSION_NAME
done
