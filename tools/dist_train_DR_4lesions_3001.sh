#!/usr/bin/env bash

PYTHON=${PYTHON:-"python3"}

$PYTHON -m torch.distributed.launch --master_addr='127.0.0.1' --master_port=17701 --nproc_per_node=$2 $(dirname "$0")/train.py $1 --launcher pytorch ${@:3}
