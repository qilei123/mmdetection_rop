#!/usr/bin/env bash

PYTHON=${PYTHON:-"python3"}

$PYTHON -m torch.distributed.launch --master_addr='10.0.3.29' --master_port=9901 --nproc_per_node=$2 $(dirname "$0")/train.py $1 --launcher pytorch ${@:3}
