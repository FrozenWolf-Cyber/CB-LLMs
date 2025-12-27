#!/bin/bash

get_best_gpus() {
    n=$1
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
        | awk -F',' '
    {
        mem = $2 + 0; util = $3 + 0;
        score = mem * 2 + util;
        print $1, score;
    }' | sort -k2 -n | head -n "$n" | awk '{print $1}'
}

if [[ "$1" == "--get" ]]; then
    n=${2:-1}  # Default to 1 if not specified
    get_best_gpus "$n"
else
    best_gpu=$(get_best_gpus 1)
    export CUDA_VISIBLE_DEVICES=$best_gpu
    echo "✅ Exported CUDA_VISIBLE_DEVICES=$best_gpu"
fi
