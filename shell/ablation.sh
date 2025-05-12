#!/bin/bash

# Define arrays for all parameters
grids=(3 5 10)
models=("fast" "faster")
seeds=(0 1 2 3 4 5)

# Run all combinations
for s in "${seeds[@]}"; do
    for g in "${grids[@]}"; do
        for m in "${models[@]}"; do
            python main.py --grid_size ${g} --hidden_dims 5 --model ${m} --seed ${s}
            python main.py --grid_size ${g} --hidden_dims 5 -model ${m} --seed ${s} --ecoc
        done
    done
done