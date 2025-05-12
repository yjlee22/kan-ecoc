#!/bin/bash

# Define arrays for all parameters
seeds=(0 1 2 3 4 5)

# Run all combinations
for s in "${seeds[@]}"; do
    python main.py --hidden_dims 5 --seed ${o}
    python main.py --hidden_dims 5 --seed ${o} --ecoc
done