#!/bin/bash

# Define arrays for all parameters
grids=(3 5 10)
orders=(1 2 3)
seeds=(0 1 2 3 4 5)

# Run all combinations
for s in "${seeds[@]}"; do
    for g in "${grids[@]}"; do
        for o in "${orders[@]}"; do
            python main.py --grid_size ${g} --spline_order ${o} --hidden_dims 5 --seed ${o}
            python main.py --grid_size ${g} --spline_order ${o} --hidden_dims 5 --seed ${o} --ecoc
        done
    done
done

for s in "${seeds[@]}"; do
    for g in "${grids[@]}"; do
        for o in "${orders[@]}"; do
            python main.py --grid_size ${g} --spline_order ${o} --hidden_dims 5 5 --seed ${o}
            python main.py --grid_size ${g} --spline_order ${o} --hidden_dims 5 5 --seed ${o} --ecoc
        done
    done
done

for s in "${seeds[@]}"; do
    for g in "${grids[@]}"; do
        for o in "${orders[@]}"; do
            python main.py --grid_size ${g} --spline_order ${o} --hidden_dims 5 5 5 --seed ${o}
            python main.py --grid_size ${g} --spline_order ${o} --hidden_dims 5 5 5 --seed ${o} --ecoc
        done
    done
done