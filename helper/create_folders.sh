#!/bin/bash

# Define arrays for each part of the directory structure
dirs_a=("mult" "self_mm" "misa")
dirs_b=("mosei" "mosi" "sims")
dirs_c=("powmix" "base")


# Loop through all combinations and create the directories
for a in "${dirs_a[@]}"; do
  for b in "${dirs_b[@]}"; do
    for c in "${dirs_c[@]}"; do
        mkdir -p "ckpt/$a/$b/$c"
    done
  done
done
