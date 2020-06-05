#!/bin/bash
#SBATCH -p gpu --gres=gpu:2 -t 1:00:00
module load cuda
nvcc -o perlin main.cu
./perlin
