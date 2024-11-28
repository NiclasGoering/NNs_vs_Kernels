#!/bin/bash

# Load necessary modules
module load python
module load openmpi


# Activate your virtual environment if you have one
source /mnt/users/goringn/rs1/rs1_env/bin/activate  # Change this path if needed

# Change to the directory where the script is located
cd /mnt/users/goringn/NNs_vs_Kernels/env_dev_1

# Run the Python script using MPI
mpirun -n 30 python modadd_old.py

# Note: Replace modular_addition_nn_mpi.py with the actual name of your Python script if different
