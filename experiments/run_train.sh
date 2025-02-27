#!/bin/bash

# # Set the MASTER_ADDR and MASTER_PORT environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500  # You can change the port number if needed

# Run the Python training script
CUDA_VISIBLE_DEVICES=0,1,2,3 python overall_train_dist_normal.py