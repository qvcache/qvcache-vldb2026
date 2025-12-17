#!/bin/bash

# Script to reproduce the results (backend-only) of the experiments in Figure 4 in the paper

set -e

# Change to project root
cd "$(dirname "$0")/../.." || exit 1

# Define variables
DATASET="siftsmall"
DATA_TYPE="float"
DATA_PATH="data/$DATASET/${DATASET}_base.bin"

# Noisy query parameters
N_SPLIT=10
N_SPLIT_REPEAT=5
NOISE_RATIO=0.01

# Construct query and groundtruth paths based on noisy query parameters
NOISE_STR=$(echo "$NOISE_RATIO" | sed 's/\.0*$//;s/\.$//')
QUERY_PATH="data/$DATASET/${DATASET}_query_nsplit-${N_SPLIT}_nrepeat-${N_SPLIT_REPEAT}_noise-${NOISE_STR}.bin"
GROUNDTRUTH_PATH="data/$DATASET/${DATASET}_groundtruth_nsplit-${N_SPLIT}_nrepeat-${N_SPLIT_REPEAT}_noise-${NOISE_STR}.bin"

# Backend parameters
R=64
DISK_L=64
K=10
B=8
M=8
SEARCH_THREADS=24
BUILD_THREADS=8
DISK_INDEX_PREFIX="./index/${DATASET}/${DATASET}"
DISK_INDEX_ALREADY_BUILT=1
BEAMWIDTH=2
SECTOR_LEN=4096
METRIC="l2" # Distance metric: "l2", "cosine", or "inner_product"

# Check if query file exists
if [ ! -f "$QUERY_PATH" ]; then
    echo "Error: Query file not found: $QUERY_PATH"
    echo "Please run generate_noisy_queries.sh first to generate the query file."
    exit 1
fi

# Check if groundtruth file exists
if [ ! -f "$GROUNDTRUTH_PATH" ]; then
    echo "Error: Groundtruth file not found: $GROUNDTRUTH_PATH"
    echo "Please run generate_noisy_queries.sh first to generate the groundtruth file."
    exit 1
fi

echo "=========================================="
echo "Backend Benchmark - Noisy Queries"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Query file: $QUERY_PATH"
echo "Groundtruth file: $GROUNDTRUTH_PATH"
echo "Noise parameters: n_split=$N_SPLIT, n_repeat=$N_SPLIT_REPEAT, noise_ratio=$NOISE_RATIO"
echo "=========================================="
echo ""

# Run the benchmark with all parameters
./build/benchmarks/backend_benchmark \
  --data_type "$DATA_TYPE" \
  --data_path "$DATA_PATH" \
  --query_path "$QUERY_PATH" \
  --groundtruth_path "$GROUNDTRUTH_PATH" \
  --disk_index_prefix "$DISK_INDEX_PREFIX" \
  --R "$R" \
  --disk_L "$DISK_L" \
  --K "$K" \
  --B "$B" \
  --M "$M" \
  --search_threads "$SEARCH_THREADS" \
  --build_threads "$BUILD_THREADS" \
  --disk_index_already_built "$DISK_INDEX_ALREADY_BUILT" \
  --beamwidth "$BEAMWIDTH" \
  --n_splits "$N_SPLIT" \
  --n_split_repeat "$N_SPLIT_REPEAT" \
  --sector_len "$SECTOR_LEN" \
  --metric "$METRIC"

