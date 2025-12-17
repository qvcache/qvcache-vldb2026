#!/bin/bash

set -e

# Change to project root
cd "$(dirname "$0")/../../.." || exit 1

# Define variables
DATASET="siftsmall"
DATA_TYPE="float"
DATA_PATH="data/$DATASET/${DATASET}_base.bin"

# Noisy query parameters
N_SPLIT=10
N_SPLIT_REPEAT=5
NOISE_RATIO=0.01

# Construct query and groundtruth paths based on noisy query parameters
# Format noise_ratio to match Python script (remove trailing zeros)
NOISE_STR=$(echo "$NOISE_RATIO" | sed 's/\.0*$//;s/\.$//')
QUERY_PATH="data/$DATASET/${DATASET}_query_nsplit-${N_SPLIT}_nrepeat-${N_SPLIT_REPEAT}_noise-${NOISE_STR}.bin"
GROUNDTRUTH_PATH="data/$DATASET/${DATASET}_groundtruth_nsplit-${N_SPLIT}_nrepeat-${N_SPLIT_REPEAT}_noise-${NOISE_STR}.bin"

# Search parameters
K=10
SEARCH_THREADS=24
METRIC="l2" # Distance metric: "l2", "cosine", or "inner_product"

# Pinecone parameters
INDEX_NAME=$DATASET  # Pinecone index name

# Pinecone configuration
# Your Pinecone API Key (starts with "pcsk_")
# Get this from your Pinecone dashboard: https://app.pinecone.io/
# Can also be set via PINECONE_API_KEY environment variable
PINECONE_API_KEY="pcsk_5A9fQ3_9rJt2c14Ugs1yA99yRzARcNhPqbLxc99S18NVL863yerADFekcqwHf8RxXbtkcW"

# Get API key from environment
if [ -z "$PINECONE_API_KEY" ]; then
    echo "Error: PINECONE_API_KEY environment variable is not set"
    echo "Please set it with: export PINECONE_API_KEY=your_api_key"
    exit 1
fi

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

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Add python directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/python"

echo "=========================================="
echo "Backend-Only Benchmark - Noisy Queries (Pinecone)"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Query file: $QUERY_PATH"
echo "Groundtruth file: $GROUNDTRUTH_PATH"
echo "Noise parameters: n_split=$N_SPLIT, n_repeat=$N_SPLIT_REPEAT, noise_ratio=$NOISE_RATIO"
echo "Pinecone index: $INDEX_NAME"
echo "API key: ${PINECONE_API_KEY:0:10}..."  # Show first 10 chars only
echo "=========================================="
echo ""

# Run the benchmark with all parameters
python3 python/benchmarks/backend_only_benchmark_pinecone_backend.py \
  --data_type "$DATA_TYPE" \
  --data_path "$DATA_PATH" \
  --query_path "$QUERY_PATH" \
  --groundtruth_path "$GROUNDTRUTH_PATH" \
  --index_name "$INDEX_NAME" \
  --api_key "$PINECONE_API_KEY" \
  --K "$K" \
  --search_threads "$SEARCH_THREADS" \
  --n_splits "$N_SPLIT" \
  --n_split_repeat "$N_SPLIT_REPEAT" \
  --metric "$METRIC"

