#!/usr/bin/env python3
"""
Backend-only benchmark using Pinecone backend (without QVCache)

This script benchmarks the Pinecone backend directly, processing splits sequentially.
"""

import argparse
import numpy as np
import json
import sys
import os

try:
    import qvcache as qvc
except ImportError:
    print("Error: qvcache module not found. Please build the Python bindings first.")
    print("Run: cd build && cmake .. && make")
    sys.exit(1)

from backends.pinecone_backend import PineconeBackend
from benchmarks.utils import backend_search, calculate_backend_recall, log_backend_split_metrics


def experiment_benchmark(
    data_path: str,
    query_path: str,
    groundtruth_path: str,
    K: int,
    search_threads: int,
    n_splits: int,
    n_split_repeat: int,
    backend: PineconeBackend,
    index_name: str,
    api_key: str = None,
    environment: str = None
):
    """Run the backend-only benchmark experiment with Pinecone backend."""
    # Load ground truth
    groundtruth_ids, groundtruth_dists = qvc.load_ground_truth_data(groundtruth_path)
    n_groundtruth, groundtruth_dim = groundtruth_ids.shape

    # Load queries
    queries, query_dim, query_aligned_dim = qvc.load_aligned_binary_data(query_path)
    query_num = queries.shape[0]
    
    # Ensure queries are float32
    queries = queries.astype(np.float32)
    
    # Query file structure: all copies of split 0, then all copies of split 1, etc.
    # Each split has n_split_repeat copies, and each copy has queries_per_original_split queries
    queries_per_original_split = query_num // (n_splits * n_split_repeat)
    
    # Process splits one by one sequentially
    for split_idx in range(n_splits):
        print(json.dumps({
            "event": "split_start",
            "split_idx": split_idx
        }))
        
        # Process all copies for this split
        for copy_idx in range(n_split_repeat):
            # Calculate query range for this specific copy of this split
            # Structure: split 0 (all copies), split 1 (all copies), ...
            # For split i, copy j: offset = i * (n_split_repeat * queries_per_original_split) + j * queries_per_original_split
            split_offset = split_idx * n_split_repeat * queries_per_original_split
            copy_offset = copy_idx * queries_per_original_split
            query_start = split_offset + copy_offset
            query_end = min(query_start + queries_per_original_split, query_num)
            
            if query_start >= query_end:
                continue
            
            this_split_size = query_end - query_start
            split_queries = queries[query_start:query_end]
            
            query_result_tags, metrics = backend_search(
                backend,
                split_queries,
                K,
                search_threads
            )
            
            # Calculate groundtruth offset (same structure as queries)
            gt_start = split_offset + copy_offset
            recall_all = calculate_backend_recall(
                K, groundtruth_ids[gt_start:gt_start + this_split_size], 
                query_result_tags, this_split_size, groundtruth_dim
            )
            
            log_backend_split_metrics(metrics, recall_all, split_idx=split_idx)
        
        print(json.dumps({
            "event": "split_end",
            "split_idx": split_idx
        }))


def main():
    parser = argparse.ArgumentParser(description="Backend-only benchmark experiment with Pinecone backend")
    parser.add_argument("--data_path", type=str, required=True, help="Path to base data file")
    parser.add_argument("--query_path", type=str, required=True, help="Path to query data file")
    parser.add_argument("--groundtruth_path", type=str, required=True, help="Path to groundtruth file")
    parser.add_argument("--index_name", type=str, required=True, help="Pinecone index name")
    parser.add_argument("--api_key", type=str, default=None, help="Pinecone API key (or use PINECONE_API_KEY env var)")
    parser.add_argument("--environment", type=str, default=None, help="Pinecone environment (deprecated, not used)")
    
    # Search parameters
    parser.add_argument("--K", type=int, default=100, help="Number of nearest neighbors")
    parser.add_argument("--search_threads", type=int, default=24, help="Search threads")
    parser.add_argument("--n_splits", type=int, required=True, help="Number of splits for queries")
    parser.add_argument("--n_split_repeat", type=int, required=True, help="Number of repeats per split pattern")
    parser.add_argument("--data_type", type=str, default="float", help="Data type")
    parser.add_argument("--metric", type=str, default="l2", choices=["l2", "cosine", "inner_product"],
                       help="Distance metric: l2, cosine, or inner_product")
    
    args = parser.parse_args()
    
    # Get API key from environment if not provided
    api_key = args.api_key or os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("Error: Pinecone API key must be provided via --api_key or PINECONE_API_KEY environment variable", file=sys.stderr)
        sys.exit(1)
    
    # Print parameters
    params = {
        "event": "params",
        "backend": "Pinecone",
        "index_name": args.index_name,
        "data_type": args.data_type,
        "data_path": args.data_path,
        "query_path": args.query_path,
        "groundtruth_path": args.groundtruth_path,
        "K": args.K,
        "search_threads": args.search_threads,
        "n_splits": args.n_splits,
        "n_split_repeat": args.n_split_repeat,
        "metric": args.metric
    }
    print(json.dumps(params))
    
    # Get vector dimension from data file
    with open(args.data_path, 'rb') as f:
        num_vectors = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        dimension = np.frombuffer(f.read(4), dtype=np.uint32)[0]
    
    # Create Pinecone backend (assumes index already exists)
    print(f"\nConnecting to Pinecone index {args.index_name}...", file=sys.stderr)
    backend = PineconeBackend(
        index_name=args.index_name,
        dimension=dimension,
        api_key=api_key,
        environment=args.environment,
        data_path=args.data_path,
        recreate_index=False
    )
    
    # Run experiment
    experiment_benchmark(
        args.data_path, args.query_path, args.groundtruth_path,
        args.K, args.search_threads,
        args.n_splits, args.n_split_repeat,
        backend, args.index_name, api_key, args.environment
    )


if __name__ == "__main__":
    main()

